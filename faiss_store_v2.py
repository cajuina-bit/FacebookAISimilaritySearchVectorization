import os
import logging
import json
import numpy as np
import faiss
import spacy
import re
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

def sanitize_text(text):
    """
    Clean text while preserving markdown code blocks and handling Unicode characters.
    This utility function ensures text is free from problematic Unicode characters
    that might cause encoding issues when displaying results.
    
    Args:
        text: Input text that may contain Unicode characters
        
    Returns:
        Sanitized text with Unicode characters converted to ASCII equivalents
    """
    if not text:
        return ""
        
    # Handle non-string inputs
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logger.error(f"Error converting to string in sanitize_text: {e}")
            return "Text content unavailable"

    # Temporarily replace code blocks with placeholders to protect them
    code_blocks = []
    code_pattern = r'```(?:\w*\n)?[\s\S]*?```'

    def replace_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    # Replace code blocks with placeholders
    text = re.sub(code_pattern, replace_code, text)

    # List of common problematic Unicode characters to replace
    problematic_chars = {
        # Arrows and symbols found in FAQs and error messages
        '\u2192': '->',       # right arrow →
        '\u2190': '<-',       # left arrow ←
        '\u2194': '<->',      # left-right arrow ↔
        '\u2191': '^',        # up arrow ↑
        '\u2193': 'v',        # down arrow ↓
        '\u21d2': '=>',       # rightwards double arrow ⇒
        '\u21d0': '<=',       # leftwards double arrow ⇐
        '\u21e8': '->',       # rightwards white arrow ⇨
        '\u2794': '->',       # heavy wide-headed rightwards arrow ➔
        
        # Mathematical symbols
        '\u2261': '=',        # identical to ≡
        '\u2248': '~=',       # almost equal to ≈
        '\u2260': '!=',       # not equal to ≠
        '\u2264': '<=',       # less than or equal to ≤
        '\u2265': '>=',       # greater than or equal to ≥
        '\u00b1': '+/-',      # plus-minus ±
        '\u221e': 'inf',      # infinity ∞
        '\u2211': 'sum',      # n-ary summation ∑
        '\u220f': 'prod',     # n-ary product ∏
        '\u222b': 'int',      # integral ∫
        
        # Punctuation
        '\u2022': '*',        # bullet •
        '\u2023': '>',        # triangular bullet ‣
        '\u2043': '-',        # hyphen bullet ⁃
        '\u2013': '-',        # en dash –
        '\u2014': '--',       # em dash —
        '\u2026': '...',      # horizontal ellipsis …
        '\u201c': '"',        # left double quote "
        '\u201d': '"',        # right double quote "
        '\u2018': "'",        # left single quote '
        '\u2019': "'",        # right single quote '
        
        # Symbols
        '\u00a9': '(c)',      # copyright ©
        '\u00ae': '(R)',      # registered trademark ®
        '\u2122': 'TM',       # trademark ™
        '\u25cf': '*',        # black circle ●
        '\u2714': 'v',        # heavy check mark ✔
        '\u2713': 'v',        # check mark ✓
        '\u2716': 'x',        # heavy multiplication x ✖
        '\u25a0': '[]',       # black square ■
        '\u25a1': '[]',       # white square □
        '\u25b2': '^',        # black up-pointing triangle ▲
        '\u25bc': 'v',        # black down-pointing triangle ▼
        
        # Technical symbols
        '\u2329': '<',        # left-pointing angle bracket ⟨
        '\u232a': '>',        # right-pointing angle bracket ⟩
    }

    # Replace known problematic characters
    for char, replacement in problematic_chars.items():
        if char in text:
            text = text.replace(char, replacement)

    # For any remaining non-ASCII characters, try to normalize or replace
    normalized = []
    for char in text:
        # If character is ASCII, keep it as is
        if ord(char) < 128:
            normalized.append(char)
        else:
            # Try to normalize non-ASCII characters to ASCII equivalents
            try:
                # NFKD normalization + ASCII encoding will handle many common cases
                normalized_char = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('ascii')
                if normalized_char:
                    normalized.append(normalized_char)
                else:
                    # If normalization doesn't produce an ASCII character, replace with a space
                    normalized.append(' ')
            except Exception as e:
                # Log the error with more context
                logger.warning(f"Error normalizing character '{char}' (code point {ord(char)}): {e}")
                normalized.append(' ')

    text = ''.join(normalized)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        # Also sanitize the code blocks using the same logic
        sanitized_block = []
        for char in block:
            if ord(char) < 128:
                sanitized_block.append(char)
            else:
                try:
                    norm_char = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('ascii')
                    if norm_char:
                        sanitized_block.append(norm_char)
                    else:
                        sanitized_block.append(' ')
                except:
                    sanitized_block.append(' ')
        
        sanitized_block = ''.join(sanitized_block)
        text = text.replace(f"__CODE_BLOCK_{i}__", sanitized_block)

    # Clean up any multiple spaces that may have been created
    text = re.sub(r'\s+', ' ', text)
    
    # Final safety check - ensure the output is 100% ASCII
    if not text.isascii():
        logger.warning("Text still contains non-ASCII characters after sanitization, forcing ASCII encoding")
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


class FAISSVectorStore:
    """
    A FAISS vector search engine implementation for document retrieval.
    
    This class provides document indexing, vector search, BM25 scoring, and hybrid ranking
    capabilities to efficiently retrieve relevant documents from a corpus.
    """

    def __init__(self, store_dir: str = "../.vector_stores", dim: int = 300,
                 model_name: str = "en_core_web_md", k1: float = 1.5, b: float = 0.75):
        """
        Initialize the FAISS search engine with parameters.
        
        Args:
            store_dir: Directory to store FAISS indices
            dim: Vector dimension size
            model_name: Name of the spaCy model to use
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
            
        Example:
            >>> store = FAISSVectorStore(store_dir="/app/vector_stores", dim=300)
        """
        self.store_dir = store_dir
        self.dimension = dim
        self.nlp = spacy.load(model_name)
        self.index_cache = {}
        self.metadata_cache = {}
        self.content_cache = {}
        self.title_cache = {}
        self.term_freq_cache = {}
        self.doc_len_cache = {}
        self.avg_doc_len_cache = {}
        self.idf_cache = {}

        self.k1 = k1
        self.b = b

        self.vector_weight = 0.3
        self.bm25_weight = 0.6
        self.contextual_weight = 0.15
        self.section_weight = 0.05

        os.makedirs(store_dir, exist_ok=True)
        self.debug = True

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching by converting to lowercase and removing extra whitespace.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text string
            
        Example:
            >>> store.preprocess_text("  Hello   WORLD  ")
            "hello world"
        """
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words using spaCy, filtering out stopwords and punctuation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
            
        Example:
            >>> store.tokenize("How to configure AWS services?")
            ['configure', 'aws', 'services']
        """
        text = self.preprocess_text(text)
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def vectorize_text(self, text: str, idf_dict: Dict[str, float] = None) -> np.ndarray:
        """
        Convert text to vector using TF-IDF weighted token vectors with linguistic features.
        
        Args:
            text: Input text to vectorize
            idf_dict: Dictionary of inverse document frequencies
            
        Returns:
            Normalized vector representation of the text
            
        Example:
            >>> vector = store.vectorize_text("How to configure Kubernetes?")
            >>> vector.shape
            (300,)
        """
        text = self.preprocess_text(text)

        if not text or len(text) < 3:
            np.random.seed(0)
            vector = np.random.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vector)
            return vector / max(norm, 1e-10)

        doc = self.nlp(text[:5000])

        text_lower = text.lower()
        boilerplate_phrases = [
            "for more information",
            "contact us",
            "please refer to",
            "click here",
            "visit our website",
            "copyright",
            "all rights reserved"
        ]

        has_boilerplate = any(phrase in text_lower for phrase in boilerplate_phrases)
        if has_boilerplate and len(text) > 200:
            logger.debug(f"Detected boilerplate in text: {text[:100]}...")

        term_freq = Counter()
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                lemma = token.lemma_.lower()
                term_freq[lemma] += 1

        max_freq = max(term_freq.values()) if term_freq else 1

        token_vectors = []
        token_weights = []

        processed_lemmas = set()

        for token in doc:
            if (token.is_stop or token.is_punct or len(token.text) <= 1 or
                    not token.has_vector or token.lemma_.lower() in processed_lemmas):
                continue

            lemma = token.lemma_.lower()
            processed_lemmas.add(lemma)

            if lemma in ['click', 'link', 'page', 'home', 'contact', 'copyright'] and idf_dict:
                if lemma in idf_dict and idf_dict[lemma] < 2.0:
                    continue

            weight = 1.0

            if token.pos_ in ['NOUN', 'PROPN']:
                weight *= 2.5
            elif token.pos_ in ['VERB']:
                weight *= 1.5
            elif token.pos_ in ['ADJ', 'ADV']:
                weight *= 1.2

            if token.ent_type_:
                if token.ent_type_ in ['ORG', 'PRODUCT', 'GPE', 'LOC', 'PERSON']:
                    weight *= 2.0
                else:
                    weight *= 1.5

            tf = term_freq[lemma]
            norm_tf = 0.5 + 0.5 * (math.log(1 + tf) / math.log(1 + max_freq))
            weight *= norm_tf

            if idf_dict and lemma in idf_dict:
                idf_value = min(idf_dict[lemma], 10.0)
                weight *= idf_value

            position = token.i / len(doc)
            if position < 0.2:
                weight *= 1.3

            token_vectors.append(token.vector)
            token_weights.append(weight)

        if token_vectors:
            token_vectors = np.array(token_vectors)
            token_weights = np.array(token_weights)

            if sum(token_weights) > 0:
                token_weights = token_weights / sum(token_weights)
                vector = np.average(token_vectors, axis=0, weights=token_weights).astype(np.float32)
            else:
                vector = np.mean(token_vectors, axis=0).astype(np.float32)
        else:
            if doc.has_vector:
                vector = doc.vector.astype(np.float32)
            else:
                text_hash = hash(text) & 0xffffffff
                np.random.seed(text_hash % 10000)
                vector = np.random.randn(self.dimension).astype(np.float32)

        text_hash = hash(text) & 0xffffffff
        np.random.seed(text_hash % 10000)
        noise = np.random.randn(self.dimension).astype(np.float32) * 0.001
        vector = vector + noise

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            text_hash = hash(text) & 0xffffffff
            np.random.seed(text_hash % 10000)
            vector = np.random.randn(self.dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)

        return vector

    def create_store(self, name: str, texts: List[Dict[str, Any]]) -> None:
        """
        Create a FAISS index for a set of texts with duplicate detection.
        
        Args:
            name: Name of the vector store to create
            texts: List of text items, each with 'content' and optional 'title' fields
            
        Example:
            >>> texts = [
            ...     {"title": "How to use Docker", "content": "Docker is a containerization platform..."},
            ...     {"title": "Kubernetes basics", "content": "Kubernetes is an orchestration system..."}
            ... ]
            >>> store.create_store("tech_docs", texts)
        """
        if not texts:
            logger.warning(f"No texts provided for store {name}")
            return

        vectors = []
        metadata = []
        content_texts = []
        title_texts = []
        term_frequencies = []
        doc_lengths = []
        original_indices = []

        all_terms = set()
        term_doc_frequencies = Counter()

        for item in texts:
            if 'content' not in item or not item['content']:
                continue

            text = item['content']
            processed_text = self.preprocess_text(text)
            tokens = self.tokenize(text)

            all_terms.update(tokens)
            for term in set(tokens):
                term_doc_frequencies[term] += 1

        num_docs = len(texts)
        idf_dict = {}
        for term in all_terms:
            df = term_doc_frequencies[term]
            idf = math.log((num_docs + 1) / (df + 1)) + 1
            idf_dict[term] = idf

        content_hashes = {}

        for i, item in enumerate(texts):
            if 'content' not in item or not item['content']:
                continue

            text = item['content']
            title = item.get('title', '')

            content_hash = hash(text.lower())
            if content_hash in content_hashes:
                dup_idx = content_hashes[content_hash]
                logger.warning(
                    f"Duplicate content detected: item {i} ('{title}') is identical to item {dup_idx} ('{texts[dup_idx].get('title', '')}')")
                continue

            content_hashes[content_hash] = i

            processed_text = self.preprocess_text(text)
            content_texts.append(processed_text)

            processed_title = self.preprocess_text(title)
            title_texts.append(processed_title)

            tokens = self.tokenize(text)
            token_counter = Counter(tokens)
            term_frequencies.append(token_counter)
            doc_lengths.append(len(tokens))

            vector = self.vectorize_text(text, idf_dict)

            if np.all(np.abs(vector) < 1e-10):
                logger.warning(f"Skipping text with zero vector: {text[:50]}")
                continue

            vectors.append(vector)
            original_indices.append(i)

            meta_item = item.copy()
            meta_item['id'] = len(vectors) - 1
            metadata.append(meta_item)

        if not vectors:
            logger.warning(f"No valid vectors created for store {name}")
            return

        vectors_np = np.array(vectors).astype('float32')

        norms = np.linalg.norm(vectors_np, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5):
            logger.warning(f"Vectors not properly normalized. Norms: min={norms.min():.6f}, max={norms.max():.6f}")
            vectors_np = vectors_np / np.maximum(norms[:, np.newaxis], 1e-10)

        if len(vectors) > 1:
            max_vectors_for_pairwise = 1000

            if len(vectors) <= max_vectors_for_pairwise:
                dot_products = np.dot(vectors_np, vectors_np.T)
                np.fill_diagonal(dot_products, 0)

                max_sim = np.max(dot_products)
                if max_sim > 0.98:
                    i, j = np.unravel_index(np.argmax(dot_products), dot_products.shape)
                    title_i = metadata[i].get('title', f'item_{original_indices[i]}')
                    title_j = metadata[j].get('title', f'item_{original_indices[j]}')

                    logger.warning(
                        f"Near-duplicate vectors in '{name}': '{title_i}' vs '{title_j}' (sim={max_sim:.4f})")

                    high_sim_indices = np.where(dot_products > 0.95)
                    if len(high_sim_indices[0]) > 2:
                        logger.warning(f"Found {len(high_sim_indices[0])} near-duplicate pairs with similarity > 0.95")
                        avg_sim = np.sum(dot_products) / (len(vectors) ** 2 - len(vectors))
                        logger.info(f"Average inter-document similarity in store '{name}': {avg_sim:.3f}")
            else:
                logger.info(
                    f"Skipping detailed duplicate check for {name} as it has {len(vectors)} vectors (> {max_vectors_for_pairwise})")

        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors_np)

        index_path = os.path.join(self.store_dir, f"{name}.index")
        faiss.write_index(index, index_path)

        metadata_path = os.path.join(self.store_dir, f"{name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        bm25_path = os.path.join(self.store_dir, f"{name}.bm25.json")
        bm25_data = {
            "term_frequencies": [dict(tf) for tf in term_frequencies],
            "doc_lengths": doc_lengths,
            "avg_doc_len": avg_doc_len,
            "idf": idf_dict
        }
        with open(bm25_path, 'w') as f:
            json.dump(bm25_data, f)

        self.term_freq_cache[name] = term_frequencies
        self.doc_len_cache[name] = doc_lengths
        self.avg_doc_len_cache[name] = avg_doc_len
        self.idf_cache[name] = idf_dict
        self.content_cache[name] = content_texts
        self.title_cache[name] = title_texts

        logger.info(f"Created vector store {name} with {len(vectors)} vectors and BM25 data")

    def load_store(self, name: str) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
        """
        Load a FAISS index and its metadata.
        
        Args:
            name: Name of the vector store to load
            
        Returns:
            Tuple of (index, metadata) where index is the FAISS index and metadata is the associated document data
            
        Example:
            >>> index, metadata = store.load_store("tech_docs")
            >>> len(metadata)
            42
        """
        if name in self.index_cache and name in self.metadata_cache:
            return self.index_cache[name], self.metadata_cache[name]

        index_path = os.path.join(self.store_dir, f"{name}.index")
        if not os.path.exists(index_path):
            logger.warning(f"Index file {index_path} not found")
            return None, None

        index = faiss.read_index(index_path)

        metadata_path = os.path.join(self.store_dir, f"{name}.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file {metadata_path} not found")
            return index, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.index_cache[name] = index
        self.metadata_cache[name] = metadata

        self._load_bm25_data(name, metadata)

        return index, metadata

    def _load_bm25_data(self, name: str, metadata: List[Dict[str, Any]]) -> bool:
        """
        Load BM25 data from file or generate it if not available.
        
        Args:
            name: Name of the vector store
            metadata: Metadata for the vector store
            
        Returns:
            Boolean indicating if BM25 data was successfully loaded
            
        Example:
            >>> success = store._load_bm25_data("tech_docs", metadata)
            >>> success
            True
        """
        if (name in self.term_freq_cache and name in self.doc_len_cache and
                name in self.avg_doc_len_cache and name in self.idf_cache):
            return True

        bm25_path = os.path.join(self.store_dir, f"{name}.bm25.json")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'r') as f:
                    bm25_data = json.load(f)

                term_frequencies = [Counter(tf) for tf in bm25_data["term_frequencies"]]
                doc_lengths = bm25_data["doc_lengths"]
                avg_doc_len = bm25_data["avg_doc_len"]
                idf_dict = bm25_data["idf"]

                self.term_freq_cache[name] = term_frequencies
                self.doc_len_cache[name] = doc_lengths
                self.avg_doc_len_cache[name] = avg_doc_len
                self.idf_cache[name] = idf_dict

                if name not in self.content_cache or name not in self.title_cache:
                    content_texts = []
                    title_texts = []
                    for item in metadata:
                        if 'content' in item and item['content']:
                            content_texts.append(self.preprocess_text(item['content']))
                        else:
                            content_texts.append("")

                        if 'title' in item and item['title']:
                            title_texts.append(self.preprocess_text(item['title']))
                        else:
                            title_texts.append("")

                    self.content_cache[name] = content_texts
                    self.title_cache[name] = title_texts

                logger.info(f"Loaded BM25 data for store {name}")
                return True
            except Exception as e:
                logger.warning(f"Error loading BM25 data for {name}: {e}")

        logger.info(f"Generating BM25 data for store {name}")
        try:
            content_texts = []
            title_texts = []
            for item in metadata:
                if 'content' in item and item['content']:
                    content_texts.append(self.preprocess_text(item['content']))
                else:
                    content_texts.append("")

                if 'title' in item and item['title']:
                    title_texts.append(self.preprocess_text(item['title']))
                else:
                    title_texts.append("")

            self.content_cache[name] = content_texts
            self.title_cache[name] = title_texts

            term_frequencies = []
            doc_lengths = []
            all_terms = set()
            term_doc_frequencies = Counter()

            for i, item in enumerate(metadata):
                content = item.get('content', '')
                tokens = self.tokenize(content)
                token_counter = Counter(tokens)

                all_terms.update(tokens)
                for term in set(tokens):
                    term_doc_frequencies[term] += 1

                term_frequencies.append(token_counter)
                doc_lengths.append(len(tokens))

            avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

            num_docs = len(doc_lengths)
            idf_dict = {}
            for term in all_terms:
                idf = math.log((num_docs + 1) / (term_doc_frequencies[term] + 1)) + 1
                idf_dict[term] = idf

            self.term_freq_cache[name] = term_frequencies
            self.doc_len_cache[name] = doc_lengths
            self.avg_doc_len_cache[name] = avg_doc_len
            self.idf_cache[name] = idf_dict

            bm25_data = {
                "term_frequencies": [dict(tf) for tf in term_frequencies],
                "doc_lengths": doc_lengths,
                "avg_doc_len": avg_doc_len,
                "idf": idf_dict
            }

            with open(bm25_path, 'w') as f:
                json.dump(bm25_data, f)

            logger.info(f"Generated and saved BM25 data for store {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate BM25 data for {name}: {e}")
            return False

    def keyword_match(self, query: str, store_name: str, title_only: bool = False) -> Dict[int, float]:
        """
        Perform exact keyword matching that identifies and scores direct matches.
        
        Args:
            query: Search query
            store_name: Name of the vector store
            title_only: Whether to search only in titles
            
        Returns:
            Dictionary mapping document indices to match scores
            
        Example:
            >>> matches = store.keyword_match("kubernetes installation", "tech_docs")
            >>> matches
            {3: 0.65, 12: 0.45, 27: 0.2}
        """
        if store_name not in self.content_cache or store_name not in self.title_cache:
            _, metadata = self.load_store(store_name)
            if not metadata:
                return {}
        else:
            _, metadata = self.load_store(store_name)

        content_texts = self.content_cache[store_name]
        title_texts = self.title_cache[store_name]

        query = self.preprocess_text(query)
        if not query:
            return {}

        matches = {}

        try:
            query_doc = self.nlp(query)

            query_terms = []
            for token in query_doc:
                if token.is_stop or token.is_punct or len(token.text) < 3:
                    continue
                if token.text.lower() not in query_terms:
                    query_terms.append(token.text.lower())
                if token.lemma_.lower() not in query_terms:
                    query_terms.append(token.lemma_.lower())

            for chunk in query_doc.noun_chunks:
                if len(chunk.text) >= 3 and chunk.text.lower() not in query_terms:
                    query_terms.append(chunk.text.lower())

            raw_query_lower = query.lower()

        except Exception as e:
            logger.warning(f"Error processing query with spaCy: {e}")
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]
            raw_query_lower = query.lower()

        if not query_terms:
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]

        for i in range(len(content_texts)):
            if i >= len(metadata):
                continue

            score = 0.0

            if title_only:
                if i >= len(title_texts) or not title_texts[i]:
                    continue
                text = title_texts[i].lower()
                text_for_raw_match = title_texts[i]
            else:
                if i >= len(content_texts) or not content_texts[i]:
                    continue
                text = content_texts[i].lower()
                title_text = title_texts[i].lower() if i < len(title_texts) and title_texts[i] else ""
                text_for_raw_match = content_texts[i]

            matched_terms = []

            if raw_query_lower in text:
                score += 0.8
                matched_terms.append("FULL_QUERY")

            for term in query_terms:
                if term in text:
                    term_score = min(0.2, 0.05 * text.count(term))
                    score += term_score
                    matched_terms.append(term)

            if not title_only and title_text:
                title_matches = sum(1 for term in query_terms if term in title_text)
                if title_matches > 0:
                    title_score = min(0.5, 0.15 * title_matches)
                    score += title_score
                    matched_terms.append(f"TITLE({title_matches})")

                if raw_query_lower in title_text:
                    score += 0.5
                    matched_terms.append("FULL_QUERY_IN_TITLE")

            if 'raw_data' in metadata[i]:
                raw_data = metadata[i]['raw_data']
                if 'Issue' in raw_data:
                    issue_text = raw_data['Issue'].lower()
                    issue_matches = sum(1 for term in query_terms if term in issue_text)
                    if issue_matches > 0:
                        issue_score = min(0.4, 0.12 * issue_matches)
                        score += issue_score
                        matched_terms.append(f"ISSUE({issue_matches})")

                    if raw_query_lower in issue_text:
                        score += 0.6
                        matched_terms.append("FULL_QUERY_IN_ISSUE")

            if score > 0:
                matches[i] = min(1.0, score)
                if self.debug:
                    logger.debug(f"Keyword match for doc {i}: score={score:.3f}, matches={matched_terms}")

        return matches

    def calculate_bm25_scores(self, query: str, store_name: str) -> Dict[int, float]:
        """
        Calculate BM25 scores based on term importance and linguistic features.
        
        Args:
            query: Search query
            store_name: Name of the vector store
            
        Returns:
            Dictionary mapping document indices to BM25 scores
            
        Example:
            >>> scores = store.calculate_bm25_scores("configure aws permissions", "cloud_docs")
            >>> scores
            {5: 0.78, 13: 0.65, 42: 0.43}
        """
        if store_name not in self.term_freq_cache:
            _, metadata = self.load_store(store_name)
            if metadata and store_name not in self.term_freq_cache:
                success = self._load_bm25_data(store_name, metadata)
                if not success:
                    logger.warning(f"BM25 scoring not available for {store_name}")
                    return {}

        term_frequencies = self.term_freq_cache[store_name]
        doc_lengths = self.doc_len_cache[store_name]
        avg_doc_len = self.avg_doc_len_cache[store_name]
        idf_dict = self.idf_cache[store_name]

        query_terms = []
        term_importance = {}

        try:
            query_doc = self.nlp(query)

            for chunk in query_doc.noun_chunks:
                chunk_text = chunk.text.lower()
                query_terms.append(chunk_text)
                term_importance[chunk_text] = 2.0

            for token in query_doc:
                if token.is_stop or token.is_punct or len(token.text) < 3:
                    continue

                lemma = token.lemma_.lower()
                if lemma in query_terms:
                    continue

                query_terms.append(lemma)

                if token.ent_type_:
                    term_importance[lemma] = 2.0
                elif token.pos_ in ['NOUN', 'PROPN']:
                    term_importance[lemma] = 1.8
                elif token.pos_ in ['VERB']:
                    term_importance[lemma] = 1.5
                elif token.pos_ in ['ADJ', 'ADV']:
                    term_importance[lemma] = 1.3
                else:
                    term_importance[lemma] = 1.0

            for i, token in enumerate(query_doc):
                if i < 3 and token.lemma_.lower() in term_importance:
                    term_importance[token.lemma_.lower()] *= 1.2

        except Exception as e:
            logger.warning(f"Error processing query with spaCy: {e}")
            query_terms = [term.lower() for term in query.split() if len(term) >= 3]
            for term in query_terms:
                term_importance[term] = 1.0

        scores = {}
        for i, doc_tf in enumerate(term_frequencies):
            score = 0.0
            doc_len = doc_lengths[i]

            if doc_len == 0:
                continue

            matched_terms = []
            matched_weights = []

            for term in query_terms:
                if ' ' in term:
                    term_parts = term.split()
                    term_score = 0

                    if term in idf_dict:
                        tf = doc_tf.get(term, 0)
                        importance = term_importance.get(term, 1.0)

                        full_term_score = (
                                idf_dict[term] *
                                ((tf * (self.k1 + 1)) /
                                 (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))) *
                                importance
                        )
                        term_score += full_term_score

                        if full_term_score > 0:
                            matched_terms.append(term)
                            matched_weights.append(importance)

                    for part in term_parts:
                        if part in idf_dict:
                            tf = doc_tf.get(part, 0)
                            part_importance = term_importance.get(term, 1.0) * 0.7

                            part_score = (
                                    idf_dict[part] *
                                    ((tf * (self.k1 + 1)) /
                                     (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))) *
                                    part_importance
                            )
                            term_score += part_score / len(term_parts)

                            if part_score > 0:
                                matched_terms.append(part)
                                matched_weights.append(part_importance)

                    score += term_score
                else:
                    if term not in idf_dict:
                        continue

                    tf = doc_tf.get(term, 0)
                    importance = term_importance.get(term, 1.0)

                    term_score = (
                            idf_dict[term] *
                            ((tf * (self.k1 + 1)) /
                             (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))) *
                            importance
                    )
                    score += term_score

                    if term_score > 0:
                        matched_terms.append(term)
                        matched_weights.append(importance)

            if query_terms and matched_terms:
                unique_matched = len(set(matched_terms))
                coverage = unique_matched / len(query_terms)

                if coverage > 0.7:
                    coverage_bonus = coverage * 0.2
                    score += coverage_bonus

            if score > 0:
                scores[i] = min(1.0, score / 20.0)

                if self.debug and scores[i] > 0.5:
                    logger.debug(f"BM25 high score for doc {i}: {scores[i]:.3f}, matched terms: {matched_terms}")

        return scores

    def expand_query(self, query: str, expansion_terms: int = 2) -> str:
        """
        Expand the query with semantically similar terms to improve retrieval.
        
        Args:
            query: The original search query
            expansion_terms: Number of similar terms to add per important term
            
        Returns:
            Expanded query string with additional related terms
            
        Example:
            >>> store.expand_query("kubernetes deployment")
            "kubernetes deployment orchestration container k8s"
        """
        query = self.preprocess_text(query)
        if not query:
            return query

        doc = self.nlp(query)

        important_terms = [token for token in doc if not token.is_stop and not token.is_punct]

        expanded_terms = []

        for term in important_terms:
            if not term.has_vector:
                continue

            similar_terms = []
            sample_size = 1000
            sample_count = 0

            for lexeme in self.nlp.vocab:
                if sample_count > sample_size:
                    break

                if lexeme.has_vector and lexeme.is_alpha and len(lexeme.text) > 2 and not lexeme.is_stop:
                    sample_count += 1
                    similarity = lexeme.similarity(term)
                    if similarity > 0.7 and similarity < 0.99:
                        similar_terms.append((lexeme.text, similarity))

            sorted_similar = sorted(similar_terms, key=lambda x: x[1], reverse=True)
            for similar_term, _ in sorted_similar[:expansion_terms]:
                if similar_term not in query and similar_term not in expanded_terms:
                    expanded_terms.append(similar_term)

        expanded_query = query
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms)
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")

        return expanded_query

    def calculate_contextual_similarity(self, query: str, text: str) -> float:
        """
        Calculate contextual similarity between query and document text.
        
        Args:
            query: The search query
            text: The document text to compare against
            
        Returns:
            A float similarity score between 0 and 1
            
        Example:
            >>> store.calculate_contextual_similarity("configure aws s3", "How to set up AWS S3 buckets")
            0.82
        """
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length]

        query = self.preprocess_text(query)
        text = self.preprocess_text(text)

        if not query or not text:
            return 0.0

        try:
            query_doc = self.nlp(query)
            text_doc = self.nlp(text)

            if not query_doc.has_vector or not text_doc.has_vector:
                return 0.0

            doc_similarity = query_doc.similarity(text_doc)

            query_terms = set()
            text_terms = set()

            for token in query_doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    query_terms.add(token.lemma_.lower())

            for token in text_doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    text_terms.add(token.lemma_.lower())

            term_overlap = 0.0
            if query_terms:
                term_overlap = len(query_terms.intersection(text_terms)) / len(query_terms)

            query_entities = set(ent.text.lower() for ent in query_doc.ents)
            text_entities = set(ent.text.lower() for ent in text_doc.ents)

            entity_overlap = 0.0
            if query_entities:
                entity_overlap = len(query_entities.intersection(text_entities)) / len(query_entities)

            if len(query.split()) > 5:
                contextual_similarity = 0.4 * doc_similarity + 0.4 * term_overlap + 0.2 * entity_overlap
            else:
                contextual_similarity = 0.3 * doc_similarity + 0.5 * term_overlap + 0.2 * entity_overlap

            if len(query) > 10 and query.lower() in text.lower():
                contextual_similarity = max(contextual_similarity, 0.9)

            query_lower = query.lower()
            if ('how' in query_lower or 'what' in query_lower) and 'how' in text.lower():
                contextual_similarity += 0.1

            return max(0.0, min(1.0, contextual_similarity))

        except Exception as e:
            logger.warning(f"Error calculating contextual similarity: {e}")
            return 0.0

    def calculate_section_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """
        Calculate relevance based on document structure and sections.
        
        Args:
            query: The search query
            document: Document metadata including title and type
            
        Returns:
            A relevance score for document structure
            
        Example:
            >>> doc = {"title": "AWS Security Best Practices", "type": "section"}
            >>> store.calculate_section_relevance("aws security", doc)
            0.6
        """
        base_score = 0.0

        if 'title' in document and document['title']:
            title = self.preprocess_text(document['title'])
            query_terms = self.tokenize(query)

            title_matches = sum(1 for term in query_terms if term in title)
            if title_matches > 0:
                title_score = min(0.5, 0.15 * title_matches)
                base_score += title_score

        doc_type = document.get('type', '')
        if doc_type == 'section':
            base_score += 0.2
        elif doc_type == 'table_row':
            base_score += 0.1

        return base_score

    def enhanced_keyword_match(self, query: str, store_name: str, title_only: bool = False) -> Dict[int, float]:
        """
        Enhanced keyword matching that better handles error codes without affecting FAQ matching.
        
        Args:
            query: Search query
            store_name: Name of the vector store
            title_only: Whether to search only in titles
            
        Returns:
            Dictionary mapping document indices to match scores
        """
        # Run the original keyword match first
        regular_matches = self.keyword_match(query, store_name, title_only)
        
        # If we're not searching in the error codes store, return regular matches
        if 'error_code' not in store_name.lower():
            return regular_matches
        
        # Get content and title data
        if store_name not in self.content_cache or store_name not in self.title_cache:
            _, metadata = self.load_store(store_name)
            if not metadata:
                return regular_matches
        else:
            _, metadata = self.load_store(store_name)
        
        content_texts = self.content_cache[store_name]
        title_texts = self.title_cache[store_name]
        
        query = self.preprocess_text(query)
        if not query:
            return regular_matches
        
        # Preprocess query for error code specific handling
        query_lower = query.lower()
        query_no_hyphens = query_lower.replace('-', '').replace(' ', '')
        
        # Enhanced matches dictionary - start with regular matches
        enhanced_matches = dict(regular_matches)
        
        # Extract possible error code components
        code_pattern = re.compile(r'([a-z])[- ]?([a-z]+)[- ]?(\d+)', re.IGNORECASE)
        name_pattern = re.compile(r'([a-z]+)error\b', re.IGNORECASE)
        
        code_matches = code_pattern.findall(query_lower)
        name_matches = name_pattern.findall(query_lower)
        
        # Process each title for specialized error code matching
        for i, title in enumerate(title_texts):
            if i >= len(metadata):
                continue
                
            title_lower = title.lower()
            title_no_hyphens = title_lower.replace('-', '').replace(' ', '')
            
            # Skip if already highly scored
            if i in enhanced_matches and enhanced_matches[i] > 0.8:
                continue
                
            score = 0.0
            matched_terms = []
            
            # 1. Check for error code pattern matches
            for prefix, category, number in code_matches:
                # Original format check
                code = f"{prefix}-{category}-{number}"
                if code.lower() in title_lower:
                    score += 0.8
                    matched_terms.append(f"CODE({code})")
                    
                # No hyphens check (normalized)
                elif f"{prefix}{category}{number}".lower() in title_no_hyphens:
                    score += 0.7
                    matched_terms.append(f"CODE_NORM({prefix}{category}{number})")
            
            # 2. Check for error name matches
            for name in name_matches:
                error_name = f"{name}error"
                if error_name in title_lower:
                    score += 0.6
                    matched_terms.append(f"NAME({error_name})")
                
                # Check for name parts (like "workflow" in "WorkflowDiscoveryError")
                if name in title_lower and "error" in title_lower:
                    score += 0.4
                    matched_terms.append(f"NAME_PART({name})")
            
            # 3. Look for specialized patterns that were problematic
            if "fix" in query_lower or "resolv" in query_lower or "solution" in query_lower:
                if i in enhanced_matches:
                    # Boost existing matches for "fix/resolve" queries
                    enhanced_matches[i] *= 1.2
                    matched_terms.append("RESOLUTION_BOOST")
            
            # Only add if score is significant and better than existing
            if score > 0 and (i not in enhanced_matches or score > enhanced_matches[i]):
                enhanced_matches[i] = min(1.0, score)
                if self.debug:
                    logger.debug(f"Enhanced error code match for doc {i}: score={score:.3f}, matches={matched_terms}")
        
        return enhanced_matches

    def combine_scores_weighted(self, doc_ids: Set[int], vector_scores: Dict[int, float],
                                bm25_scores: Dict[int, float], contextual_scores: Dict[int, float],
                                section_scores: Dict[int, float], keyword_scores: Dict[int, float],
                                is_error_code: bool = False) -> Dict[int, float]:
        """
        Combine multiple scoring signals using weighted sum with adaptive weights.
        
        Args:
            doc_ids: Set of document IDs to score
            vector_scores: Vector similarity scores
            bm25_scores: BM25 relevance scores
            contextual_scores: Contextual similarity scores
            section_scores: Document structure relevance scores
            keyword_scores: Exact keyword match scores
            is_error_code: Whether this is an error code search
            
        Returns:
            Dictionary mapping document IDs to combined scores
            
        Example:
            >>> combined = store.combine_scores_weighted(doc_ids, vector_scores, bm25_scores, 
            ...                                         contextual_scores, section_scores, keyword_scores)
            >>> combined
            {5: 0.89, 12: 0.76, 42: 0.54}
        """
        combined_scores = {}

        base_weights = {
            'vector': 0.05,
            'bm25': 0.55,
            'keyword': 0.25,
            'contextual': 0.15,
            'section': 0.05
        }
        
        # Adjust weights for error code searches
        if is_error_code:
            base_weights['keyword'] = 0.4  # Increase keyword match importance
            base_weights['vector'] = 0.1   # Decrease vector importance
            base_weights['contextual'] = 0.1  # Decrease contextual importance

        for idx in doc_ids:
            vs = vector_scores.get(idx, 0.0)
            bs = bm25_scores.get(idx, 0.0)
            cs = contextual_scores.get(idx, 0.0)
            ss = section_scores.get(idx, 0.0)
            ks = keyword_scores.get(idx, 0.0)

            weights = base_weights.copy()

            if ks > 0.7:
                weights['keyword'] *= 1.5
                weights['bm25'] *= 1.1
                weights['vector'] *= 0.5

            if bs > 0.6 and ks < 0.3:
                weights['bm25'] *= 1.2
                weights['contextual'] *= 1.1

            if cs > 0.6:
                weights['contextual'] *= 1.2

            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

            weighted_score = (
                    vs * normalized_weights['vector'] +
                    bs * normalized_weights['bm25'] +
                    ks * normalized_weights['keyword'] +
                    cs * normalized_weights['contextual'] +
                    ss * normalized_weights['section']
            )

            combined_scores[idx] = weighted_score

        return combined_scores

    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize scores to a 0-1 range using square root transformation.
        
        Args:
            scores: Dictionary mapping document IDs to scores
            
        Returns:
            Dictionary with normalized scores
            
        Example:
            >>> normalized = store.normalize_scores({1: 0.2, 2: 0.5, 3: 0.8})
            >>> normalized
            {1: 0.0, 2: 0.5, 3: 1.0}
        """
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score == min_score:
            return {idx: 1.0 for idx in scores}

        normalized = {}
        score_range = max_score - min_score

        for idx, score in scores.items():
            normalized_value = ((score - min_score) / score_range) ** 0.5
            normalized[idx] = normalized_value

        return normalized

    def search(self, name: str, query: str, top_k: int = 3, threshold: float = 0.2,
               use_bm25: bool = True, use_expansion: bool = True, title_only: bool = False,
               clarify_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search a FAISS index with hybrid ranking system.
        
        Args:
            name: Name of the vector store to search
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum score threshold for results
            use_bm25: Whether to use BM25 scoring
            use_expansion: Whether to use query expansion
            title_only: Whether to search only in titles
            clarify_threshold: Score threshold below which multiple results are provided
        
        Returns:
            List of search results with scores and optional clarification flags
            
        Example:
            >>> results = store.search("tech_docs", "kubernetes deployment options", top_k=2)
            >>> results
            [
                {"title": "Kubernetes Deployment Strategies", "content": "...", "score": 0.92},
                {"title": "Advanced Deployment Patterns", "content": "...", "score": 0.76}
            ]
        """
        logger.info(f"Searching store '{name}' for: '{query}', title_only={title_only}")

        # Check if this is an error code search
        is_error_code_search = 'error_code' in name.lower()
        
        # Save original weights if this is an error code search
        original_weights = None
        if is_error_code_search:
            original_weights = {
                'vector': self.vector_weight,
                'bm25': self.bm25_weight,
                'contextual': self.contextual_weight,
                'section': self.section_weight
            }

        try:
            index, metadata = self.load_store(name)
            if index is None or metadata is None:
                logger.warning(f"Could not load index or metadata for {name}")
                return []

            if len(query.strip()) < 3:
                logger.warning(f"Query is too short: '{query}'")
                return []

            # Use enhanced keyword matching for error codes
            if is_error_code_search:
                keyword_scores = self.enhanced_keyword_match(query, name, title_only)
                # Temporarily adjust weights for error code search
                self.vector_weight = 0.1  # Reduce vector weight
                self.bm25_weight = 0.4    # Adjust BM25 weight
                self.contextual_weight = 0.1  # Reduce contextual weight
            else:
                keyword_scores = self.keyword_match(query, name, title_only)
                
            keyword_matches = set(keyword_scores.keys())
            logger.info(f"Found {len(keyword_matches)} keyword matches")

            has_bm25 = name in self.term_freq_cache
            if use_bm25 and not has_bm25:
                logger.warning(f"BM25 data not available for {name}, using fallback method")

            query_vector = self.vectorize_text(query, self.idf_cache.get(name))
            query_vector_np = np.array([query_vector]).astype('float32')

            search_k = min(top_k * 5, index.ntotal)
            D, I = index.search(query_vector_np, search_k)

            cosine_cutoff = 0.3

            vector_scores = {}
            for i, idx in enumerate(I[0]):
                if idx < 0 or idx >= len(metadata):
                    continue

                raw_cosine = float(D[0][i])

                if raw_cosine < cosine_cutoff:
                    logger.debug(f"Filtered out vector with low cosine: {raw_cosine:.3f} < {cosine_cutoff}")
                    continue

                norm_score = (raw_cosine + 1) / 2
                vector_scores[idx] = norm_score

            bm25_scores = {}
            if use_bm25 and has_bm25:
                bm25_scores = self.calculate_bm25_scores(query, name)
                logger.debug(f"BM25 found scores for {len(bm25_scores)} documents")

            candidates = set()
            candidates.update(vector_scores.keys())
            candidates.update(bm25_scores.keys())
            candidates.update(keyword_matches)

            if not candidates:
                logger.info(f"No candidates found for query: '{query}'")
                return []

            content_texts = self.content_cache.get(name, [])
            title_texts = self.title_cache.get(name, [])

            contextual_scores = {}
            section_scores = {}

            for idx in candidates:
                if idx >= len(metadata):
                    continue

                if title_only:
                    if idx < len(title_texts) and title_texts[idx]:
                        contextual_score = self.calculate_contextual_similarity(query, title_texts[idx])
                        contextual_scores[idx] = contextual_score
                else:
                    if idx < len(content_texts) and content_texts[idx]:
                        contextual_score = self.calculate_contextual_similarity(query, content_texts[idx])
                        contextual_scores[idx] = contextual_score

                section_score = self.calculate_section_relevance(query, metadata[idx])
                section_scores[idx] = section_score

            combined_scores = self.combine_scores_weighted(
                candidates, vector_scores, bm25_scores, contextual_scores,
                section_scores, keyword_scores, is_error_code=is_error_code_search
            )

            raw_max_score = max(combined_scores.values()) if combined_scores else 0
            raw_threshold = 0.15

            if raw_max_score < raw_threshold:
                logger.info(
                    f"Best raw score ({raw_max_score:.3f}) below threshold ({raw_threshold:.3f}), returning no results")
                return []

            normalized_scores = self.normalize_scores(combined_scores)

            if not normalized_scores:
                logger.info("No results after score normalization and filtering")
                return []

            filtered_results = []
            for idx, score in normalized_scores.items():
                if score >= threshold and idx < len(metadata):
                    result = metadata[idx].copy()

                    result['score'] = score
                    result['vector_score'] = vector_scores.get(idx, 0)
                    result['bm25_score'] = bm25_scores.get(idx, 0) if has_bm25 else 0
                    result['contextual_score'] = contextual_scores.get(idx, 0)
                    result['section_score'] = section_scores.get(idx, 0)
                    result['keyword_match'] = idx in keyword_matches

                    filtered_results.append(result)

            filtered_results.sort(key=lambda x: x['score'], reverse=True)

            if filtered_results:
                logger.info(
                    f"Top result: '{filtered_results[0].get('title', 'No title')}' with score {filtered_results[0]['score']:.3f}")

                needs_clarification = False
                low_confidence_gap = False

                if filtered_results[0]['score'] < clarify_threshold:
                    needs_clarification = True
                    logger.info(f"Top result score ({filtered_results[0]['score']:.3f}) below clarification threshold ({clarify_threshold:.3f})")

                if len(filtered_results) >= 2:
                    score_gap = filtered_results[0]['score'] - filtered_results[1]['score']
                    if score_gap < 0.15:
                        low_confidence_gap = True
                        logger.info(f"Small score gap between top results: {score_gap:.3f}")

                if needs_clarification or low_confidence_gap:
                    for result in filtered_results[:min(3, len(filtered_results))]:
                        result['needs_clarification'] = True

                for i, result in enumerate(filtered_results[:3]):
                    logger.debug(f"Result {i + 1}: {result.get('title', 'No title')}")
                    logger.debug(f"  Vector: {result['vector_score']:.3f}, BM25: {result['bm25_score']:.3f}, "
                                f"Keyword match: {result['keyword_match']}, "
                                f"Score: {result['score']:.3f}")

            if filtered_results and any(r.get('needs_clarification', False) for r in filtered_results[:1]):
                return_count = min(3, len(filtered_results))
                logger.info(f"Returning {return_count} results for clarification")
                return filtered_results[:return_count]
            else:
                return filtered_results[:top_k]
                
        finally:
            # Restore original weights if they were changed
            if original_weights:
                self.vector_weight = original_weights['vector']
                self.bm25_weight = original_weights['bm25']
                self.contextual_weight = original_weights['contextual']
                self.section_weight = original_weights['section']

    def search_all_stores(self, query: str, top_k: int = 3, threshold: float = 0.2,
                          domain_prefix: Optional[str] = None, use_bm25: bool = True,
                          title_only: bool = False, clarify_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search across all available vector stores with global ranking.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum score threshold for results
            domain_prefix: Optional domain prefix to filter stores
            use_bm25: Whether to use BM25 scoring
            title_only: Whether to search only in titles
            clarify_threshold: Score threshold for clarification
            
        Returns:
            List of search results from all stores with scores
            
        Example:
            >>> results = store.search_all_stores("kubernetes setup guide", domain_prefix="cloud_")
            >>> results
            [
                {"title": "Kubernetes Setup on AWS", "content": "...", "score": 0.95, "store": "cloud_aws"},
                {"title": "GKE Quick Start", "content": "...", "score": 0.87, "store": "cloud_gcp"}
            ]
        """
        all_raw_results = []
        matching_stores = []

        for store_name in self.list_stores():
            if domain_prefix and not store_name.startswith(domain_prefix):
                continue

            matching_stores.append(store_name)

        logger.info(f"Searching across {len(matching_stores)} stores matching domain prefix '{domain_prefix}'")

        for store_name in matching_stores:
            index, metadata = self.load_store(store_name)
            if index is None or metadata is None:
                logger.warning(f"Could not load index or metadata for {store_name}")
                continue

            if len(query.strip()) < 3:
                logger.warning(f"Query is too short: '{query}'")
                continue
            
            # Use enhanced keyword matching for error code stores
            is_error_code_store = 'error_code' in store_name.lower()
            if is_error_code_store:
                keyword_scores = self.enhanced_keyword_match(query, store_name, title_only)
            else:
                keyword_scores = self.keyword_match(query, store_name, title_only)
                
            keyword_matches = set(keyword_scores.keys())

            has_bm25 = store_name in self.term_freq_cache

            query_vector = self.vectorize_text(query, self.idf_cache.get(store_name))
            query_vector_np = np.array([query_vector]).astype('float32')

            search_k = min(top_k * 5, index.ntotal)
            D, I = index.search(query_vector_np, search_k)

            cosine_cutoff = 0.3

            vector_scores = {}
            for i, idx in enumerate(I[0]):
                if idx < 0 or idx >= len(metadata):
                    continue

                raw_cosine = float(D[0][i])

                if raw_cosine < cosine_cutoff:
                    continue

                norm_score = (raw_cosine + 1) / 2
                vector_scores[idx] = norm_score

            bm25_scores = {}
            if use_bm25 and has_bm25:
                bm25_scores = self.calculate_bm25_scores(query, store_name)

            candidates = set()
            candidates.update(vector_scores.keys())
            candidates.update(bm25_scores.keys())
            candidates.update(keyword_matches)

            if not candidates:
                continue

            content_texts = self.content_cache.get(store_name, [])
            title_texts = self.title_cache.get(store_name, [])

            contextual_scores = {}
            section_scores = {}

            for idx in candidates:
                if idx >= len(metadata):
                    continue

                if title_only:
                    if idx < len(title_texts) and title_texts[idx]:
                        contextual_score = self.calculate_contextual_similarity(query, title_texts[idx])
                        contextual_scores[idx] = contextual_score
                else:
                    if idx < len(content_texts) and content_texts[idx]:
                        contextual_score = self.calculate_contextual_similarity(query, content_texts[idx])
                        contextual_scores[idx] = contextual_score

                section_score = self.calculate_section_relevance(query, metadata[idx])
                section_scores[idx] = section_score

            # Adjust weights for error code stores
            if is_error_code_store:
                weights = {
                    'vector': 0.1,
                    'bm25': 0.4,
                    'keyword': 0.4,
                    'contextual': 0.1,
                    'section': 0.05,
                }
            else:
                weights = {
                    'vector': 0.05,
                    'bm25': 0.55,
                    'keyword': 0.25,
                    'contextual': 0.15,
                    'section': 0.05,
                }

            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum for k, v in weights.items()}

            for idx in candidates:
                vs = vector_scores.get(idx, 0.0)
                bs = bm25_scores.get(idx, 0.0)
                ks = keyword_scores.get(idx, 0.0)
                cs = contextual_scores.get(idx, 0.0)
                ss = section_scores.get(idx, 0.0)

                raw_score = (
                        vs * weights['vector'] +
                        bs * weights['bm25'] +
                        ks * weights['keyword'] +
                        cs * weights['contextual'] +
                        ss * weights['section']
                )

                min_score_threshold = 0.1
                if raw_score < min_score_threshold:
                    continue

                result = metadata[idx].copy()
                result['store'] = store_name
                result['score'] = raw_score
                result['vector_score'] = vs
                result['bm25_score'] = bs
                result['keyword_match'] = idx in keyword_matches
                result['contextual_score'] = cs
                result['section_score'] = ss

                all_raw_results.append(result)

        if not all_raw_results:
            logger.info(f"No results found across {len(matching_stores)} stores")
            return []

        min_score = min(result['score'] for result in all_raw_results)
        max_score = max(result['score'] for result in all_raw_results)

        score_range = max_score - min_score
        if score_range > 0:
            for result in all_raw_results:
                norm_score = ((result['score'] - min_score) / score_range) ** (1 / 3)
                result['normalized_score'] = norm_score
        else:
            for result in all_raw_results:
                result['normalized_score'] = 1.0

        all_raw_results.sort(key=lambda x: x['normalized_score'], reverse=True)

        filtered_results = [r for r in all_raw_results if r['normalized_score'] >= threshold]

        for result in filtered_results:
            result['score'] = result['normalized_score']
            if 'normalized_score' in result:
                del result['normalized_score']

        if filtered_results:
            logger.info(f"Top result across all stores: '{filtered_results[0].get('title', 'No title')}' "
                        f"from store '{filtered_results[0].get('store', 'unknown')}' with score {filtered_results[0]['score']:.3f}")

            needs_clarification = False
            low_confidence_gap = False

            if filtered_results[0]['score'] < clarify_threshold:
                needs_clarification = True
                logger.info(f"Top result score ({filtered_results[0]['score']:.3f}) below clarification threshold ({clarify_threshold:.3f})")

            if len(filtered_results) >= 2:
                score_gap = filtered_results[0]['score'] - filtered_results[1]['score']
                if score_gap < 0.15:
                    low_confidence_gap = True
                    logger.info(f"Small score gap between top results: {score_gap:.3f}")

            if needs_clarification or low_confidence_gap:
                for result in filtered_results[:min(3, len(filtered_results))]:
                    result['needs_clarification'] = True

            for i, result in enumerate(filtered_results[:3]):
                logger.debug(
                    f"Result {i + 1}: {result.get('title', 'No title')} (store: {result.get('store', 'unknown')})")
                logger.debug(f"  Vector: {result['vector_score']:.3f}, BM25: {result.get('bm25_score', 0):.3f}, "
                             f"Contextual: {result['contextual_score']:.3f}, "
                             f"Keyword match: {result['keyword_match']}, "
                             f"Score: {result['score']:.3f}")

        if filtered_results and any(r.get('needs_clarification', False) for r in filtered_results[:1]):
            return_count = min(3, len(filtered_results))
            logger.info(f"Returning {return_count} results for clarification")
            return filtered_results[:return_count]
        else:
            return filtered_results[:top_k]

    def list_stores(self) -> List[str]:
        """
        List all available vector stores in the configured directory.
        
        Returns:
            List of store names
            
        Example:
            >>> store.list_stores()
            ['tech_docs', 'cloud_aws', 'cloud_gcp', 'deployment_guide']
        """
        stores = []
        for file in os.listdir(self.store_dir):
            if file.endswith('.index'):
                stores.append(file[:-6])
        return stores

    def create_store_from_confluence(self, name: str, connector, url: str) -> None:
        """
        Create a vector store from a single Confluence page.
        
        Args:
            name: Name of the vector store to create
            connector: Confluence connector instance
            url: URL of the Confluence page
            
        Example:
            >>> store.create_store_from_confluence(
            ...     "deployment_docs", 
            ...     confluence_connector,
            ...     "https://confluence.example.com/display/TECH/Deployment+Guide"
            ... )
        """
        page_content = connector.get_content_by_url(url)
        if 'error' in page_content:
            logger.error(f"Error fetching content from {url}: {page_content['error']}")
            return

        content_chunks = page_content.get('content', [])
        self.create_store(name, content_chunks)

    def create_multi_store_from_confluence(self, stores_dict: Dict[str, List[str]], connector) -> None:
        """
        Create multiple vector stores from Confluence pages.
        
        Args:
            stores_dict: Dictionary mapping store names to lists of page URLs
            connector: Confluence connector instance
            
        Example:
            >>> pages = {
            ...     "aws_docs": [
            ...         "https://confluence.example.com/display/CLOUD/AWS+Overview",
            ...         "https://confluence.example.com/display/CLOUD/AWS+Services"
            ...     ],
            ...     "gcp_docs": [
            ...         "https://confluence.example.com/display/CLOUD/GCP+Overview"
            ...     ]
            ... }
            >>> store.create_multi_store_from_confluence(pages, confluence_connector)
        """
        for name, urls in stores_dict.items():
            logger.info(f"Creating multi-store {name} from {len(urls)} Confluence pages")
            all_chunks = []

            for url in urls:
                page_content = connector.get_content_by_url(url)
                if 'error' in page_content:
                    logger.error(f"Error fetching content from {url}: {page_content['error']}")
                    continue

                content_chunks = page_content.get('content', [])
                logger.info(f"Fetched {len(content_chunks)} chunks from {url}")
                all_chunks.extend(content_chunks)

            if not all_chunks:
                logger.warning(f"No content chunks found for store {name}")
                return

            self.create_store(name, all_chunks)
