import os
import logging
import json
import numpy as np
import faiss
import spacy
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    FAISS vector store optimized for FAQ and error code content (tabular Q&A data).
    """

    def __init__(self, store_dir: str = "../.vector_stores", dim: int = 300,
                 model_name: str = "en_core_web_md", k1: float = 1.5, b: float = 0.75):
        """Initialize the FAISS vector store."""
        self.store_dir = store_dir
        self.dimension = dim
        self.nlp = spacy.load(model_name)
        # Caches for loaded data
        self.index_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.content_cache: Dict[str, List[str]] = {}
        self.title_cache: Dict[str, List[str]] = {}
        self.term_freq_cache: Dict[str, List[Counter]] = {}
        self.doc_len_cache: Dict[str, List[int]] = {}
        self.avg_doc_len_cache: Dict[str, float] = {}
        self.idf_cache: Dict[str, Dict[str, float]] = {}
        # BM25 parameters
        self.k1 = k1
        self.b = b
        # Scoring weight defaults (BM25: 0.5, vector: 0.3, contextual: 0.2)
        self.vector_weight = 0.3
        self.bm25_weight = 0.5
        self.contextual_weight = 0.2
        # Ensure store directory exists
        os.makedirs(store_dir, exist_ok=True)
        # Debug flag
        self.debug = True

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by lowercasing and collapsing whitespace."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy, removing stopwords and punctuation."""
        text = self.preprocess_text(text)
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def vectorize_text(self, text: str, idf_dict: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Convert text to vector using spaCy token vectors weighted by TF-IDF."""
        text = self.preprocess_text(text)
        if not text or len(text) < 3:
            # Return a deterministic random unit vector for empty or very short text
            np.random.seed(0)
            vec = np.random.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / max(norm, 1e-10)
        # Analyze text with spaCy (limit length for performance)
        doc = self.nlp(text[:5000])
        # Compute term frequencies in this document
        term_freq = Counter()
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                lemma = token.lemma_.lower()
                term_freq[lemma] += 1
        max_freq = max(term_freq.values()) if term_freq else 1
        token_vectors = []
        token_weights = []
        processed_lemmas: set = set()
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) <= 1:
                continue
            if not token.has_vector:
                continue
            lemma = token.lemma_.lower()
            if lemma in processed_lemmas:
                continue
            processed_lemmas.add(lemma)
            # Optionally skip common boilerplate terms if low IDF
            if lemma in ['click', 'link', 'page', 'home', 'contact', 'copyright'] and idf_dict:
                if lemma in idf_dict and idf_dict[lemma] < 2.0:
                    continue
            weight = 1.0
            # Boost certain POS tags
            if token.pos_ in ['NOUN', 'PROPN']:
                weight *= 2.5
            elif token.pos_ == 'VERB':
                weight *= 1.5
            elif token.pos_ in ['ADJ', 'ADV']:
                weight *= 1.2
            # Boost named entities
            if token.ent_type_:
                if token.ent_type_ in ['ORG', 'PRODUCT', 'GPE', 'LOC', 'PERSON']:
                    weight *= 2.0
                else:
                    weight *= 1.5
            # Normalize term frequency (log scaling)
            tf = term_freq[lemma]
            norm_tf = 0.5 + 0.5 * (math.log(1 + tf) / math.log(1 + max_freq))
            weight *= norm_tf
            # Apply IDF weight if provided (capped to avoid over-weighting extremely rare terms)
            if idf_dict and lemma in idf_dict:
                weight *= min(idf_dict[lemma], 10.0)
            # Boost early-position terms (first 20% of document)
            if token.i / len(doc) < 0.2:
                weight *= 1.3
            token_vectors.append(token.vector)
            token_weights.append(weight)
        if token_vectors:
            token_vectors = np.array(token_vectors, dtype=np.float32)
            token_weights = np.array(token_weights, dtype=np.float32)
            if token_weights.sum() > 0:
                token_weights /= token_weights.sum()
            else:
                token_weights = np.ones_like(token_weights) / len(token_weights)
            vec = np.average(token_vectors, axis=0, weights=token_weights).astype(np.float32)
        else:
            vec = doc.vector.astype(np.float32) if doc.has_vector else np.random.randn(self.dimension).astype(np.float32)
        # Add slight noise for uniqueness and normalize
        text_hash = hash(text) & 0xffffffff
        np.random.seed(text_hash % 10000)
        noise = np.random.randn(self.dimension).astype(np.float32) * 0.001
        vec = vec + noise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)
        return vec

    def create_store(self, name: str, texts: List[Dict[str, Any]]) -> None:
        """Create a FAISS index from a list of documents (with 'content' and optional 'title')."""
        if not texts:
            logger.warning(f"No texts provided for store {name}")
            return
        vectors = []
        metadata = []
        content_texts = []
        title_texts = []
        term_frequencies = []
        doc_lengths = []
        # Compute collection-wide IDF
        all_terms = set()
        term_doc_frequencies = Counter()
        for item in texts:
            if not item.get('content'):
                continue
            tokens = self.tokenize(item['content'])
            all_terms.update(tokens)
            for term in set(tokens):
                term_doc_frequencies[term] += 1
        num_docs = len(texts)
        idf_dict = {term: math.log((num_docs + 1) / (df + 1)) + 1 for term, df in term_doc_frequencies.items()}
        # Build index entries and vectors
        content_hashes = {}
        for i, item in enumerate(texts):
            if not item.get('content'):
                continue
            text = item['content']
            title = item.get('title', '')
            # Skip duplicate content
            content_hash = hash(text.lower())
            if content_hash in content_hashes:
                dup_idx = content_hashes[content_hash]
                logger.warning(f"Duplicate content: item {i} ('{title}') same as item {dup_idx} ('{metadata[dup_idx].get('title', '')}') - skipping.")
                continue
            content_hashes[content_hash] = i
            processed_text = self.preprocess_text(text)
            content_texts.append(processed_text)
            processed_title = self.preprocess_text(title)
            title_texts.append(processed_title)
            # Prepare BM25 data
            tokens = self.tokenize(text)
            term_frequencies.append(Counter(tokens))
            doc_lengths.append(len(tokens))
            # Vectorize content
            vec = self.vectorize_text(text, idf_dict)
            if np.all(np.abs(vec) < 1e-8):
                logger.warning(f"Skipping text with near-zero vector: {text[:50]}")
                continue
            vectors.append(vec)
            # Store metadata (copy original item and assign new id index)
            meta_item = item.copy()
            meta_item['id'] = len(vectors) - 1
            metadata.append(meta_item)
        if not vectors:
            logger.warning(f"No vectors created for store {name}")
            return
        # Create FAISS index (inner product for cosine similarity)
        vectors_np = np.array(vectors, dtype=np.float32)
        # Normalize all vectors
        norms = np.linalg.norm(vectors_np, axis=1)
        vectors_np = vectors_np / np.maximum(norms[:, np.newaxis], 1e-10)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors_np)
        # Save index and metadata
        os.makedirs(self.store_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(self.store_dir, f"{name}.index"))
        with open(os.path.join(self.store_dir, f"{name}.json"), 'w') as f:
            json.dump(metadata, f)
        # Save BM25 data
        bm25_data = {
            "term_frequencies": [dict(tf) for tf in term_frequencies],
            "doc_lengths": doc_lengths,
            "avg_doc_len": sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            "idf": idf_dict
        }
        with open(os.path.join(self.store_dir, f"{name}.bm25.json"), 'w') as f:
            json.dump(bm25_data, f)
        # Update caches
        self.index_cache[name] = index
        self.metadata_cache[name] = metadata
        self.content_cache[name] = content_texts
        self.title_cache[name] = title_texts
        self.term_freq_cache[name] = term_frequencies
        self.doc_len_cache[name] = doc_lengths
        self.avg_doc_len_cache[name] = bm25_data["avg_doc_len"]
        self.idf_cache[name] = idf_dict
        logger.info(f"Created vector store '{name}' with {len(vectors)} entries")

    def load_store(self, name: str) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
        """Load a FAISS index and metadata from disk (if not already loaded)."""
        if name in self.index_cache and name in self.metadata_cache:
            return self.index_cache[name], self.metadata_cache[name]
        index_path = os.path.join(self.store_dir, f"{name}.index")
        metadata_path = os.path.join(self.store_dir, f"{name}.json")
        if not os.path.exists(index_path):
            logger.warning(f"Index file {index_path} not found")
            return None, None
        index = faiss.read_index(index_path)
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file {metadata_path} not found")
            self.index_cache[name] = index
            return index, None
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Cache loaded index and metadata
        self.index_cache[name] = index
        self.metadata_cache[name] = metadata
        # Load or generate BM25 data
        self._load_bm25_data(name, metadata)
        return index, metadata

    def _load_bm25_data(self, name: str, metadata: List[Dict[str, Any]]) -> bool:
        """Ensure BM25 data is loaded in caches (load from file or compute)."""
        if name in self.term_freq_cache and name in self.idf_cache:
            return True
        bm25_path = os.path.join(self.store_dir, f"{name}.bm25.json")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'r') as f:
                    bm25_data = json.load(f)
                self.term_freq_cache[name] = [Counter(tf) for tf in bm25_data.get("term_frequencies", [])]
                self.doc_len_cache[name] = bm25_data.get("doc_lengths", [])
                self.avg_doc_len_cache[name] = bm25_data.get("avg_doc_len", 0)
                self.idf_cache[name] = bm25_data.get("idf", {})
                # Ensure content/title caches are filled
                if name not in self.content_cache or name not in self.title_cache:
                    content_texts = []
                    title_texts = []
                    for item in metadata:
                        content_texts.append(self.preprocess_text(item.get('content', '')))
                        title_texts.append(self.preprocess_text(item.get('title', '')))
                    self.content_cache[name] = content_texts
                    self.title_cache[name] = title_texts
                logger.info(f"Loaded BM25 data for store {name}")
                return True
            except Exception as e:
                logger.warning(f"Error loading BM25 data for {name}: {e}")
        # Compute BM25 data if not available
        logger.info(f"Generating BM25 data for store {name}")
        try:
            content_texts = []
            title_texts = []
            term_frequencies = []
            doc_lengths = []
            all_terms = set()
            term_doc_frequencies = Counter()
            for item in metadata:
                text = item.get('content', '')
                content_texts.append(self.preprocess_text(text))
                title_texts.append(self.preprocess_text(item.get('title', '')))
                tokens = self.tokenize(text)
                token_counter = Counter(tokens)
                term_frequencies.append(token_counter)
                doc_lengths.append(len(tokens))
                all_terms.update(tokens)
                for term in set(tokens):
                    term_doc_frequencies[term] += 1
            self.content_cache[name] = content_texts
            self.title_cache[name] = title_texts
            avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
            idf_dict = {term: math.log((len(doc_lengths) + 1) / (df + 1)) + 1 for term, df in term_doc_frequencies.items()}
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
            logger.info(f"Generated BM25 data for store {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate BM25 data for {name}: {e}")
            return False

    def keyword_match(self, query: str, store_name: str) -> Dict[int, float]:
        """Lexical matching with boosts for exact query and Issue field matches."""
        # Ensure content and title caches are available
        if store_name not in self.content_cache or store_name not in self.title_cache:
            _, metadata = self.load_store(store_name)
            if not metadata:
                return {}
        content_texts = self.content_cache.get(store_name, [])
        title_texts = self.title_cache.get(store_name, [])
        query = self.preprocess_text(query)
        if not query:
            return {}
        matches: Dict[int, float] = {}
        # Use spaCy to extract query terms (lemmas and noun chunks)
        query_terms = []
        raw_query_lower = query.lower()
        try:
            query_doc = self.nlp(query)
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
        except Exception as e:
            logger.warning(f"Error processing query with spaCy: {e}")
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not query_terms:
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]
        # Check each document content and title
        for i in range(len(content_texts)):
            if i >= len(content_texts) or not content_texts[i]:
                continue
            text = content_texts[i].lower()
            title_text = title_texts[i].lower() if i < len(title_texts) and title_texts[i] else ""
            score = 0.0
            matched_terms = []
            content_matched = False
            issue_matched = False
            # Full query exact match in content
            if raw_query_lower and raw_query_lower in text:
                score += 0.8
                content_matched = True
                matched_terms.append("FULL_QUERY")
            # Individual term matches in content
            for term in query_terms:
                if term in text:
                    term_score = min(0.2, 0.05 * text.count(term))
                    if term_score > 0:
                        score += term_score
                        content_matched = True
                        matched_terms.append(term)
            # If title is available, check matches in title (for error codes or headings)
            if title_text:
                title_matches = sum(1 for term in query_terms if term in title_text)
                if title_matches > 0:
                    title_score = min(0.5, 0.15 * title_matches)
                    score += title_score
                    matched_terms.append(f"TITLE({title_matches})")
                if raw_query_lower and raw_query_lower in title_text:
                    score += 0.5
                    matched_terms.append("FULL_QUERY_IN_TITLE")
            # Boost matches in Issue field if present in metadata raw_data
            if store_name in self.metadata_cache and i < len(self.metadata_cache[store_name]):
                meta = self.metadata_cache[store_name][i]
                if 'raw_data' in meta and 'Issue' in meta['raw_data']:
                    issue_text = meta['raw_data']['Issue'].lower()
                    if issue_text:
                        issue_term_matches = sum(1 for term in query_terms if term in issue_text)
                        if issue_term_matches > 0:
                            issue_score = min(0.4, 0.12 * issue_term_matches)
                            score += issue_score
                            issue_matched = True
                            matched_terms.append(f"ISSUE({issue_term_matches})")
                        if raw_query_lower and raw_query_lower in issue_text:
                            score += 0.6
                            issue_matched = True
                            matched_terms.append("FULL_QUERY_IN_ISSUE")
            # Extra boost if query terms appear in both Issue and Details
            if content_matched and issue_matched:
                score += 0.2
                matched_terms.append("BOTH_FIELDS")
            if score > 0:
                matches[i] = min(1.0, score)
                if self.debug:
                    logger.debug(f"Keyword match doc {i}: score={score:.3f}, terms={matched_terms}")
        return matches

    def calculate_bm25_scores(self, query: str, store_name: str) -> Dict[int, float]:
        """Calculate BM25 scores for the query against the stored corpus."""
        if store_name not in self.term_freq_cache:
            _, metadata = self.load_store(store_name)
            if metadata and store_name not in self.term_freq_cache:
                success = self._load_bm25_data(store_name, metadata)
                if not success:
                    logger.warning(f"BM25 data not available for {store_name}")
                    return {}
        term_frequencies = self.term_freq_cache.get(store_name, [])
        doc_lengths = self.doc_len_cache.get(store_name, [])
        avg_doc_len = self.avg_doc_len_cache.get(store_name, 0)
        idf = self.idf_cache.get(store_name, {})
        query_terms = []
        term_importance: Dict[str, float] = {}
        try:
            query_doc = self.nlp(query)
            for chunk in query_doc.noun_chunks:
                qt = chunk.text.lower()
                query_terms.append(qt)
                term_importance[qt] = 2.0
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
                elif token.pos_ == 'VERB':
                    term_importance[lemma] = 1.5
                elif token.pos_ in ['ADJ', 'ADV']:
                    term_importance[lemma] = 1.3
                else:
                    term_importance[lemma] = 1.0
            for i, token in enumerate(query_doc):
                lem = token.lemma_.lower()
                if i < 3 and lem in term_importance:
                    term_importance[lem] *= 1.2
        except Exception as e:
            logger.warning(f"Error processing query for BM25: {e}")
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]
            for term in query_terms:
                term_importance[term] = 1.0
        scores: Dict[int, float] = {}
        for i, tf_counter in enumerate(term_frequencies):
            if i >= len(doc_lengths) or doc_lengths[i] == 0:
                continue
            doc_len = doc_lengths[i]
            score = 0.0
            for term in query_terms:
                if ' ' in term:
                    # Phrase: try phrase as whole, then individual words
                    parts = term.split()
                    # Whole phrase
                    if term in idf:
                        tf = tf_counter.get(term, 0)
                        imp = term_importance.get(term, 1.0)
                        score += idf[term] * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / (avg_doc_len or 1)))) * imp
                    # Individual words of phrase
                    for part in parts:
                        if part in idf:
                            tf = tf_counter.get(part, 0)
                            imp = term_importance.get(term, 1.0) * 0.7
                            score += idf[part] * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / (avg_doc_len or 1)))) * imp / len(parts)
                else:
                    if term not in idf:
                        continue
                    tf = tf_counter.get(term, 0)
                    imp = term_importance.get(term, 1.0)
                    score += idf[term] * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / (avg_doc_len or 1)))) * imp
            # Bonus for covering most query terms
            if query_terms:
                matched_terms = [t for t in query_terms if (' ' in t and any(part in tf_counter for part in t.split())) or (t in tf_counter)]
                if matched_terms:
                    coverage = len(set(matched_terms)) / len(query_terms)
                    if coverage > 0.7:
                        score += coverage * 0.2
            if score > 0:
                scores[i] = min(1.0, score / 20.0)
                if self.debug and scores[i] > 0.5:
                    logger.debug(f"BM25 score doc {i}: {scores[i]:.3f}")
        return scores

    def calculate_contextual_similarity(self, query: str, text: str) -> float:
        """Compute a semantic similarity score between query and text (0 to 1)."""
        query = self.preprocess_text(query)
        text = self.preprocess_text(text)
        if not query or not text:
            return 0.0
        try:
            query_doc = self.nlp(query)
            text_doc = self.nlp(text)
            if not query_doc.has_vector or not text_doc.has_vector:
                return 0.0
            doc_sim = query_doc.similarity(text_doc)
            query_terms = {token.lemma_.lower() for token in query_doc if not token.is_stop and not token.is_punct and len(token.text) > 2}
            text_terms = {token.lemma_.lower() for token in text_doc if not token.is_stop and not token.is_punct and len(token.text) > 2}
            term_overlap = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
            query_entities = {ent.text.lower() for ent in query_doc.ents}
            text_entities = {ent.text.lower() for ent in text_doc.ents}
            entity_overlap = len(query_entities & text_entities) / len(query_entities) if query_entities else 0.0
            if len(query.split()) > 5:
                score = 0.4 * doc_sim + 0.4 * term_overlap + 0.2 * entity_overlap
            else:
                score = 0.3 * doc_sim + 0.5 * term_overlap + 0.2 * entity_overlap
            if len(query) > 10 and query in text:
                score = max(score, 0.9)
            if ('how' in query or 'what' in query) and 'how' in text:
                score += 0.1
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Contextual similarity error: {e}")
            return 0.0

    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize a dictionary of scores to 0-1 range (using square root scaling)."""
        if not scores:
            return {}
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score == min_score:
            return {idx: 1.0 for idx in scores}
        normalized_scores: Dict[int, float] = {}
        range_score = max_score - min_score
        for idx, sc in scores.items():
            normalized_scores[idx] = ((sc - min_score) / range_score) ** 0.5
        return normalized_scores

    def search(self, name: str, query: str, intent: Optional[str] = None,
               top_k: int = 3, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Search the specified vector store for the query and return top results."""
        logger.info(f"Searching store '{name}' for query: '{query}' (intent={intent})")
        # Load index and metadata
        index, metadata = self.load_store(name)
        if index is None or metadata is None:
            logger.warning(f"Store '{name}' not found or not loaded")
            return []
        if len(query.strip()) < 3:
            logger.warning(f"Query too short: '{query}'")
            return []
        # Determine weight distribution based on intent
        vw, bw, cw = self.vector_weight, self.bm25_weight, self.contextual_weight
        if intent:
            intent_low = intent.lower()
            if intent_low == 'error_code':
                # Emphasize contextual (exact code matches) slightly more, vector slightly less
                vw, bw, cw = 0.2, 0.5, 0.3
            elif intent_low == 'faq':
                vw, bw, cw = 0.3, 0.5, 0.2
        # Keyword-based matching (including Issue field boost)
        keyword_scores = self.keyword_match(query, name)
        keyword_matches = set(keyword_scores.keys())
        logger.info(f"Keyword matches found in {len(keyword_matches)} documents")
        # Vector similarity search
        query_vec = self.vectorize_text(query, self.idf_cache.get(name))
        D, I = index.search(np.array([query_vec], dtype='float32'), min(top_k * 5, index.ntotal))
        cosine_cutoff = 0.3
        vector_scores: Dict[int, float] = {}
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            cos_sim = float(score)
            if cos_sim < cosine_cutoff:
                continue
            # Convert inner product to [0,1] similarity (vectors are normalized, inner product = cosine)
            norm_score = (cos_sim + 1.0) / 2.0
            vector_scores[idx] = norm_score
        # BM25 scores for query
        bm25_scores = self.calculate_bm25_scores(query, name)
        # Merge candidate indices from all sources
        candidates = set(vector_scores.keys()) | set(bm25_scores.keys()) | keyword_matches
        if not candidates:
            logger.info(f"No candidates found for query '{query}'")
            return []
        # Calculate contextual similarity for each candidate (semantic relevance of content)
        contextual_scores: Dict[int, float] = {}
        content_list = self.content_cache.get(name, [])
        for idx in candidates:
            if idx < len(content_list) and content_list[idx]:
                contextual_scores[idx] = self.calculate_contextual_similarity(query, content_list[idx])
        # Combine scores with weights
        combined_scores: Dict[int, float] = {}
        for idx in candidates:
            vs = vector_scores.get(idx, 0.0)
            bs = bm25_scores.get(idx, 0.0)
            ks = keyword_scores.get(idx, 0.0)
            cs = contextual_scores.get(idx, 0.0)
            # Combine keyword and contextual similarity into one contextual signal
            combined_contextual = ks + cs
            if combined_contextual > 1.0:
                combined_contextual = 1.0
            score = vw * vs + bw * bs + cw * combined_contextual
            combined_scores[idx] = score
        # Apply raw score threshold for relevance
        if combined_scores:
            max_raw = max(combined_scores.values())
            if max_raw < 0.15:
                logger.info(f"Max score {max_raw:.3f} below relevance threshold, no results returned")
                return []
        # Normalize scores for output
        normalized_scores = self.normalize_scores(combined_scores)
        if not normalized_scores:
            return []
        # Prepare final results above threshold
        results = []
        for idx, norm_score in normalized_scores.items():
            if norm_score < threshold or idx >= len(metadata):
                continue
            result = metadata[idx].copy()
            result['score'] = norm_score
            result['vector_score'] = vector_scores.get(idx, 0.0)
            result['bm25_score'] = bm25_scores.get(idx, 0.0)
            result['contextual_score'] = contextual_scores.get(idx, 0.0)
            result['keyword_match'] = (idx in keyword_matches)
            results.append(result)
        results.sort(key=lambda x: x['score'], reverse=True)
        if results:
            top = results[0]
            logger.info(f"Top result: '{top.get('title', 'No title')}' (score={top['score']:.3f})")
        return results[:top_k]

def create_store_from_confluence(self, name: str, connector, url: str) -> None:
    """
    Create a vector store from a single Confluence page.

    Args:
        name: Name of the vector store
        connector: Confluence connector instance
        url: URL of the Confluence page
    """
    page_content = connector.get_content_by_url(url)
    if 'error' in page_content:
        logger.error(f"Error fetching content from {url}: {page_content['error']}")
        return

    content_chunks = page_content.get('content', [])
    self.create_store(name, content_chunks)

def create_multi_store_from_confluence(self, stores_dict: Dict[str, List[str]], connector) -> None:
    """
    Create vector stores from multiple Confluence pages.

    Args:
        stores_dict: Dictionary mapping store names to lists of URLs
        connector: Confluence connector instance
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
