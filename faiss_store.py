import os
import logging
import json
import numpy as np
import faiss
import spacy
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set, Union

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    A focused implementation of FAISS vector store optimized for FAQ and error code retrieval.
    """

    def __init__(self, store_dir: str = "../.vector_stores", dim: int = 300,
                 model_name: str = "en_core_web_md"):
        """Initialize the FAISS vector store."""
        self.store_dir = store_dir
        self.dimension = dim
        self.nlp = spacy.load(model_name)

        # Cache to improve performance
        self.index_cache = {}
        self.metadata_cache = {}
        self.text_cache = {}  # Cache for processed text

        # BM25 data
        self.term_freq_cache = {}
        self.doc_len_cache = {}
        self.avg_doc_len_cache = {}
        self.idf_cache = {}

        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation
        self.b = 0.75  # Document length normalization

        # Ensure store directory exists
        os.makedirs(store_dir, exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into meaningful terms using spaCy."""
        text = self.preprocess_text(text)
        doc = self.nlp(text)

        # Filter out stopwords and punctuation
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct
                  and len(token.text) > 2]

        return tokens

    def vectorize_text(self, text: str, doc_type: str = "general",
                       idf_dict: Dict[str, float] = None) -> np.ndarray:
        """
        Convert text to a vector optimized for FAQ or error code matching.

        Args:
            text: The text to vectorize
            doc_type: Type of document ('faq_question', 'faq_answer', 'error_code', 'general')
            idf_dict: Optional IDF dictionary for term weighting
        """
        text = self.preprocess_text(text)

        if not text or len(text) < 3:
            # Return zero vector for empty text
            return np.zeros(self.dimension, dtype=np.float32)

        # Process with spaCy
        doc = self.nlp(text[:5000])  # Limit to 5000 chars for performance

        # Get tokens with vectors
        tokens_with_vectors = [token for token in doc
                               if not token.is_stop and not token.is_punct
                               and len(token.text) > 2 and token.has_vector]

        if not tokens_with_vectors:
            return np.zeros(self.dimension, dtype=np.float32)

        # Calculate token weights based on document type
        token_vectors = []
        token_weights = []

        for token in tokens_with_vectors:
            # Base weight
            weight = 1.0

            # POS-based weighting
            if token.pos_ in ['NOUN', 'PROPN']:
                weight *= 2.0  # Nouns most important
            elif token.pos_ in ['VERB']:
                weight *= 1.5  # Verbs important
            elif token.pos_ in ['ADJ', 'ADV']:
                weight *= 1.2  # Adjectives and adverbs

            # Entity type boosting
            if token.ent_type_:
                weight *= 1.5

            # Document type specific weighting
            if doc_type == 'faq_question':
                # For FAQ questions, emphasize question words and key terms
                if token.text.lower() in ['how', 'what', 'why', 'when', 'where', 'which', 'who']:
                    weight *= 1.5
                # Emphasize technical terms in questions
                if token.pos_ == 'NOUN' and len(token.text) > 3:
                    weight *= 1.3

            elif doc_type == 'error_code':
                # For error codes, emphasize error-related terms
                if token.text.lower() in ['error', 'exception', 'fail', 'issue', 'problem', 'code']:
                    weight *= 2.0
                # Emphasize alphanumeric codes
                if re.match(r'.*[A-Z]-[A-Z]+-\d+.*', token.text, re.IGNORECASE):
                    weight *= 3.0

            # Apply IDF weighting if available
            if idf_dict and token.lemma_.lower() in idf_dict:
                weight *= idf_dict[token.lemma_.lower()]

            token_vectors.append(token.vector)
            token_weights.append(weight)

        # Compute weighted average
        token_vectors = np.array(token_vectors)
        token_weights = np.array(token_weights)

        # Normalize weights
        token_weights = token_weights / token_weights.sum()

        # Weighted average
        vector = np.average(token_vectors, axis=0, weights=token_weights).astype(np.float32)

        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def create_store(self, name: str, items: List[Dict[str, Any]], store_type: str = "general") -> None:
        """
        Create a FAISS index for a set of items.

        Args:
            name: Name of the store
            items: List of items to index
            store_type: Type of store ('faq' or 'error_code' or 'general')
        """
        if not items:
            logger.warning(f"No items provided for store {name}")
            return

        vectors = []
        metadata = []
        processed_texts = []
        term_frequencies = []
        doc_lengths = []

        # Collection-wide term frequency for IDF
        doc_term_frequencies = Counter()

        # First pass - collect term frequencies for IDF
        for item in items:
            # Extract the text based on store type
            if 'content' in item:
                text = item['content']
            elif 'title' in item:
                text = item['title']
            else:
                continue

            if not text:
                continue

            # Tokenize for term frequencies
            tokens = self.tokenize(text)

            # Update document term frequencies
            for term in set(tokens):  # Count each term once per document
                doc_term_frequencies[term] += 1

        # Calculate IDF
        num_docs = len(items)
        idf_dict = {}
        for term, doc_freq in doc_term_frequencies.items():
            # Smooth IDF: log((N+1)/(df+1)) + 1
            idf = math.log((num_docs + 1) / (doc_freq + 1)) + 1
            idf_dict[term] = idf

        # Second pass - vectorize and store
        for i, item in enumerate(items):
            # Extract text
            if 'content' in item:
                text = item['content']
            elif 'title' in item:
                text = item['title']
            else:
                continue

            if not text:
                continue

            # Determine vector type based on store type
            vec_type = store_type

            # Process text
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)

            # Tokenize for BM25
            tokens = self.tokenize(text)
            term_freq = Counter(tokens)
            term_frequencies.append(term_freq)
            doc_lengths.append(len(tokens))

            # Vectorize with appropriate weights
            vector = self.vectorize_text(text, vec_type, idf_dict)

            # Skip if we get a zero vector
            if np.all(np.abs(vector) < 1e-10):
                continue

            vectors.append(vector)

            # Store metadata
            meta_item = item.copy()
            meta_item['id'] = len(vectors) - 1
            meta_item['processed_text'] = processed_text
            metadata.append(meta_item)

        if not vectors:
            logger.warning(f"No valid vectors created for store {name}")
            return

        # Convert to numpy array
        vectors_np = np.array(vectors).astype('float32')

        # Normalize vectors
        norms = np.linalg.norm(vectors_np, axis=1)
        vectors_np = vectors_np / np.maximum(norms[:, np.newaxis], 1e-10)

        # Create FAISS index (flat index with inner product for cosine similarity)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors_np)

        # Save the index
        index_path = os.path.join(self.store_dir, f"{name}.index")
        faiss.write_index(index, index_path)

        # Save the metadata
        metadata_path = os.path.join(self.store_dir, f"{name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Calculate average document length for BM25
        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        # Save BM25 data
        bm25_path = os.path.join(self.store_dir, f"{name}.bm25.json")
        bm25_data = {
            "term_frequencies": [dict(tf) for tf in term_frequencies],
            "doc_lengths": doc_lengths,
            "avg_doc_len": avg_doc_len,
            "idf": idf_dict
        }
        with open(bm25_path, 'w') as f:
            json.dump(bm25_data, f)

        # Update caches
        self.index_cache[name] = index
        self.metadata_cache[name] = metadata
        self.text_cache[name] = processed_texts
        self.term_freq_cache[name] = term_frequencies
        self.doc_len_cache[name] = doc_lengths
        self.avg_doc_len_cache[name] = avg_doc_len
        self.idf_cache[name] = idf_dict

        logger.info(f"Created vector store {name} with {len(vectors)} vectors")

    def load_store(self, name: str) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]]]:
        """Load a FAISS index and its metadata."""
        # Return from cache if available
        if name in self.index_cache and name in self.metadata_cache:
            return self.index_cache[name], self.metadata_cache[name]

        # Load index
        index_path = os.path.join(self.store_dir, f"{name}.index")
        if not os.path.exists(index_path):
            logger.warning(f"Index file {index_path} not found")
            return None, None

        index = faiss.read_index(index_path)

        # Load metadata
        metadata_path = os.path.join(self.store_dir, f"{name}.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file {metadata_path} not found")
            return index, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Update cache
        self.index_cache[name] = index
        self.metadata_cache[name] = metadata

        # Extract processed text for keyword matching
        processed_texts = [item.get('processed_text', '') for item in metadata]
        self.text_cache[name] = processed_texts

        # Load BM25 data
        self._load_bm25_data(name)

        return index, metadata

    def _load_bm25_data(self, name: str) -> bool:
        """Load BM25 data for a store."""
        # Skip if already loaded
        if (name in self.term_freq_cache and name in self.doc_len_cache and
                name in self.avg_doc_len_cache and name in self.idf_cache):
            return True

        # Load from file
        bm25_path = os.path.join(self.store_dir, f"{name}.bm25.json")
        if not os.path.exists(bm25_path):
            logger.warning(f"BM25 data file {bm25_path} not found")
            return False

        try:
            with open(bm25_path, 'r') as f:
                bm25_data = json.load(f)

            # Convert term frequency dictionaries to Counter objects
            term_frequencies = [Counter(tf) for tf in bm25_data["term_frequencies"]]
            doc_lengths = bm25_data["doc_lengths"]
            avg_doc_len = bm25_data["avg_doc_len"]
            idf_dict = bm25_data["idf"]

            # Update cache
            self.term_freq_cache[name] = term_frequencies
            self.doc_len_cache[name] = doc_lengths
            self.avg_doc_len_cache[name] = avg_doc_len
            self.idf_cache[name] = idf_dict

            return True

        except Exception as e:
            logger.error(f"Error loading BM25 data: {e}")
            return False

    def keyword_match(self, query: str, store_name: str) -> Dict[int, float]:
        """
        Find exact keyword matches between query and stored texts.

        Args:
            query: The search query
            store_name: Name of the store to search

        Returns:
            Dictionary mapping document IDs to match scores
        """
        # Ensure store is loaded
        if store_name not in self.text_cache:
            _, metadata = self.load_store(store_name)
            if not metadata:
                return {}

        # Process query
        query = self.preprocess_text(query)
        if not query:
            return {}

        # Get stored texts
        texts = self.text_cache[store_name]

        # Process query for relevant terms
        query_doc = self.nlp(query)
        query_terms = []

        # Extract important terms from query
        for token in query_doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                query_terms.append(token.lemma_.lower())

        # Extract noun phrases for multi-word matching
        for chunk in query_doc.noun_chunks:
            chunk_text = self.preprocess_text(chunk.text)
            if len(chunk_text) > 3 and chunk_text not in query_terms:
                query_terms.append(chunk_text)

        # Find matches
        matches = {}

        for idx, text in enumerate(texts):
            if not text:
                continue

            score = 0.0

            # Check for full query match
            if query in text:
                score += 0.8  # High score for exact match

            # Check for term matches
            term_matches = 0
            for term in query_terms:
                if term in text:
                    term_matches += 1
                    score += 0.1  # Add for each matching term

            # Bonus for high coverage
            if query_terms:
                coverage = term_matches / len(query_terms)
                if coverage > 0.5:
                    score += 0.2 * coverage

            # Only include documents with non-zero scores
            if score > 0:
                matches[idx] = min(1.0, score)  # Cap at 1.0

        return matches

    def calculate_bm25_scores(self, query: str, store_name: str) -> Dict[int, float]:
        """
        Calculate BM25 relevance scores.

        Args:
            query: The search query
            store_name: Name of the store to search

        Returns:
            Dictionary mapping document IDs to BM25 scores
        """
        # Ensure BM25 data is loaded
        if store_name not in self.term_freq_cache:
            self._load_bm25_data(store_name)
            if store_name not in self.term_freq_cache:
                return {}

        # Get BM25 parameters
        term_frequencies = self.term_freq_cache[store_name]
        doc_lengths = self.doc_len_cache[store_name]
        avg_doc_len = self.avg_doc_len_cache[store_name]
        idf_dict = self.idf_cache[store_name]

        # Process query
        query_terms = self.tokenize(query)

        # Calculate BM25 score for each document
        scores = {}

        for idx, doc_tf in enumerate(term_frequencies):
            score = 0.0
            doc_len = doc_lengths[idx]

            # Skip empty documents
            if doc_len == 0:
                continue

            # Calculate score for each query term
            for term in query_terms:
                if term not in idf_dict:
                    continue

                # Get term frequency in document
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue

                # Calculate BM25 score for term
                idf = idf_dict[term]
                term_score = idf * ((tf * (self.k1 + 1)) /
                                    (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)))

                score += term_score

            # Only include documents with non-zero scores
            if score > 0:
                # Normalize to 0-1 range
                scores[idx] = min(1.0, score / 10.0)  # Divisor adjusts scale

        return scores

    def calculate_contextual_similarity(self, query: str, text: str) -> float:
        """
        Calculate contextual similarity between query and text.

        Args:
            query: The search query
            text: The text to compare

        Returns:
            Similarity score between 0 and 1
        """
        # Process inputs
        query = self.preprocess_text(query)
        text = self.preprocess_text(text)

        if not query or not text:
            return 0.0

        # Process with spaCy
        query_doc = self.nlp(query)
        text_doc = self.nlp(text[:5000])  # Limit for performance

        # Skip if either has no vector
        if not query_doc.has_vector or not text_doc.has_vector:
            return 0.0

        # Calculate cosine similarity
        similarity = query_doc.similarity(text_doc)

        # Calculate term overlap
        query_terms = set(token.lemma_.lower() for token in query_doc
                          if not token.is_stop and not token.is_punct)
        text_terms = set(token.lemma_.lower() for token in text_doc
                         if not token.is_stop and not token.is_punct)

        term_overlap = 0.0
        if query_terms:
            term_overlap = len(query_terms.intersection(text_terms)) / len(query_terms)

        # Weighted combination
        contextual_score = 0.6 * similarity + 0.4 * term_overlap

        return max(0.0, min(1.0, contextual_score))

    def search(self, name: str, query: str, top_k: int = 3,
               threshold: float = 0.2, query_type: str = 'general') -> List[Dict[str, Any]]:
        """
        Search a store with combined scoring mechanisms.

        Args:
            name: Name of the store to search
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum score threshold
            query_type: Type of query ('faq', 'error_code', 'general')

        Returns:
            List of results with scores
        """
        # Load store
        index, metadata = self.load_store(name)
        if index is None or metadata is None:
            return []

        # Check query length
        if len(query.strip()) < 3:
            return []

        # Get keyword matches
        keyword_scores = self.keyword_match(query, name)

        # Vector search
        query_vector = self.vectorize_text(query, query_type, self.idf_cache.get(name))
        query_vector_np = np.array([query_vector]).astype('float32')

        # Get more results for reranking
        search_k = min(top_k * 3, index.ntotal)
        D, I = index.search(query_vector_np, search_k)

        # Process vector results
        vector_scores = {}
        for i, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(metadata):
                continue

            # Convert similarity to score in [0,1]
            similarity = float(D[0][i])
            vector_scores[idx] = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]

        # Calculate BM25 scores
        bm25_scores = self.calculate_bm25_scores(query, name)

        # Collect all candidate documents
        candidates = set()
        candidates.update(vector_scores.keys())
        candidates.update(bm25_scores.keys())
        candidates.update(keyword_scores.keys())

        # Calculate contextual similarity for candidates
        contextual_scores = {}
        for idx in candidates:
            if idx >= len(metadata):
                continue

            # Get text for contextual comparison
            text = metadata[idx].get('processed_text', '')
            if not text and idx < len(self.text_cache.get(name, [])):
                text = self.text_cache[name][idx]

            if text:
                contextual_scores[idx] = self.calculate_contextual_similarity(query, text)

        # Determine weights based on query type
        if query_type == 'faq':
            weights = {
                'vector': 0.3,
                'bm25': 0.4,
                'keyword': 0.2,
                'contextual': 0.1
            }
        elif query_type == 'error_code':
            weights = {
                'vector': 0.2,
                'bm25': 0.3,
                'keyword': 0.4,  # Error codes often need exact matching
                'contextual': 0.1
            }
        else:
            weights = {
                'vector': 0.3,
                'bm25': 0.4,
                'keyword': 0.2,
                'contextual': 0.1
            }

        # Calculate combined scores
        combined_scores = {}
        for idx in candidates:
            # Get component scores
            vs = vector_scores.get(idx, 0.0)
            bs = bm25_scores.get(idx, 0.0)
            ks = keyword_scores.get(idx, 0.0)
            cs = contextual_scores.get(idx, 0.0)

            # Weighted sum
            score = (vs * weights['vector'] +
                     bs * weights['bm25'] +
                     ks * weights['keyword'] +
                     cs * weights['contextual'])

            combined_scores[idx] = score

        # Filter by threshold
        filtered_results = []
        for idx, score in combined_scores.items():
            if score >= threshold and idx < len(metadata):
                result = metadata[idx].copy()

                # Add component scores for debugging
                result['score'] = score
                result['vector_score'] = vector_scores.get(idx, 0)
                result['bm25_score'] = bm25_scores.get(idx, 0)
                result['contextual_score'] = contextual_scores.get(idx, 0)
                result['keyword_match'] = idx in keyword_scores

                filtered_results.append(result)

        # Sort by score
        filtered_results.sort(key=lambda x: x['score'], reverse=True)

        return filtered_results[:top_k]

    def search_all_stores(self, query: str, top_k: int = 3, threshold: float = 0.2,
                          domain_prefix: Optional[str] = None, use_bm25: bool = True,
                          query_type: str = 'general') -> List[Dict[str, Any]]:
        """
        Search across all stores.

        Args:
            query: The search query
            top_k: Maximum results to return
            threshold: Minimum score threshold
            domain_prefix: Optional prefix to filter stores
            use_bm25: Whether to use BM25 scoring
            query_type: Type of query ('faq', 'error_code', 'general')

        Returns:
            List of results with scores
        """
        all_results = []

        # Find matching stores
        matching_stores = []
        for store_name in self.list_stores():
            if domain_prefix and not store_name.startswith(domain_prefix):
                continue
            matching_stores.append(store_name)

        logger.info(f"Searching across {len(matching_stores)} stores for '{query}'")

        # Search each store
        for store_name in matching_stores:
            store_results = self.search(
                store_name,
                query,
                top_k=top_k,
                threshold=threshold,
                query_type=query_type
            )

            # Add store name
            for result in store_results:
                result['store'] = store_name

            all_results.extend(store_results)

        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)

        return all_results[:top_k]

    def list_stores(self) -> List[str]:
        """List all available vector stores."""
        stores = []
        for file in os.listdir(self.store_dir):
            if file.endswith('.index'):
                stores.append(file[:-6])  # Remove .index extension
        return stores

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
                continue

            # Determine store type from name
            store_type = 'general'
            if 'faq' in name.lower():
                store_type = 'faq'
            elif 'error' in name.lower():
                store_type = 'error_code'

            # Create store with appropriate type
            self.create_store(name, all_chunks, store_type)