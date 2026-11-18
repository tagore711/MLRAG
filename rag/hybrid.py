"""Hybrid Retrieval (BM25 + FAISS via Reciprocal Rank Fusion) and Cross-Encoder Reranking."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


def tokenize(text: str) -> List[str]:
    """Simple multilingual word/character tokenizer."""
    # Match alphanumeric sequences across scripts (including non-Latin)
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    return tokens


class SimpleBM25:
    """Pure-Python, zero-dependency BM25 Okapi implementation supporting multilingual text."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)
        self.doc_lengths: List[int] = []
        self.avg_doc_len = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Counter] = []

        self._initialize()

    def _initialize(self):
        total_len = 0
        for doc in self.documents:
            tokens = tokenize(doc.page_content)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_len += length

            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)

    def _idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        return math.log(((self.corpus_size - n + 0.5) / (n + 0.5)) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self._idf(term)
            for idx, tf_dict in enumerate(self.term_freqs):
                freq = tf_dict.get(term, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[idx] / self.avg_doc_len))
                scores[idx] += idf * (num / den)
        return scores

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        tokens = tokenize(query)
        if not tokens or self.corpus_size == 0:
            return []
        scores = self.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked_indices[:top_k] if scores[i] > 0]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 5,
) -> List[Document]:
    """Merge multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).

    Score(d) = sum(1 / (k + rank_i(d))) for each ranked list.

    Args:
        ranked_lists: List of document candidate lists from different retrievers.
        k: Smoothing constant (standard: 60).
        top_n: Number of fused results to return.

    Returns:
        Deduplicated, re-ranked list of top_n Document objects.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in ranked_lists:
        for rank, doc in enumerate(doc_list, start=1):
            # Unique identifier based on source, page, and content prefix
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            doc_id = f"{source}::{page}::{doc.page_content[:80]}"

            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort descending by RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_n]]


class CrossEncoderReranker:
    """Reranker using BAAI/bge-reranker-v2-m3 or sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Rerank document chunks using cross-attention relevance scores."""
        if not documents:
            return []
        try:
            model = self._load_model()
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception:
            # Fallback gracefully to original order if cross-encoder fails to load
            return documents[:top_k]


def compare_and_retrieve(
    query: str,
    vectorstore: Any,
    bm25: Optional[SimpleBM25] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    use_hybrid: bool = True,
    use_reranker: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve and benchmark across Dense (FAISS), Sparse (BM25), Hybrid (RRF), and Reranker.

    Args:
        query: User question.
        vectorstore: FAISS vectorstore instance.
        bm25: Optional pre-built SimpleBM25 index.
        reranker: Optional CrossEncoderReranker instance.
        use_hybrid: Whether to use RRF fused results as the primary context.
        use_reranker: Whether to apply cross-encoder re-ranking.
        top_k: Number of chunks to select.

    Returns:
        Dictionary with 'chosen', 'dense', 'sparse', 'hybrid', 'reranked', and 'stats'.
    """
    # 1. Dense retrieval via FAISS
    dense_docs = vectorstore.similarity_search(query=query, k=top_k) if vectorstore else []

    # 2. Sparse retrieval via BM25
    sparse_docs = bm25.search(query=query, top_k=top_k) if bm25 else []

    # 3. Hybrid fusion via Reciprocal Rank Fusion (RRF)
    if dense_docs and sparse_docs:
        hybrid_docs = reciprocal_rank_fusion([dense_docs, sparse_docs], k=60, top_n=top_k)
    elif dense_docs:
        hybrid_docs = dense_docs
    else:
        hybrid_docs = sparse_docs

    # Select base candidates
    candidates = hybrid_docs if use_hybrid else dense_docs

    # 4. Cross-Encoder Reranking
    reranked_docs = []
    if use_reranker and reranker and candidates:
        reranked_docs = reranker.rerank(query=query, documents=candidates, top_k=top_k)
        chosen = reranked_docs
    else:
        chosen = candidates

    # Compute overlap stats between dense and sparse
    def _doc_id(d):
        return f"{d.metadata.get('source', '')}::p{d.metadata.get('page', '')}"

    dense_set = {_doc_id(d) for d in dense_docs}
    sparse_set = {_doc_id(d) for d in sparse_docs}
    overlap = len(dense_set & sparse_set)

    return {
        "chosen": chosen,
        "dense": dense_docs,
        "sparse": sparse_docs,
        "hybrid": hybrid_docs,
        "reranked": reranked_docs,
        "stats": {
            "dense_count": len(dense_docs),
            "sparse_count": len(sparse_docs),
            "overlap_count": overlap,
            "hybrid_enabled": use_hybrid,
            "reranker_enabled": use_reranker,
        },
    }

