"""Unit tests for Hybrid BM25 Retrieval and Reciprocal Rank Fusion."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from rag.hybrid import CrossEncoderReranker, SimpleBM25, reciprocal_rank_fusion


def test_simple_bm25_search():
    """Verify keyword matching with SimpleBM25."""
    docs = [
        Document(page_content="Deep neural networks for image classification.", metadata={"page": 1}),
        Document(page_content="Retrieval-augmented generation for multilingual NLP.", metadata={"page": 2}),
        Document(page_content="Quantum computing algorithms and qubits.", metadata={"page": 3}),
    ]

    bm25 = SimpleBM25(docs)
    results = bm25.search("multilingual NLP", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["page"] == 2


def test_reciprocal_rank_fusion():
    """Verify RRF merges two candidate lists and ranks overlap higher."""
    doc_a = Document(page_content="Doc A content", metadata={"source": "a.pdf", "page": 1})
    doc_b = Document(page_content="Doc B content", metadata={"source": "b.pdf", "page": 1})
    doc_c = Document(page_content="Doc C content", metadata={"source": "c.pdf", "page": 1})

    # List 1: [B, A]
    # List 2: [C, B]
    # Doc B appears in both lists, so its RRF score is 1/(60+1) + 1/(60+2), which should place it first!
    list1 = [doc_b, doc_a]
    list2 = [doc_c, doc_b]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=2)
    assert len(fused) == 2
    assert fused[0].page_content == "Doc B content"


def test_cross_encoder_reranker_mock():
    """Verify cross-encoder reranking ordering with mocked scores."""
    doc1 = Document(page_content="Low relevance snippet", metadata={"page": 1})
    doc2 = Document(page_content="High relevance exact match", metadata={"page": 2})

    reranker = CrossEncoderReranker(model_name="test-model")
    mock_model = MagicMock()
    # Return higher score for doc2
    mock_model.predict.return_value = [0.1, 0.95]
    reranker._model = mock_model

    reranked = reranker.rerank(query="exact match", documents=[doc1, doc2], top_k=2)
    assert len(reranked) == 2
    assert reranked[0].metadata["page"] == 2
    assert reranked[1].metadata["page"] == 1

