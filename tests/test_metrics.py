"""Unit tests for Live RAG Triad Metrics: Groundedness, Context Relevance, and Latency."""

import pytest
from langchain_core.documents import Document

from rag.metrics import (
    compute_context_relevance,
    compute_groundedness,
    compute_rag_triad,
)


def test_groundedness_exact_not_found_is_100_percent():
    """Verify that honest refusal fallback has 100% groundedness (zero hallucination)."""
    score = compute_groundedness(
        answer="The answer is not available in the provided documents.",
        context_docs=[],
    )
    assert score == 1.0


def test_groundedness_supported_sentences():
    """Verify high groundedness when answer sentences are supported by context."""
    docs = [
        Document(
            page_content="Dr. Alan Turing was an English mathematician and computer scientist who cracked Enigma.",
            metadata={"source": "history.pdf", "page": 1},
        )
    ]
    answer = "Alan Turing was an English mathematician and computer scientist [history.pdf, p.1]."
    score = compute_groundedness(answer=answer, context_docs=docs)
    assert score >= 0.8


def test_groundedness_hallucinated_answer():
    """Verify low groundedness when answer claims unsupported facts."""
    docs = [
        Document(
            page_content="This text only discusses baking sourdough bread and flour types.",
            metadata={"source": "recipes.pdf", "page": 1},
        )
    ]
    answer = "Quantum entanglement allows instant communication across light years according to physics [space.pdf, p.5]."
    score = compute_groundedness(answer=answer, context_docs=docs)
    assert score <= 0.4


def test_context_relevance_high_signal():
    """Verify high context relevance when chunks contain question keywords."""
    docs = [
        Document(
            page_content="PyTorch and TensorFlow are popular deep learning frameworks in computer vision.",
            metadata={"source": "dl.pdf", "page": 1},
        )
    ]
    score = compute_context_relevance(question="Which deep learning frameworks are used in computer vision?", context_docs=docs)
    assert score >= 0.6


def test_compute_rag_triad_output_structure():
    """Verify complete structure and formatted badges from compute_rag_triad."""
    docs = [
        Document(
            page_content="Candidate holds a Master of Science in Data Science from Stanford University.",
            metadata={"source": "cv.pdf", "page": 1},
        )
    ]
    answer = "The candidate graduated from Stanford University with a Master of Science in Data Science [cv.pdf, p.1]."

    triad = compute_rag_triad(
        question="Where did the candidate graduate from?",
        answer=answer,
        context_docs=docs,
        latency_ms=385.2,
        grade_score=0.95,
    )

    assert "groundedness_pct" in triad
    assert "context_relevance_pct" in triad
    assert "latency_int" in triad
    assert triad["latency_int"] == 385
    assert triad["groundedness_pct"] >= 80
    assert "Groundedness:" in triad["badge_markdown"]
    assert "Latency:" in triad["badge_markdown"]

