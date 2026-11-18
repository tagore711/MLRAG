"""Unit tests for RAG chain, context capping, and answer generation."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from rag.chain import (
    EXACT_NOT_FOUND_RESPONSE,
    format_context_and_cap,
    invoke_chain_with_backoff,
    query_rag,
)


def test_format_context_and_cap_within_limit():
    """Verify that context formatter respects the max_tokens cap."""
    docs = [
        Document(
            page_content="Sentence A describing multilingual machine translation." * 30,
            metadata={"source": "paper1.pdf", "page": 1},
        ),
        Document(
            page_content="Sentence B describing neural search indexing." * 30,
            metadata={"source": "paper2.pdf", "page": 2},
        ),
        Document(
            page_content="Sentence C describing cross-lingual embedding spaces." * 50,
            metadata={"source": "paper3.pdf", "page": 5},
        ),
    ]

    # Set a tight token cap (e.g., 200 tokens)
    formatted_context, included_docs = format_context_and_cap(docs, max_tokens=200)

    assert "--- Source: [paper1.pdf, p.1] ---" in formatted_context
    assert len(included_docs) < len(docs)  # Should have stopped before including all 3 docs


def test_answer_not_found_when_vectorstore_empty():
    """Verify that query_rag returns exact fallback if no vector store or no results."""
    # Test with None vectorstore
    res = query_rag(question="What is the capital of Mars?", vectorstore=None)
    assert res["answer"] == EXACT_NOT_FOUND_RESPONSE
    assert res["source_documents"] == []


def test_answer_not_found_when_no_matching_chunks():
    """Verify exact fallback when similarity search returns empty."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []

    res = query_rag(question="Where is Atlantis?", vectorstore=mock_vs)
    assert res["answer"] == EXACT_NOT_FOUND_RESPONSE
    assert res["source_documents"] == []


def test_generation_with_citations_and_mock_llm():
    """Verify standard answering pipeline with mock LLM and source citation format."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="Bilingual BERT supports cross-lingual retrieval across 100 languages.",
            metadata={"source": "bert_paper.pdf", "page": 3},
        )
    ]

    mock_llm = MagicMock()
    # Mock LLM returning expected formatted citation answer
    mock_llm.invoke.return_value = type(
        "MockAIMessage", (),
        {"content": "Bilingual BERT supports 100 languages for cross-lingual tasks [bert_paper.pdf, p.3]."}
    )()

    with patch("rag.chain.create_lcel_chain") as mock_chain_builder:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Bilingual BERT supports 100 languages for cross-lingual tasks [bert_paper.pdf, p.3]."
        mock_chain_builder.return_value = mock_chain

        res = query_rag(
            question="How many languages does Bilingual BERT support?",
            vectorstore=mock_vs,
            llm=mock_llm,
        )

        assert "[bert_paper.pdf, p.3]" in res["answer"]
        assert len(res["source_documents"]) == 1
        assert res["source_documents"][0].metadata["source"] == "bert_paper.pdf"


def test_exact_not_found_response_from_llm():
    """Verify that when context does not contain answer, LLM output is preserved verbatim."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="This text only discusses baking recipes and bread.",
            metadata={"source": "cooking.pdf", "page": 1},
        )
    ]

    with patch("rag.chain.create_lcel_chain") as mock_chain_builder:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = EXACT_NOT_FOUND_RESPONSE
        mock_chain_builder.return_value = mock_chain

        res = query_rag(
            question="What is quantum entanglement?",
            vectorstore=mock_vs,
        )

        assert res["answer"] == EXACT_NOT_FOUND_RESPONSE


def test_groq_retry_on_rate_limit():
    """Verify exponential backoff retry on HTTP 429 rate-limit error."""
    mock_chain = MagicMock()
    # Fails first with 429, succeeds on retry
    mock_chain.invoke.side_effect = [
        Exception("Error 429: Rate limit reached. Please retry after 1s."),
        "Success answer after retry [doc.pdf, p.1].",
    ]

    with patch("time.sleep") as mock_sleep:
        answer = invoke_chain_with_backoff(
            chain=mock_chain,
            inputs={"context": "some context", "question": "test question"},
            max_retries=2,
            initial_delay=0.1,
        )

        assert answer == "Success answer after retry [doc.pdf, p.1]."
        assert mock_chain.invoke.call_count == 2
        mock_sleep.assert_called_once_with(0.1)

