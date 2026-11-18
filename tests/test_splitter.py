"""Unit tests for document chunking and metadata preservation."""

import pytest
from langchain_core.documents import Document

from rag.splitter import split_documents


def test_split_preserves_metadata():
    """Verify that every generated chunk preserves the original 'source' and 'page' metadata."""
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an architecture that optimizes the output of an LLM. "
        "By referencing an authoritative knowledge base outside of its training data sources, "
        "the LLM can provide more relevant and up-to-date responses. "
    ) * 10  # Long text to force splitting

    doc = Document(
        page_content=sample_text,
        metadata={"source": "report_2026.pdf", "page": 4},
    )

    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "report_2026.pdf"
        assert chunk.metadata["page"] == 4
        assert len(chunk.page_content) <= 300


def test_split_multiple_documents_with_different_pages():
    """Ensure chunking across multiple pages properly maintains distinct page numbers."""
    docs = [
        Document(
            page_content="Content from page one of the document discussing multilingual NLP.",
            metadata={"source": "nlp.pdf", "page": 1},
        ),
        Document(
            page_content="Content from page two focusing on vector indexes and cosine distance metrics.",
            metadata={"source": "nlp.pdf", "page": 2},
        ),
    ]

    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 1
    assert chunks[1].metadata["page"] == 2


def test_split_filters_empty_or_whitespace_chunks():
    """Ensure blank or whitespace-only pages/chunks are discarded."""
    docs = [
        Document(page_content="   \n\n   ", metadata={"source": "empty.pdf", "page": 1}),
        Document(page_content="", metadata={"source": "empty.pdf", "page": 2}),
        Document(page_content="Valid actionable content.", metadata={"source": "empty.pdf", "page": 3}),
    ]

    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Valid actionable content."
    assert chunks[0].metadata["page"] == 3

