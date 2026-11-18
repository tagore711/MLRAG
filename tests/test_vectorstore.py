"""Unit tests for FAISS vectorstore persistence and index operations."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from rag.vectorstore import (
    clear_vectorstore,
    create_or_update_vectorstore,
    load_vectorstore,
)


class MockEmbeddings:
    """Mock embeddings generating deterministic 4-dimensional normalized vectors."""

    def embed_documents(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    def embed_query(self, text):
        return [0.5, 0.5, 0.5, 0.5]

    def __call__(self, text):
        return self.embed_query(text)


def test_vectorstore_creation_persistence_and_reload(tmp_path):
    """Verify FAISS vector store creation, saving to disk, and loading back."""
    test_index_dir = str(tmp_path / "test_faiss_index")
    embeddings = MockEmbeddings()

    docs = [
        Document(
            page_content="Multilingual AI model enables cross-lingual knowledge retrieval.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
        Document(
            page_content="FAISS provides efficient dense vector similarity indexing.",
            metadata={"source": "doc2.pdf", "page": 3},
        ),
    ]

    # Create new vectorstore
    vs = create_or_update_vectorstore(
        documents=docs,
        embeddings=embeddings,
        index_path=test_index_dir,
    )
    assert vs is not None

    # Check files exist
    assert (Path(test_index_dir) / "index.faiss").exists()
    assert (Path(test_index_dir) / "index.pkl").exists()

    # Load from disk
    loaded_vs = load_vectorstore(embeddings=embeddings, index_path=test_index_dir)
    assert loaded_vs is not None

    # Similarity search on loaded vector store
    results = loaded_vs.similarity_search("AI model", k=1)
    assert len(results) == 1
    assert "doc1.pdf" in results[0].metadata["source"]


def test_vectorstore_clear(tmp_path):
    """Verify that clear_vectorstore completely deletes persisted index directory."""
    test_index_dir = str(tmp_path / "test_faiss_index_to_delete")
    embeddings = MockEmbeddings()

    docs = [Document(page_content="Test data", metadata={"source": "a.pdf", "page": 1})]
    create_or_update_vectorstore(documents=docs, embeddings=embeddings, index_path=test_index_dir)
    assert Path(test_index_dir).exists()

    success = clear_vectorstore(index_path=test_index_dir)
    assert success is True
    assert not Path(test_index_dir).exists()

