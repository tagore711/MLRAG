"""Document text chunking module for MLRAG."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_config


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into smaller chunks while preserving per-page metadata.

    Uses RecursiveCharacterTextSplitter with configurable chunk size and overlap.
    Filters out empty or whitespace-only chunks.

    Args:
        documents: List of LangChain Document objects to split.
        chunk_size: Maximum characters per chunk (default from config: 1000).
        chunk_overlap: Overlap between consecutive chunks (default from config: 150).

    Returns:
        List of chunked Document objects retaining original 'source' and 'page' metadata.
    """
    config = get_config()
    c_size = chunk_size if chunk_size is not None else config.chunk_size
    c_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # Filter out empty pages before splitting
    valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        return []

    split_docs = splitter.split_documents(valid_docs)

    # Clean up and ensure metadata is intact
    cleaned_chunks: List[Document] = []
    for chunk in split_docs:
        text = chunk.page_content.strip()
        if text:
            # Preserve original metadata
            cleaned_chunks.append(
                Document(
                    page_content=text,
                    metadata=dict(chunk.metadata),
                )
            )

    return cleaned_chunks

