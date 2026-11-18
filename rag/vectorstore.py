"""Vector store and embedding management for MLRAG using FAISS."""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Optimize CPU inference speed and disable autograd graph construction
torch.set_grad_enabled(False)

from rag.config import get_config

logger = logging.getLogger(__name__)


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Initialize HuggingFaceEmbeddings with normalized vectors for cosine similarity.

    Args:
        model_name: HuggingFace model identifier. Defaults to config (BAAI/bge-m3).

    Returns:
        HuggingFaceEmbeddings instance with normalize_embeddings=True.
    """
    config = get_config()
    target_model = model_name or config.embed_model

    logger.info(f"Initializing embedding model: {target_model}")
    return HuggingFaceEmbeddings(
        model_name=target_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    index_path: Optional[str] = None,
) -> Optional[FAISS]:
    """Load persisted FAISS index from disk if it exists.

    Args:
        embeddings: Embeddings instance used by the index.
        index_path: Path to the FAISS index folder. Defaults to config path.

    Returns:
        Loaded FAISS vectorstore instance, or None if index does not exist.
    """
    config = get_config()
    target_path = Path(index_path or config.faiss_index_path)

    index_file = target_path / "index.faiss"
    pkl_file = target_path / "index.pkl"

    if index_file.exists() and pkl_file.exists():
        try:
            logger.info(f"Loading FAISS index from {target_path}")
            return FAISS.load_local(
                folder_path=str(target_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Error loading FAISS index from {target_path}: {e}")
            return None

    logger.info(f"No existing FAISS index found at {target_path}")
    return None


def create_or_update_vectorstore(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_path: Optional[str] = None,
) -> Optional[FAISS]:
    """Create a new FAISS vector store or merge documents into existing persisted index.

    Args:
        documents: List of chunked Document objects to embed.
        embeddings: HuggingFaceEmbeddings instance.
        index_path: Destination folder for index persistence.

    Returns:
        FAISS vectorstore instance, or None if no documents provided.
    """
    if not documents:
        logger.warning("No documents provided to create_or_update_vectorstore.")
        return None

    config = get_config()
    target_path = Path(index_path or config.faiss_index_path)

    existing_vs = load_vectorstore(embeddings=embeddings, index_path=str(target_path))

    if existing_vs is not None:
        logger.info(f"Adding {len(documents)} document chunks to existing FAISS index.")
        existing_vs.add_documents(documents)
        vectorstore = existing_vs
    else:
        logger.info(f"Creating new FAISS index with {len(documents)} document chunks.")
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    # Persist index to disk
    target_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target_path))
    logger.info(f"FAISS index saved successfully to {target_path}")

    return vectorstore


def clear_vectorstore(index_path: Optional[str] = None) -> bool:
    """Clear and delete persisted FAISS index from disk.

    Args:
        index_path: Directory path of the index to delete.

    Returns:
        True if index was removed or did not exist, False on failure.
    """
    config = get_config()
    target_path = Path(index_path or config.faiss_index_path)

    if target_path.exists():
        try:
            shutil.rmtree(target_path)
            logger.info(f"Successfully deleted FAISS index directory: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete FAISS index at {target_path}: {e}")
            return False
    return True

