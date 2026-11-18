"""Configuration module for MLRAG."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application configuration parameters loaded from environment variables."""

    # LLM Settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    # Embeddings Settings
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

    # OCR Settings
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "eng+hin+tel+fra+deu+spa")
    min_page_char_count: int = int(os.getenv("MIN_PAGE_CHAR_COUNT", "50"))

    # Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval & Context Settings
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))

    # Vector Store Persistence
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")


def get_config() -> AppConfig:
    """Return the application configuration instance."""
    return AppConfig()

