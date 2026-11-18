"""PDF document loader with OCR fallback for MLRAG."""

import io
import logging
import os
from typing import BinaryIO, List, Optional, Union

import pypdf
from langchain_core.documents import Document

from rag.config import get_config

logger = logging.getLogger(__name__)


def _extract_page_with_ocr(
    pdf_bytes: bytes,
    page_number: int,
    ocr_languages: str,
) -> str:
    """Fallback OCR extraction for a single PDF page using pdf2image and pytesseract.

    Args:
        pdf_bytes: Raw bytes of the PDF.
        page_number: 1-indexed page number to extract.
        ocr_languages: Plus-separated language string for tesseract (e.g., 'eng+hin').

    Returns:
        Extracted text via OCR, or empty string if OCR failed or binaries missing.
    """
    try:
        import pdf2image
        import pytesseract

        images = pdf2image.convert_from_bytes(
            pdf_bytes,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
        if not images:
            return ""

        ocr_text = pytesseract.image_to_string(images[0], lang=ocr_languages)
        return ocr_text.strip() if ocr_text else ""
    except Exception as e:
        logger.warning(
            f"OCR fallback failed for page {page_number} ({e}). "
            "Ensure Tesseract OCR and Poppler binaries are installed if OCR is required."
        )
        return ""


def load_pdf(
    file_source: Union[str, BinaryIO, bytes],
    filename: Optional[str] = None,
    ocr_languages: Optional[str] = None,
    min_char_count: Optional[int] = None,
) -> List[Document]:
    """Extract text per page from a PDF with OCR fallback for low-text pages.

    Preserves metadata: {'source': filename, 'page': page_number} (1-indexed).

    Args:
        file_source: File path (str), file-like object, or raw PDF bytes.
        filename: Name of the source file for metadata (inferred if file_source is str).
        ocr_languages: Tesseract OCR language string. Defaults to config if None.
        min_char_count: Minimum characters threshold below which OCR is triggered.

    Returns:
        List of LangChain Document objects with page text and metadata.
    """
    config = get_config()
    langs = ocr_languages or config.ocr_languages
    threshold = min_char_count if min_char_count is not None else config.min_page_char_count

    # Determine filename
    if filename:
        source_name = filename
    elif isinstance(file_source, str):
        source_name = os.path.basename(file_source)
    elif hasattr(file_source, "name"):
        source_name = getattr(file_source, "name")
    else:
        source_name = "document.pdf"

    # Extract bytes for potential OCR and pypdf stream
    if isinstance(file_source, str):
        with open(file_source, "rb") as f:
            pdf_bytes = f.read()
    elif isinstance(file_source, bytes):
        pdf_bytes = file_source
    else:
        # File-like object (e.g. BytesIO or Streamlit UploadedFile)
        file_source.seek(0)
        pdf_bytes = file_source.read()
        file_source.seek(0)

    documents: List[Document] = []

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF '{source_name}': {e}")
        return documents

    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        page_text = ""
        try:
            extracted = page.extract_text()
            if extracted:
                page_text = extracted.strip()
        except Exception as e:
            logger.warning(f"Error extracting text from '{source_name}' p.{page_number}: {e}")

        # Check if text length is below threshold to trigger OCR fallback
        if len(page_text) < threshold:
            logger.info(
                f"Page {page_number} of '{source_name}' has only {len(page_text)} chars (<{threshold}). "
                "Triggering OCR fallback..."
            )
            ocr_result = _extract_page_with_ocr(
                pdf_bytes=pdf_bytes,
                page_number=page_number,
                ocr_languages=langs,
            )
            if ocr_result:
                page_text = ocr_result

        # Even if page text is empty (e.g. blank page or failed OCR), record document cleanly
        doc = Document(
            page_content=page_text,
            metadata={
                "source": source_name,
                "page": page_number,
            },
        )
        documents.append(doc)

    return documents


def load_multiple_pdfs(
    files: List[Union[str, BinaryIO, bytes]],
    filenames: Optional[List[str]] = None,
    ocr_languages: Optional[str] = None,
    min_char_count: Optional[int] = None,
) -> List[Document]:
    """Load multiple PDFs and aggregate their extracted pages with metadata.

    Args:
        files: List of file paths, file-like objects, or bytes.
        filenames: Optional matching list of filenames.
        ocr_languages: OCR language string.
        min_char_count: OCR threshold.

    Returns:
        List of all Document objects across all PDFs.
    """
    all_docs: List[Document] = []
    for i, file_obj in enumerate(files):
        fname = filenames[i] if filenames and i < len(filenames) else None
        docs = load_pdf(
            file_source=file_obj,
            filename=fname,
            ocr_languages=ocr_languages,
            min_char_count=min_char_count,
        )
        all_docs.extend(docs)
    return all_docs

