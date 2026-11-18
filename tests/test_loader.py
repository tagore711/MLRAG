"""Unit tests for PDF loader and OCR fallback."""

import io
from unittest.mock import patch

import pypdf
from pypdf import PdfWriter
import pytest

from rag.loader import load_pdf, load_multiple_pdfs


def _create_synthetic_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with specified text per page."""
    writer = PdfWriter()
    for text in pages_text:
        # Create a blank page
        page = writer.add_blank_page(width=612, height=792)
        # Note: blank_page with no text extraction returns empty or minimal text in pypdf
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_empty_or_corrupt_pdf_handling():
    """Ensure invalid or corrupt PDF input does not crash the loader."""
    docs = load_pdf(b"not a valid pdf content", filename="corrupt.pdf")
    assert isinstance(docs, list)
    assert len(docs) == 0


def test_loader_extracts_per_page_and_preserves_metadata():
    """Verify that pages are loaded with accurate 'source' and 'page' metadata."""
    # Create 2 pages
    pdf_bytes = _create_synthetic_pdf(["Page 1 content", "Page 2 content"])

    # Mock pypdf page extract_text
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_reader = mock_reader_cls.return_value
        page1 = type("MockPage", (), {"extract_text": lambda *args, **kwargs: "This is a comprehensive overview of machine learning models that easily exceeds 50 characters threshold."})()
        page2 = type("MockPage", (), {"extract_text": lambda *args, **kwargs: "Another detailed paragraph describing multilingual vector search techniques and cosine distance."})()
        mock_reader.pages = [page1, page2]

        docs = load_pdf(pdf_bytes, filename="sample_ml.pdf")

        assert len(docs) == 2
        # Verify 1-indexed page numbering
        assert docs[0].metadata["source"] == "sample_ml.pdf"
        assert docs[0].metadata["page"] == 1
        assert "machine learning models" in docs[0].page_content

        assert docs[1].metadata["source"] == "sample_ml.pdf"
        assert docs[1].metadata["page"] == 2
        assert "multilingual vector search" in docs[1].page_content


def test_ocr_fallback_triggered_for_low_text_pages():
    """Verify that OCR fallback is triggered when extracted text length < 50 chars."""
    pdf_bytes = _create_synthetic_pdf(["Short"])

    with patch("pypdf.PdfReader") as mock_reader_cls, \
         patch("rag.loader._extract_page_with_ocr") as mock_ocr:

        mock_reader = mock_reader_cls.return_value
        page1 = type("MockPage", (), {"extract_text": lambda *args, **kwargs: "Short text"})()  # 10 chars < 50
        mock_reader.pages = [page1]
        mock_ocr.return_value = "Recovered text via OCR fallback after scan analysis."

        docs = load_pdf(pdf_bytes, filename="scanned_doc.pdf", min_char_count=50)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "scanned_doc.pdf"
        assert docs[0].metadata["page"] == 1
        assert docs[0].page_content == "Recovered text via OCR fallback after scan analysis."
        mock_ocr.assert_called_once()


def test_ocr_fallback_not_triggered_when_text_exceeds_threshold():
    """Verify OCR fallback is bypassed when text exceeds min_char_count."""
    pdf_bytes = _create_synthetic_pdf(["Long text"])

    with patch("pypdf.PdfReader") as mock_reader_cls, \
         patch("rag.loader._extract_page_with_ocr") as mock_ocr:

        mock_reader = mock_reader_cls.return_value
        long_content = "This page contains ample text directly parsed by pypdf and should not trigger OCR fallback."
        page1 = type("MockPage", (), {"extract_text": lambda *args, **kwargs: long_content})()
        mock_reader.pages = [page1]

        docs = load_pdf(pdf_bytes, filename="clean_doc.pdf", min_char_count=50)

        assert len(docs) == 1
        assert docs[0].page_content == long_content
        mock_ocr.assert_not_called()


def test_load_multiple_pdfs():
    """Verify batch loading across multiple PDF sources."""
    pdf1 = _create_synthetic_pdf(["Doc1"])
    pdf2 = _create_synthetic_pdf(["Doc2"])

    with patch("rag.loader.load_pdf") as mock_load:
        doc1 = type("Doc", (), {"page_content": "P1", "metadata": {"source": "a.pdf", "page": 1}})
        doc2 = type("Doc", (), {"page_content": "P2", "metadata": {"source": "b.pdf", "page": 1}})
        mock_load.side_effect = [[doc1], [doc2]]

        all_docs = load_multiple_pdfs([pdf1, pdf2], filenames=["a.pdf", "b.pdf"])
        assert len(all_docs) == 2
        assert mock_load.call_count == 2

