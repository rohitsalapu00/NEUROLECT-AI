"""
NEUROLECT AI - Test PDF Processor
Validates PyMuPDF extraction, page tracking, corrupted PDF handling, and empty file detection.
"""

import pytest
from pathlib import Path
import pymupdf as fitz
from modules.pdf_processor import PDFProcessor, PDFProcessingError
from modules.content_models import LectureContent


@pytest.fixture
def pdf_processor():
    return PDFProcessor()


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a synthetic 2-page PDF with realistic lecture content."""
    pdf_path = tmp_path / "test_lecture.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(50, 72),
        "Lecture 1: Introduction to Recurrent Neural Networks.\n"
        "RNNs are designed to recognize patterns in sequences of data such as text, genomes, or stock markets.\n"
        "Traditional neural networks assume all inputs and outputs are independent of each other."
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        fitz.Point(50, 72),
        "Lecture 2: The Vanishing Gradient Problem.\n"
        "As recurrent connections backpropagate through time, gradients can exponentially diminish or explode.\n"
        "Long Short-Term Memory (LSTM) networks resolve this using gating mechanisms: input, forget, and output gates."
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_extract_valid_pdf(pdf_processor, sample_pdf):
    """Verify text extraction from multi-page PDF with exact page number preservation."""
    result = pdf_processor.extract_text(sample_pdf, title="RNN and LSTM Lecture")

    assert isinstance(result, LectureContent)
    assert result.source_type == "pdf"
    assert result.title == "RNN and LSTM Lecture"
    assert result.source_filename == "test_lecture.pdf"
    assert result.total_segments == 2
    assert result.total_words > 30

    # Verify page preservation
    assert result.segments[0].page_number == 1
    assert "Recurrent Neural Networks" in result.segments[0].text
    assert result.segments[1].page_number == 2
    assert "Vanishing Gradient" in result.segments[1].text


def test_extract_empty_pdf(tmp_path, pdf_processor):
    """Verify that a PDF without any readable text raises PDFProcessingError."""
    empty_pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page
    doc.save(str(empty_pdf))
    doc.close()

    with pytest.raises(PDFProcessingError):
        pdf_processor.extract_text(empty_pdf)


def test_extract_corrupted_pdf(tmp_path, pdf_processor):
    """Verify that a corrupted PDF raises PDFProcessingError."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 random corrupt binary payload not a valid pdf header")

    with pytest.raises(PDFProcessingError):
        pdf_processor.extract_text(corrupt_pdf)


def test_extract_nonexistent_pdf(pdf_processor):
    """Verify non-existent PDF raises PDFProcessingError."""
    with pytest.raises(PDFProcessingError):
        pdf_processor.extract_text("does_not_exist.pdf")


def test_extract_invalid_extension(tmp_path, pdf_processor):
    """Verify non-PDF file raises PDFProcessingError."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("Not a PDF file", encoding="utf-8")

    with pytest.raises(PDFProcessingError):
        pdf_processor.extract_text(txt_file)
