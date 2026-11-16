"""
NEUROLECT AI - PDF Processing Module
Extracts educational content from PDF lecture slides and textbooks using PyMuPDF (fitz),
preserving page numbers and structure in the unified LectureContent format.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union, List
import pymupdf as fitz
from config import SUPPORTED_DOC_FORMATS, MAX_UPLOAD_SIZE_MB
from utils.validators import validate_file_extension, validate_file_size, validate_text_content, ValidationError
from utils.helpers import clean_text
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments

logger = logging.getLogger("NEUROLECT.PDFProcessor")


class PDFProcessingError(Exception):
    """Raised when PDF reading, text extraction, or document parsing fails."""
    pass


class PDFProcessor:
    """
    Extracts text from PDF documents page by page using PyMuPDF, preserving
    page numbers and assembling standardized LectureContent for the NLP engine.
    """

    def validate_pdf_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validates file existence, .pdf extension, and file size.
        """
        path = Path(file_path)
        if not path.exists():
            raise PDFProcessingError(f"PDF file not found: {path}")

        try:
            validate_file_extension(path.name, [".pdf"])
            validate_file_size(path, MAX_UPLOAD_SIZE_MB)
        except ValidationError as e:
            raise PDFProcessingError(str(e)) from e

        return path

    def extract_text(
        self,
        pdf_path: Union[str, Path],
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Extracts text from each page of a PDF document using PyMuPDF.
        Preserves page provenance and converts into unified LectureContent.
        """
        valid_path = self.validate_pdf_file(pdf_path)
        doc_title = title or valid_path.stem.replace("_", " ").title()

        if progress_callback:
            progress_callback(0.10, f"Opening PDF document '{valid_path.name}'...")

        try:
            doc = fitz.open(valid_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {valid_path}: {e}")
            raise PDFProcessingError(
                f"Could not open '{valid_path.name}'. The PDF file may be corrupted, encrypted, or malformed."
            ) from e

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise PDFProcessingError(f"PDF '{valid_path.name}' contains 0 pages.")

        segments: List[ContentSegment] = []
        page_texts: List[str] = []
        total_extracted_chars = 0

        logger.info(f"Extracting text from {total_pages} pages in '{valid_path.name}'...")

        for page_idx in range(total_pages):
            try:
                page = doc[page_idx]
                raw_page_text = page.get_text("text")
                cleaned_page_text = clean_text(raw_page_text)
                page_num = page_idx + 1

                if cleaned_page_text:
                    segments.append(ContentSegment(
                        segment_id=page_idx,
                        text=cleaned_page_text,
                        source_type="pdf",
                        page_number=page_num,
                        metadata={
                            "char_count": len(cleaned_page_text),
                            "word_count": len(cleaned_page_text.split()),
                        }
                    ))
                    page_texts.append(cleaned_page_text)
                    total_extracted_chars += len(cleaned_page_text)

                if progress_callback and (page_idx % max(1, total_pages // 10) == 0 or page_idx == total_pages - 1):
                    progress = 0.10 + (0.70 * ((page_idx + 1) / total_pages))
                    progress_callback(progress, f"Processed page {page_num} of {total_pages}...")

            except Exception as page_err:
                logger.warning(f"Error extracting page {page_idx + 1} in {valid_path.name}: {page_err}")

        doc.close()

        # Validate that document contained readable text
        combined_text = "\n\n".join(page_texts).strip()

        if not combined_text or total_extracted_chars < 20:
            raise PDFProcessingError(
                f"PDF '{valid_path.name}' contains no readable text. "
                "It may contain scanned images without OCR or empty pages."
            )

        try:
            validate_text_content(combined_text, min_words=5)
        except ValidationError as ve:
            raise PDFProcessingError(str(ve)) from ve

        if progress_callback:
            progress_callback(0.85, "Structuring page segments and token-sized chunks...")

        # Build transformer chunks with page tracking
        chunks = build_chunks_from_segments(segments, combined_text)

        if progress_callback:
            progress_callback(1.0, "PDF extraction complete.")

        lecture_content = LectureContent(
            title=doc_title,
            source_type="pdf",
            source_filename=valid_path.name,
            full_text=combined_text,
            segments=segments,
            chunks=chunks,
            metadata={
                "total_pages": total_pages,
                "pages_with_text": len(segments),
                "file_size_bytes": valid_path.stat().st_size,
            }
        )

        logger.info(
            f"Successfully processed PDF '{valid_path.name}': "
            f"{len(segments)}/{total_pages} pages with text, {len(chunks)} chunks, {lecture_content.total_words} words."
        )
        return lecture_content
