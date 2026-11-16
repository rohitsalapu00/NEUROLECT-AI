"""
NEUROLECT AI - Plain Text & Document Processing Module
Processes uploaded TXT documents and directly entered/pasted educational content
into the unified LectureContent format.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union, List
from config import SUPPORTED_DOC_FORMATS, MAX_UPLOAD_SIZE_MB
from utils.validators import validate_file_extension, validate_file_size, validate_text_content, ValidationError
from utils.helpers import clean_text
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments

logger = logging.getLogger("NEUROLECT.TextProcessor")


class TextProcessingError(Exception):
    """Raised when text file loading, decoding, or processing fails."""
    pass


class TextProcessor:
    """
    Transforms TXT files and pasted text strings into standardized LectureContent.
    Performs paragraph segmentation, encoding detection, and transformer-aware chunking.
    """

    def process_raw_text(
        self,
        raw_text: str,
        title: str = "Pasted Lecture Notes",
        source_name: str = "manual_input.txt",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Processes a raw text string (e.g. from Streamlit text area) into LectureContent.
        """
        if progress_callback:
            progress_callback(0.10, "Cleaning and normalizing text content...")

        cleaned = clean_text(raw_text)

        try:
            validate_text_content(cleaned, min_words=5)
        except ValidationError as e:
            raise TextProcessingError(str(e)) from e

        if progress_callback:
            progress_callback(0.40, "Segmenting text into logical paragraphs...")

        # Segment by double newlines or single newline paragraphs
        raw_paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]

        segments: List[ContentSegment] = []
        for idx, para in enumerate(raw_paragraphs):
            segments.append(ContentSegment(
                segment_id=idx,
                text=para,
                source_type="text",
                metadata={
                    "paragraph_index": idx + 1,
                    "word_count": len(para.split()),
                    "char_count": len(para)
                }
            ))

        if progress_callback:
            progress_callback(0.75, "Generating transformer-sized content chunks...")

        chunks = build_chunks_from_segments(segments, cleaned)

        if progress_callback:
            progress_callback(1.0, "Text processing complete.")

        lecture_content = LectureContent(
            title=title.strip() or "Pasted Lecture Notes",
            source_type="text",
            source_filename=source_name,
            full_text=cleaned,
            segments=segments,
            chunks=chunks,
            metadata={
                "total_paragraphs": len(segments),
                "input_method": "manual_entry" if source_name == "manual_input.txt" else "file_upload"
            }
        )

        logger.info(
            f"Successfully processed text '{title}': "
            f"{len(segments)} paragraphs, {len(chunks)} chunks, {lecture_content.total_words} words."
        )
        return lecture_content

    def process_txt_file(
        self,
        file_path: Union[str, Path],
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Reads and decodes a TXT file from disk and transforms it into LectureContent.
        Handles UTF-8 and latin-1 fallback decodings.
        """
        path = Path(file_path)
        if not path.exists():
            raise TextProcessingError(f"Text file not found: {path}")

        try:
            validate_file_extension(path.name, [".txt"])
            validate_file_size(path, MAX_UPLOAD_SIZE_MB)
        except ValidationError as e:
            raise TextProcessingError(str(e)) from e

        doc_title = title or path.stem.replace("_", " ").title()

        if progress_callback:
            progress_callback(0.10, f"Reading text file '{path.name}'...")

        raw_content = ""
        # Try UTF-8 first, fallback to latin-1
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    raw_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not raw_content:
            raise TextProcessingError(f"Could not decode text file '{path.name}' or file is empty.")

        return self.process_raw_text(
            raw_text=raw_content,
            title=doc_title,
            source_name=path.name,
            progress_callback=progress_callback
        )
