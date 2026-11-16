"""
NEUROLECT AI - Input Validation Utilities
Validates file formats, sizes, and content sanity across all modalities.
"""

import os
from pathlib import Path
from typing import List, Union
from config import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_DOC_FORMATS,
    MAX_UPLOAD_SIZE_MB,
)


class ValidationError(Exception):
    """Custom exception raised when an input fails validation checks."""
    pass


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> str:
    """
    Validates that a filename has an allowed extension.
    Returns the lowercased extension if valid, otherwise raises ValidationError.
    """
    if not filename:
        raise ValidationError("No filename provided.")
    
    ext = Path(filename).suffix.lower()
    if not ext:
        raise ValidationError(f"File '{filename}' has no extension.")
    
    allowed = [e.lower() for e in allowed_extensions]
    if ext not in allowed:
        raise ValidationError(
            f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed)}"
        )
    return ext


def validate_file_size(file_path_or_bytes: Union[str, Path, bytes, int], max_mb: int = MAX_UPLOAD_SIZE_MB) -> int:
    """
    Validates that a file's size does not exceed the maximum allowed threshold.
    Returns the file size in bytes if valid, otherwise raises ValidationError.
    """
    max_bytes = max_mb * 1024 * 1024

    if isinstance(file_path_or_bytes, (str, Path)):
        p = Path(file_path_or_bytes)
        if not p.exists():
            raise ValidationError(f"File does not exist: {p}")
        size = p.stat().st_size
    elif isinstance(file_path_or_bytes, bytes):
        size = len(file_path_or_bytes)
    elif isinstance(file_path_or_bytes, int):
        size = file_path_or_bytes
    else:
        raise ValidationError(f"Invalid input type for file size check: {type(file_path_or_bytes)}")

    if size <= 0:
        raise ValidationError("File is empty (0 bytes).")
    
    if size > max_bytes:
        size_mb = size / (1024 * 1024)
        raise ValidationError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )

    return size


def validate_text_content(text: str, min_words: int = 5) -> str:
    """
    Validates that extracted or provided text is not empty and has sufficient length.
    Returns cleaned text if valid, otherwise raises ValidationError.
    """
    if not text or not text.strip():
        raise ValidationError("Text content is empty or contains only whitespace.")
    
    cleaned = text.strip()
    words = cleaned.split()
    if len(words) < min_words:
        raise ValidationError(
            f"Content too short for meaningful analysis ({len(words)} words found; minimum {min_words} words required)."
        )
    return cleaned
