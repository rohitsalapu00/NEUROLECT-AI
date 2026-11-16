"""
NEUROLECT AI - Helper Utilities
Handles text cleaning, tokenization-aware chunking, formatting, and logging.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NEUROLECT")


def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted or transcribed text.
    Removes extraneous whitespace, non-printable characters, and formatting artifacts.
    """
    if not text:
        return ""
    
    # Replace non-breaking spaces and other Unicode spaces
    text = re.sub(r"[\xa0\u200b\u200e\u200f]", " ", text)
    # Normalize carriage returns and line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Condense 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Condense multiple horizontal spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespaces per line
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex boundary detection.
    Robust against abbreviations and decimals.
    Splits oversized sentences (>800 chars) on semi-colons or clause boundaries.
    """
    if not text:
        return []
    # Match sentence endings with punctuation followed by space or newline
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    raw_sentences = re.split(sentence_pattern, text)
    
    refined_sentences = []
    for s in raw_sentences:
        clean_s = s.strip()
        if not clean_s:
            continue
        # If a single sentence exceeds 800 characters without ending punctuation, split on clauses
        if len(clean_s) > 800:
            sub_clauses = re.split(r'(?<=;)\s+|(?<=:\s)|\n+', clean_s)
            for sub in sub_clauses:
                sub_clean = sub.strip()
                if len(sub_clean) > 800:
                    # Break long run-on text into word blocks of ~600 chars
                    words = sub_clean.split()
                    curr = []
                    curr_len = 0
                    for w in words:
                        if curr_len + len(w) + 1 > 600 and curr:
                            refined_sentences.append(" ".join(curr))
                            curr = [w]
                            curr_len = len(w)
                        else:
                            curr.append(w)
                            curr_len += len(w) + 1
                    if curr:
                        refined_sentences.append(" ".join(curr))
                elif sub_clean:
                    refined_sentences.append(sub_clean)
        else:
            refined_sentences.append(clean_s)

    return refined_sentences


def chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits long text into manageable chunks respecting sentence boundaries.
    Enforces maximum chunk size to avoid exceeding transformer token limits.
    Returns a list of dicts with chunk index, chunk text, and character offsets.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return [{"chunk_id": 0, "text": cleaned, "start_char": 0, "end_char": len(cleaned)}]

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > max_chunk_size and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "char_length": len(chunk_str),
                "sentence_count": len(current_sentences)
            })
            chunk_id += 1

            # Keep overlap sentences from the end
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            
            current_sentences = overlap_sentences
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_len

    if current_sentences:
        chunk_str = " ".join(current_sentences)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_str,
            "char_length": len(chunk_str),
            "sentence_count": len(current_sentences)
        })

    return chunks


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
