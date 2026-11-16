"""
NEUROLECT AI - Unified Content Models
Defines the standard internal representation for all lecture inputs (Audio, Video, PDF, Text).
Completely RAG-free architecture: uses direct text segmentation, chunking, and metadata tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.helpers import clean_text, chunk_text, format_timestamp


@dataclass
class ContentSegment:
    """
    A granular segment of lecture content with explicit source provenance.
    - Audio/Video: start_time and end_time (in seconds)
    - PDF: page_number (1-indexed)
    - Text: paragraph or section sequence index
    """
    segment_id: int
    text: str
    source_type: str                                  # 'audio', 'video', 'pdf', 'text'
    page_number: Optional[int] = None                 # Preserved PDF page
    start_time: Optional[float] = None                # Audio/Video start timestamp in seconds
    end_time: Optional[float] = None                  # Audio/Video end timestamp in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Returns a user-facing readable source location tag."""
        if self.page_number is not None:
            return f"Page {self.page_number}"
        if self.start_time is not None and self.end_time is not None:
            return f"[{format_timestamp(self.start_time)} - {format_timestamp(self.end_time)}]"
        return f"Section {self.segment_id + 1}"


@dataclass
class ContentChunk:
    """
    A text chunk sized to strictly adhere to transformer token limits.
    Directly processed by Hugging Face models without vector databases or document retrieval.
    """
    chunk_id: int
    text: str
    char_count: int
    word_count: int
    source_segments: List[int] = field(default_factory=list)  # Segment IDs included in this chunk
    source_label: str = ""                                    # Primary source citation


@dataclass
class LectureContent:
    """
    Standardized internal content representation for NEUROLECT AI.
    Unifies Audio recordings, Video lectures, PDFs, and Plain Text into
    one consistent data structure consumed by all downstream NLP and AI modules.
    """
    title: str
    source_type: str                                  # 'audio', 'video', 'pdf', 'text'
    source_filename: str
    full_text: str                                    # Normalized complete text
    segments: List[ContentSegment] = field(default_factory=list)
    chunks: List[ContentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_words(self) -> int:
        return len(self.full_text.split())

    @property
    def total_characters(self) -> int:
        return len(self.full_text)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the LectureContent instance to a Python dictionary."""
        return {
            "title": self.title,
            "source_type": self.source_type,
            "source_filename": self.source_filename,
            "full_text": self.full_text,
            "segments": [asdict(seg) for seg in self.segments],
            "chunks": [asdict(chk) for chk in self.chunks],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LectureContent":
        """Reconstructs a LectureContent instance from a dictionary."""
        segments = [
            ContentSegment(**seg_data) for seg_data in data.get("segments", [])
        ]
        chunks = [
            ContentChunk(**chk_data) for chk_data in data.get("chunks", [])
        ]
        return cls(
            title=data.get("title", "Untitled Lecture"),
            source_type=data.get("source_type", "text"),
            source_filename=data.get("source_filename", "unknown"),
            full_text=data.get("full_text", ""),
            segments=segments,
            chunks=chunks,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


def build_chunks_from_segments(
    segments: List[ContentSegment],
    full_text: str,
    max_chunk_size: int = 1200,
    overlap: int = 150
) -> List[ContentChunk]:
    """
    Creates transformer-sized ContentChunks from segments while maintaining source citations.
    """
    raw_chunks = chunk_text(full_text, max_chunk_size=max_chunk_size, overlap=overlap)
    content_chunks = []

    for raw in raw_chunks:
        c_text = raw["text"]
        
        # Identify which segments overlap with this chunk's text
        matching_segment_ids = []
        matching_labels = []
        for seg in segments:
            # Check if segment text or substring is present in chunk
            if seg.text and (seg.text in c_text or c_text in seg.text or any(w in c_text for w in seg.text.split()[:5])):
                matching_segment_ids.append(seg.segment_id)
                if seg.source_label and seg.source_label not in matching_labels:
                    matching_labels.append(seg.source_label)

        if not matching_labels and segments:
            # Fallback to closest segment
            matching_labels = [segments[0].source_label]

        source_label = ", ".join(matching_labels[:3]) if matching_labels else "General"

        content_chunks.append(ContentChunk(
            chunk_id=raw["chunk_id"],
            text=c_text,
            char_count=len(c_text),
            word_count=len(c_text.split()),
            source_segments=matching_segment_ids,
            source_label=source_label
        ))

    return content_chunks
