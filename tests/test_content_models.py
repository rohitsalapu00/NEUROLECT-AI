"""
NEUROLECT AI - Test Content Models
Validates ContentSegment, ContentChunk, LectureContent, serialization, and chunk building.
"""

import pytest
from modules.content_models import (
    ContentSegment,
    ContentChunk,
    LectureContent,
    build_chunks_from_segments,
)


def test_content_segment_source_label():
    """Verify source labels are formatted correctly per modality."""
    # PDF segment
    pdf_seg = ContentSegment(segment_id=0, text="Intro to Machine Learning", source_type="pdf", page_number=3)
    assert pdf_seg.source_label == "Page 3"

    # Audio/Video segment
    audio_seg = ContentSegment(
        segment_id=1,
        text="Supervised learning uses labeled datasets.",
        source_type="audio",
        start_time=12.5,
        end_time=25.0
    )
    assert audio_seg.source_label == "[00:12 - 00:25]"

    # Text segment
    text_seg = ContentSegment(segment_id=2, text="Unsupervised clustering groups data.", source_type="text")
    assert text_seg.source_label == "Section 3"


def test_lecture_content_serialization():
    """Verify LectureContent serialization to dict and reconstruction from dict."""
    seg1 = ContentSegment(segment_id=0, text="Deep learning utilizes multi-layer neural networks.", source_type="pdf", page_number=1)
    seg2 = ContentSegment(segment_id=1, text="Backpropagation computes gradients via the chain rule.", source_type="pdf", page_number=2)
    
    full_text = f"{seg1.text}\n\n{seg2.text}"
    chunks = build_chunks_from_segments([seg1, seg2], full_text, max_chunk_size=300)

    content = LectureContent(
        title="Neural Networks 101",
        source_type="pdf",
        source_filename="lecture1.pdf",
        full_text=full_text,
        segments=[seg1, seg2],
        chunks=chunks,
        metadata={"total_pages": 2}
    )

    assert content.total_words > 10
    assert content.total_segments == 2
    assert content.total_chunks >= 1

    # Test serialization
    data = content.to_dict()
    assert data["title"] == "Neural Networks 101"
    assert data["source_type"] == "pdf"
    assert len(data["segments"]) == 2

    # Test deserialization
    reconstructed = LectureContent.from_dict(data)
    assert reconstructed.title == content.title
    assert reconstructed.full_text == content.full_text
    assert reconstructed.total_segments == 2
    assert reconstructed.segments[0].page_number == 1
    assert reconstructed.segments[1].page_number == 2


def test_build_chunks_from_segments():
    """Verify chunk generation splits long text into transformer-sized blocks with overlap."""
    sentences = [f"Educational concept number {i} explains fundamental transformer mechanics." for i in range(30)]
    full_text = " ".join(sentences)
    segments = [
        ContentSegment(segment_id=i, text=s, source_type="text")
        for i, s in enumerate(sentences)
    ]

    chunks = build_chunks_from_segments(segments, full_text, max_chunk_size=400, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_count <= 500  # Within boundary
        assert len(chunk.text) > 0
        assert chunk.chunk_id >= 0
