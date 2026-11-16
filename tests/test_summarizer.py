"""
NEUROLECT AI - Test Summarizer
Validates transformer-based concise, detailed, topic-wise summaries and structured lecture notes.
Tests long-text completeness to verify that trailing chunks are not silently discarded.
"""

import pytest
from modules.summarizer import LectureSummarizer, SummarizationError
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments


@pytest.fixture
def sample_lecture_content():
    """Builds a realistic LectureContent with multi-segment educational text."""
    seg1 = ContentSegment(
        segment_id=0,
        text="Supervised machine learning algorithms train predictive models using labeled datasets. "
             "The algorithm optimizes an objective loss function through gradient descent. "
             "Common tasks include regression for continuous targets and classification for discrete targets.",
        source_type="pdf",
        page_number=1
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="Overfitting occurs when a machine learning model memorizes training noise instead of general patterns. "
             "Regularization techniques such as L1 Lasso and L2 Ridge penalize excessive parameter weights. "
             "Cross-validation provides an unbiased estimate of generalization performance on unseen test data.",
        source_type="pdf",
        page_number=2
    )

    full_text = f"{seg1.text}\n\n{seg2.text}"
    chunks = build_chunks_from_segments([seg1, seg2], full_text, max_chunk_size=300)

    return LectureContent(
        title="Supervised Learning & Regularization",
        source_type="pdf",
        source_filename="lecture_ml.pdf",
        full_text=full_text,
        segments=[seg1, seg2],
        chunks=chunks
    )


@pytest.fixture
def summarizer():
    return LectureSummarizer()


def test_generate_concise_summary(summarizer, sample_lecture_content):
    """Verify that genuine transformer model generates a concise summary."""
    summary = summarizer.generate_concise_summary(sample_lecture_content)
    assert isinstance(summary, str)
    assert len(summary) > 20
    assert len(summary.split()) >= 5


def test_generate_detailed_summary(summarizer, sample_lecture_content):
    """Verify detailed multi-chunk summary generation."""
    detailed = summarizer.generate_detailed_summary(sample_lecture_content)
    assert isinstance(detailed, str)
    assert len(detailed) > 30
    assert len(detailed.split()) >= 10


def test_generate_structured_notes(summarizer, sample_lecture_content):
    """Verify structured lecture notes with key takeaways and executive summary."""
    notes = summarizer.generate_structured_notes(sample_lecture_content)
    assert isinstance(notes, dict)
    assert "executive_summary" in notes
    assert "key_takeaways" in notes
    assert len(notes["key_takeaways"]) > 0


def test_long_text_completeness_no_tail_discarded(summarizer):
    """
    Verifies that for a multi-chunk document, the final portion (tail chunk)
    is processed and reflected in the detailed summary without being silently truncated.
    """
    segments = []
    # Create 4 distinct educational segments with substantive text
    segments.append(ContentSegment(
        segment_id=0,
        text="Part 1 covers data preprocessing and cleaning techniques such as normalization and missing value imputation in large datasets.",
        source_type="pdf",
        page_number=1
    ))
    segments.append(ContentSegment(
        segment_id=1,
        text="Part 2 explains deep learning feature extraction architectures using multiple layers of non-linear activation functions.",
        source_type="pdf",
        page_number=2
    ))
    segments.append(ContentSegment(
        segment_id=2,
        text="Part 3 discusses model evaluation metrics including precision, recall, F1-score, and area under the receiver operating characteristic curve.",
        source_type="pdf",
        page_number=3
    ))
    # Crucial tail segment at the very end
    tail_text = "Part 4 covers quantum computing qubits, quantum entanglement, and superposition principles for advanced machine intelligence."
    segments.append(ContentSegment(
        segment_id=3,
        text=tail_text,
        source_type="pdf",
        page_number=4
    ))

    full_text = "\n\n".join(s.text for s in segments)
    chunks = build_chunks_from_segments(segments, full_text, max_chunk_size=150)

    assert len(chunks) >= 3, f"Should produce >= 3 chunks to test combination, produced {len(chunks)}"

    long_lecture = LectureContent(
        title="Comprehensive Multi-Part Lecture",
        source_type="pdf",
        source_filename="multipart.pdf",
        full_text=full_text,
        segments=segments,
        chunks=chunks
    )

    # Detailed summary must include citation to the final page/part
    detailed_summary = summarizer.generate_detailed_summary(long_lecture)
    assert f"Part {len(chunks)}" in detailed_summary or "Page 4" in detailed_summary

    # Concise summary must process all chunks hierarchically without error
    concise_summary = summarizer.generate_concise_summary(long_lecture)
    assert len(concise_summary) > 20


def test_summarize_empty_content_raises(summarizer):
    """Verify that empty LectureContent raises SummarizationError."""
    empty_content = LectureContent(
        title="Empty",
        source_type="text",
        source_filename="empty.txt",
        full_text=""
    )
    with pytest.raises(SummarizationError):
        summarizer.generate_concise_summary(empty_content)
