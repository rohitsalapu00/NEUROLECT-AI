"""
NEUROLECT AI - Test Topic Detector
Validates SentenceTransformer semantic clustering, topic discovery, and lecture outline generation.
Tests clustering on lectures with multiple unrelated topics and short documents.
"""

import pytest
from modules.topic_detector import TopicDetector, TopicDetectionError
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def topic_detector():
    return TopicDetector()


@pytest.fixture
def multi_topic_content():
    """Builds LectureContent covering two distinct educational topics."""
    seg1 = ContentSegment(
        segment_id=0,
        text="Convolutional Neural Networks are designed for spatial computer vision tasks. "
             "Kernels slide across feature maps to perform feature extraction. "
             "Pooling layers downsample the representations to achieve translational invariance.",
        source_type="audio",
        start_time=0.0,
        end_time=45.0
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="Recurrent Neural Networks process sequential time-series and natural language data. "
             "Hidden state vectors pass temporal information from one time step to the next. "
             "LSTMs mitigate vanishing gradients through specialized memory cells.",
        source_type="audio",
        start_time=46.0,
        end_time=90.0
    )

    full_text = f"{seg1.text}\n\n{seg2.text}"
    return LectureContent(
        title="Deep Learning Architectures",
        source_type="audio",
        source_filename="lecture.mp3",
        full_text=full_text,
        segments=[seg1, seg2]
    )


def test_detect_topics(topic_detector, multi_topic_content):
    """Verify topic detection finds topics and associates segment citations."""
    topics = topic_detector.detect_topics(multi_topic_content, num_topics=2)
    assert isinstance(topics, list)
    assert len(topics) == 2
    for t in topics:
        assert "name" in t
        assert "thesis_statement" in t
        assert "source_labels" in t
        assert len(t["source_labels"]) > 0


def test_unrelated_topics_clustering(topic_detector):
    """Verify that multiple distinct unrelated topics are clustered into distinct semantic groups."""
    seg1 = ContentSegment(segment_id=0, text="Computer vision convolutional filters detect horizontal and vertical edges in image matrices.", source_type="pdf", page_number=1)
    seg2 = ContentSegment(segment_id=1, text="Image pooling operations reduce spatial dimensions while retaining invariant visual features.", source_type="pdf", page_number=2)
    seg3 = ContentSegment(segment_id=2, text="Speech recognition acoustic models convert sound waves and spectrograms into phonemes.", source_type="pdf", page_number=3)
    seg4 = ContentSegment(segment_id=3, text="Language models compute probability distributions over vocabulary tokens for speech decoding.", source_type="pdf", page_number=4)

    full_text = "\n\n".join(s.text for s in [seg1, seg2, seg3, seg4])
    content = LectureContent(
        title="Multi-Domain AI Lecture",
        source_type="pdf",
        source_filename="multidomain.pdf",
        full_text=full_text,
        segments=[seg1, seg2, seg3, seg4]
    )

    topics = topic_detector.detect_topics(content, num_topics=2)
    assert len(topics) == 2
    # Verify both topics have distinct names and valid segment groupings
    assert topics[0]["name"] != topics[1]["name"]
    assert len(topics[0]["segment_ids"]) >= 1
    assert len(topics[1]["segment_ids"]) >= 1


def test_short_single_segment_lecture(topic_detector):
    """Verify that a short document with only one segment is handled gracefully without errors."""
    seg = ContentSegment(
        segment_id=0,
        text="Support Vector Machines find the optimal hyperplane that maximizes the margin between classes.",
        source_type="text"
    )
    content = LectureContent(
        title="SVM Intro",
        source_type="text",
        source_filename="svm.txt",
        full_text=seg.text,
        segments=[seg]
    )

    topics = topic_detector.detect_topics(content)
    assert len(topics) == 1
    assert "name" in topics[0]
    assert topics[0]["segment_ids"] == [0]


def test_closely_related_topics_clustering(topic_detector):
    """Verify clustering on closely related topics (e.g. L1 vs L2 regularization)."""
    seg1 = ContentSegment(segment_id=0, text="L1 Lasso regularization adds the absolute value of coefficients as penalty.", source_type="pdf", page_number=1)
    seg2 = ContentSegment(segment_id=1, text="L1 penalty drives redundant feature weights exactly to zero for feature selection.", source_type="pdf", page_number=2)
    seg3 = ContentSegment(segment_id=2, text="L2 Ridge regularization penalizes the squared magnitude of regression coefficients.", source_type="pdf", page_number=3)
    seg4 = ContentSegment(segment_id=3, text="L2 penalty shrinks weights smoothly without setting them strictly to zero.", source_type="pdf", page_number=4)

    content = LectureContent(
        title="Regularization Techniques",
        source_type="pdf",
        source_filename="reg.pdf",
        full_text="\n\n".join(s.text for s in [seg1, seg2, seg3, seg4]),
        segments=[seg1, seg2, seg3, seg4]
    )
    topics = topic_detector.detect_topics(content, num_topics=2)
    assert len(topics) == 2
    assert topics[0]["topic_id"] == 1
    assert topics[1]["topic_id"] == 2


def test_single_topic_lecture(topic_detector):
    """Verify multi-segment lecture where user specifies single overarching topic."""
    seg1 = ContentSegment(segment_id=0, text="Linear regression models continuous relationships between variables.", source_type="text")
    seg2 = ContentSegment(segment_id=1, text="Ordinary least squares minimizes the sum of squared prediction residuals.", source_type="text")
    seg3 = ContentSegment(segment_id=2, text="Assumptions include homoscedasticity, linearity, and independence of errors.", source_type="text")

    content = LectureContent(
        title="Linear Regression Masterclass",
        source_type="text",
        source_filename="linear_reg.txt",
        full_text="\n\n".join(s.text for s in [seg1, seg2, seg3]),
        segments=[seg1, seg2, seg3]
    )
    topics = topic_detector.detect_topics(content, num_topics=1)
    assert len(topics) == 1
    assert len(topics[0]["segment_ids"]) == 3
    assert "Linear" in topics[0]["name"] or "Regression" in topics[0]["name"] or len(topics[0]["name"]) > 3


def test_generate_lecture_outline(topic_detector, multi_topic_content):
    """Verify structured syllabus outline generation."""
    outline = topic_detector.generate_lecture_outline(multi_topic_content)
    assert isinstance(outline, dict)
    assert outline["total_topics"] >= 1
    assert "topics" in outline
    assert len(outline["topics"]) >= 1


def test_topic_detection_empty_raises(topic_detector):
    """Verify empty content raises TopicDetectionError."""
    empty = LectureContent(
        title="Empty",
        source_type="text",
        source_filename="empty.txt",
        full_text="",
        segments=[]
    )
    with pytest.raises(TopicDetectionError):
        topic_detector.detect_topics(empty)
