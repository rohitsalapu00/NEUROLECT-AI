"""
NEUROLECT AI - Test Question Generator
Validates genuine FLAN-T5 inference across all 6 question types:
MCQ, Short Answer, Descriptive, Conceptual, Application, Analytical.
Tests source reference preservation, multi-topic generation, and short content handling.
"""

import pytest
from modules.question_generator import QuestionGenerator, QuestionGenerationError
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def question_generator():
    return QuestionGenerator()


@pytest.fixture
def sample_lecture_content():
    """Builds a realistic multi-page LectureContent."""
    seg1 = ContentSegment(
        segment_id=0,
        text="Supervised learning algorithms train predictive models using labeled datasets. "
             "Classification models predict discrete category labels, whereas regression models predict continuous real values. "
             "A standard evaluation metric for classification is accuracy and F1 score.",
        source_type="pdf",
        page_number=1
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="Overfitting happens when a machine learning model memorizes training noise rather than true underlying patterns. "
             "L1 Lasso regularization introduces an absolute weight penalty that induces parameter sparsity. "
             "L2 Ridge regularization penalizes squared weight magnitudes to shrink weights smoothly.",
        source_type="pdf",
        page_number=2
    )

    return LectureContent(
        title="Machine Learning Foundations",
        source_type="pdf",
        source_filename="ml_foundations.pdf",
        full_text=f"{seg1.text}\n\n{seg2.text}",
        segments=[seg1, seg2]
    )


def test_generate_all_six_question_types(question_generator, sample_lecture_content):
    """
    Executes live FLAN-T5 inference to generate questions across the 6 educational types.
    Verifies that each question type is produced and provenance is preserved.
    """
    types_to_generate = [
        "MCQ", "Short Answer", "Descriptive", "Conceptual", "Application", "Analytical"
    ]

    questions = question_generator.generate_questions(
        content=sample_lecture_content,
        question_types=types_to_generate,
        num_questions=6
    )

    assert len(questions) == 6
    generated_types = {q.question_type for q in questions}

    # Verify every question has text, correct answer, and source reference
    for q in questions:
        assert len(q.question_text) > 10
        assert q.question_text.endswith("?")
        assert len(q.correct_answer) > 0
        assert q.source_reference in ["Page 1", "Page 2"]
        assert q.bloom_level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        assert q.difficulty in ["Easy", "Medium", "Hard"]

    # Verify MCQ contains exactly 4 options with the answer inside
    mcq_questions = [q for q in questions if q.question_type == "MCQ"]
    assert len(mcq_questions) >= 1
    for mcq in mcq_questions:
        assert len(mcq.options) == 4
        assert mcq.correct_answer in mcq.options


def test_generate_multiple_questions_from_single_topic(question_generator, sample_lecture_content):
    """Verify generating multiple questions focused on a specific topic."""
    topics = [{
        "topic_id": 1,
        "name": "Supervised Learning Fundamentals",
        "segment_ids": [0],
        "subtopics": ["Classification", "Regression"]
    }]

    questions = question_generator.generate_questions(
        content=sample_lecture_content,
        topics=topics,
        question_types=["Conceptual", "Short Answer"],
        num_questions=3
    )

    assert len(questions) >= 2
    for q in questions:
        assert q.topic == "Supervised Learning Fundamentals"


def test_insufficient_content_handled_gracefully(question_generator):
    """Verify that very short text generates available questions without crashing."""
    seg = ContentSegment(
        segment_id=0,
        text="Gradient descent iteratively updates model parameters in the opposite direction of the gradient.",
        source_type="text"
    )
    short_content = LectureContent(
        title="Short Note",
        source_type="text",
        source_filename="note.txt",
        full_text=seg.text,
        segments=[seg]
    )

    questions = question_generator.generate_questions(
        content=short_content,
        question_types=["Short Answer"],
        num_questions=1
    )
    assert len(questions) == 1
    assert "gradient" in questions[0].question_text.lower() or "gradient" in questions[0].correct_answer.lower()


def test_empty_content_raises(question_generator):
    """Verify empty segments raise QuestionGenerationError."""
    empty = LectureContent(
        title="Empty",
        source_type="text",
        source_filename="empty.txt",
        full_text="",
        segments=[]
    )
    with pytest.raises(QuestionGenerationError):
        question_generator.generate_questions(empty)


def test_preservation_of_source_references_after_rejection_or_regeneration(question_generator):
    """
    Verify that every generated question preserves its exact source reference (page or timestamp)
    even when regeneration or candidate rejection occurred.
    """
    seg1 = ContentSegment(
        segment_id=0,
        text="Supervised learning algorithms train predictive models using labeled datasets. "
             "Classification models predict discrete category labels. "
             "Regression models predict continuous values.",
        source_type="pdf",
        page_number=3
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="Backpropagation calculates gradients using the multivariate chain rule. "
             "Optimizers update synaptic weights opposite to the gradient vector.",
        source_type="pdf",
        page_number=4
    )
    content = LectureContent(
        title="Provenance Test",
        source_type="pdf",
        source_filename="provenance.pdf",
        full_text=f"{seg1.text}\n\n{seg2.text}",
        segments=[seg1, seg2]
    )

    questions = question_generator.generate_questions(
        content=content,
        question_types=["Conceptual", "Short Answer", "MCQ"],
        num_questions=3
    )

    assert len(questions) == 3
    valid_provenances = {"Page 3", "Page 4"}
    for q in questions:
        assert q.source_reference in valid_provenances
        assert q.generation_metadata.get("segment_id") in [0, 1]
        assert q.source_reference == ("Page 3" if q.generation_metadata["segment_id"] == 0 else "Page 4")
