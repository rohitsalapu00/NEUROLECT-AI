"""
NEUROLECT AI - Test Question Validator
Validates structural integrity, MCQ option checks, duplicate question detection,
and source grounding verification.
"""

import pytest
from modules.question_validator import QuestionValidator
from modules.question_models import GeneratedQuestion
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def validator():
    return QuestionValidator()


@pytest.fixture
def sample_content():
    text = "Convolutional neural networks use kernels to extract visual features from image matrices."
    seg = ContentSegment(segment_id=0, text=text, source_type="text")
    return LectureContent(
        title="CNN Intro",
        source_type="text",
        source_filename="cnn.txt",
        full_text=text,
        segments=[seg]
    )


def test_validate_valid_mcq(validator, sample_content):
    """Verify that a correctly formatted MCQ passes validation."""
    q = GeneratedQuestion(
        question_id="q1",
        question_text="What do convolutional neural networks use to extract visual features?",
        question_type="MCQ",
        options=["Kernels", "Recurrent cells", "Attention heads", "Markov chains"],
        correct_answer="Kernels",
        explanation="Kernels slide over input matrices.",
        topic="Computer Vision",
        subtopic="CNNs",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Section 1"
    )

    res = validator.validate_question(q, source_content=sample_content)
    assert res.is_valid is True
    assert len(res.issues) == 0
    assert res.quality_score >= 80.0


def test_validate_mcq_missing_correct_answer_in_options(validator):
    """Verify that an MCQ where the answer is not in options fails validation."""
    q = GeneratedQuestion(
        question_id="q2",
        question_text="What algorithm minimizes loss functions?",
        question_type="MCQ",
        options=["Option A", "Option B", "Option C", "Option D"],
        correct_answer="Gradient Descent",  # Not in options!
        explanation="",
        topic="ML",
        subtopic="Optimization",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Page 1"
    )

    res = validator.validate_question(q)
    assert res.is_valid is False
    assert any("not present in the MCQ options" in issue for issue in res.issues)


def test_validate_mcq_duplicate_options(validator):
    """Verify that an MCQ with duplicate options fails validation."""
    q = GeneratedQuestion(
        question_id="q3",
        question_text="What is a neural gate?",
        question_type="MCQ",
        options=["Forget gate", "Forget gate", "Input gate", "Output gate"],
        correct_answer="Forget gate",
        explanation="",
        topic="ML",
        subtopic="LSTMs",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Page 1"
    )

    res = validator.validate_question(q)
    assert res.is_valid is False
    assert any("duplicate answer options" in issue for issue in res.issues)


def test_validate_empty_fields(validator):
    """Verify empty question text or empty answer fails validation."""
    q = GeneratedQuestion(
        question_id="q4",
        question_text="",
        question_type="Short Answer",
        options=[],
        correct_answer="",
        explanation="",
        topic="General",
        subtopic="General",
        bloom_level="Understand",
        difficulty="Easy",
        source_reference="General"
    )
    res = validator.validate_question(q)
    assert res.is_valid is False
    assert any("empty" in issue.lower() for issue in res.issues)


def test_duplicate_question_detection_in_suite(validator):
    """Verify detection of identical or near-duplicate questions in a suite."""
    q1 = GeneratedQuestion(
        question_id="qa",
        question_text="What is the role of backpropagation in training neural networks?",
        question_type="Short Answer",
        options=[],
        correct_answer="Computing gradients",
        explanation="",
        topic="ML",
        subtopic="Backprop",
        bloom_level="Understand",
        difficulty="Medium",
        source_reference="Page 1"
    )
    # Near identical duplicate
    q2 = GeneratedQuestion(
        question_id="qb",
        question_text="What is the role of backpropagation in training neural networks?",
        question_type="Short Answer",
        options=[],
        correct_answer="Computing gradients",
        explanation="",
        topic="ML",
        subtopic="Backprop",
        bloom_level="Understand",
        difficulty="Medium",
        source_reference="Page 1"
    )

    suite_eval = validator.validate_question_suite([q1, q2])
    assert suite_eval["duplicate_count"] == 1
    assert len(suite_eval["duplicate_pairs"]) == 1


def test_validate_answer_lexically_grounded_but_does_not_answer_question(validator):
    """Verify that an answer whose words are in the text, but belongs to an unrelated sentence, fails validation."""
    text = "Photosynthesis converts sunlight into chemical energy. Mitochondria generate ATP for cellular processes."
    content = LectureContent(
        title="Biology",
        source_type="text",
        source_filename="bio.txt",
        full_text=text,
        segments=[ContentSegment(segment_id=0, text=text, source_type="text")]
    )

    # Question asks about mitochondria / ATP, but answer is from the photosynthesis sentence
    q = GeneratedQuestion(
        question_id="q_unrelated_ans",
        question_text="What organelle generates ATP for cellular processes?",
        question_type="Short Answer",
        options=[],
        correct_answer="Converts sunlight into chemical energy",
        explanation="",
        topic="Biology",
        subtopic="Cell",
        bloom_level="Understand",
        difficulty="Medium",
        source_reference="Section 1"
    )

    res = validator.validate_question(q, source_content=content)
    assert res.is_valid is False
    assert any("does not address the question" in issue for issue in res.issues)


def test_validate_answer_not_supported_by_source(validator):
    """Verify that an answer containing hallucinated concepts not in the source text fails validation."""
    text = "Gradient descent iteratively updates model parameters to minimize the empirical risk."
    content = LectureContent(
        title="Optimization",
        source_type="text",
        source_filename="opt.txt",
        full_text=text,
        segments=[ContentSegment(segment_id=0, text=text, source_type="text")]
    )

    # Completely hallucinated answer
    q = GeneratedQuestion(
        question_id="q_hallucinated",
        question_text="What is the objective of gradient descent in machine learning?",
        question_type="Short Answer",
        options=[],
        correct_answer="Quantum entanglement between superposition states",
        explanation="",
        topic="ML",
        subtopic="Optimization",
        bloom_level="Understand",
        difficulty="Medium",
        source_reference="Page 1"
    )

    res = validator.validate_question(q, source_content=content)
    assert res.is_valid is False
    assert any("not supported by the source text" in issue for issue in res.issues)


def test_validate_mcq_two_technically_correct_options_fails(validator):
    """Verify that an MCQ with two technically correct answers in the text fails validation."""
    text = "A standard evaluation metric for classification is accuracy and F1 score."
    content = LectureContent(
        title="ML Evaluation",
        source_type="text",
        source_filename="eval.txt",
        full_text=text,
        segments=[ContentSegment(segment_id=0, text=text, source_type="text")]
    )

    q = GeneratedQuestion(
        question_id="q_ambig_mcq",
        question_text="What is a standard evaluation metric for classification?",
        question_type="MCQ",
        options=["Accuracy", "F1 score", "Continuous values", "Training noise"],
        correct_answer="Accuracy",
        explanation="",
        topic="ML",
        subtopic="Evaluation",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Page 1"
    )

    res = validator.validate_question(q, source_content=content)
    assert res.is_valid is False
    assert any("multiple defensible answers" in issue.lower() for issue in res.issues)


def test_validate_mcq_one_defensible_answer_passes(validator):
    """Verify that an MCQ with exactly one correct option and unambiguously incorrect distractors passes validation."""
    text = (
        "Classification models predict discrete category labels, whereas regression models predict continuous real values. "
        "Overfitting happens when a model memorizes noise."
    )
    content = LectureContent(
        title="ML Models",
        source_type="text",
        source_filename="models.txt",
        full_text=text,
        segments=[ContentSegment(segment_id=0, text=text, source_type="text")]
    )

    q = GeneratedQuestion(
        question_id="q_clean_mcq",
        question_text="What do classification models predict?",
        question_type="MCQ",
        options=["Discrete category labels", "Continuous real values", "Training noise", "Unlabeled cluster centroids"],
        correct_answer="Discrete category labels",
        explanation="Directly stated in Section 1.",
        topic="ML",
        subtopic="Classification",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Page 1"
    )

    res = validator.validate_question(q, source_content=content)
    assert res.is_valid is True
    assert len(res.issues) == 0
    assert res.quality_score >= 80.0
