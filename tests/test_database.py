"""
NEUROLECT AI - Test Database Manager & Persistence Layer
Validates SQLite schema creation, lecture storage/retrieval, question persistence,
quiz attempts, answer evaluations, real-time analytics aggregation, and source provenance.
"""

import pytest
import tempfile
from pathlib import Path
from database.db_manager import DatabaseManager
from modules.content_models import LectureContent, ContentChunk
from modules.question_models import GeneratedQuestion
from modules.quiz_models import Quiz, QuizQuestion, QuestionResult, QuizResult


@pytest.fixture
def temp_db():
    """Provides a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = DatabaseManager(db_path=db_path)
    yield db
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_init_and_empty_analytics(temp_db):
    """Verify empty database returns clean non-fabricated statistics."""
    summary = temp_db.get_analytics_summary()
    assert summary["total_lectures"] == 0
    assert summary["total_questions"] == 0
    assert summary["total_quizzes_attempted"] == 0
    assert summary["average_score_percentage"] is None
    assert summary["recent_lectures"] == []
    assert summary["recent_attempts"] == []


def test_save_and_retrieve_lecture(temp_db):
    """Verify lecture persistence and listing."""
    lecture = LectureContent(
        title="Intro to Machine Learning",
        source_type="pdf",
        source_filename="lecture1.pdf",
        full_text="Machine learning algorithms build a mathematical model based on training data.",
        metadata={"lecture_id": "lec_test_1", "pages": 5}
    )
    lecture.chunks = [
        ContentChunk(chunk_id=0, text=lecture.full_text, char_count=len(lecture.full_text), word_count=10, source_label="Page 1")
    ]
    saved_id = temp_db.save_lecture(lecture)
    assert saved_id == "lec_test_1"

    retrieved = temp_db.get_lecture("lec_test_1")
    assert retrieved is not None
    assert retrieved.title == "Intro to Machine Learning"
    assert retrieved.source_type == "pdf"
    assert "training data" in retrieved.full_text

    lectures_list = temp_db.list_lectures()
    assert len(lectures_list) == 1
    assert lectures_list[0]["lecture_id"] == "lec_test_1"


def test_save_and_retrieve_questions(temp_db):
    """Verify question persistence preserving provenance."""
    # First save the lecture
    lecture = LectureContent(
        title="Intro to Machine Learning",
        source_type="pdf",
        source_filename="lecture1.pdf",
        full_text="Machine learning algorithms build a mathematical model based on training data.",
        metadata={"lecture_id": "lec_test_1"}
    )
    temp_db.save_lecture(lecture)

    questions = [
        GeneratedQuestion(
            question_id="q_db_1",
            question_text="What do classification models predict?",
            question_type="MCQ",
            options=["Discrete category labels", "Continuous real values", "Noise", "Weights"],
            correct_answer="Discrete category labels",
            explanation="Predicts discrete labels.",
            topic="Supervised Learning",
            subtopic="Classification",
            bloom_level="Remember",
            difficulty="Easy",
            source_reference="Page 1, Paragraph 2",
            generation_metadata={"segment_id": 1, "lecture_id": "lec_test_1"}
        )
    ]
    saved_count = temp_db.save_questions(questions, lecture_id="lec_test_1")
    assert saved_count == 1

    by_lec = temp_db.get_questions_by_lecture("lec_test_1")
    assert len(by_lec) == 1
    assert by_lec[0].question_id == "q_db_1"
    assert by_lec[0].source_reference == "Page 1, Paragraph 2"
    assert by_lec[0].generation_metadata.get("segment_id") == 1

    all_q = temp_db.list_all_questions()
    assert len(all_q) == 1


def test_save_and_retrieve_quiz_attempt(temp_db):
    """Verify storing and retrieving full quiz attempt with breakdown."""
    qr = QuestionResult(
        question_id="q_db_1",
        question_text="What do classification models predict?",
        question_type="MCQ",
        student_answer="Discrete category labels",
        reference_answer="Discrete category labels",
        marks_awarded=1.0,
        max_marks=1.0,
        evaluation_status="correct",
        feedback="Correct!",
        missing_concepts=[],
        score_breakdown={"exact_match": 1.0},
        source_reference="Page 1",
        topic="Supervised Learning",
        difficulty="Easy",
        bloom_level="Remember",
        metadata={"segment_id": 1}
    )

    result = QuizResult(
        quiz_id="quiz_test_1",
        quiz_title="Machine Learning Assessment",
        total_questions=1,
        answered_questions=1,
        unanswered_questions=0,
        total_marks=1.0,
        earned_marks=1.0,
        percentage=100.0,
        correct_count=1,
        partial_count=0,
        incorrect_count=0,
        topic_scores={"Supervised Learning": {"total_marks": 1.0, "earned_marks": 1.0, "percentage": 100.0, "question_count": 1, "unanswered_count": 0}},
        difficulty_scores={"Easy": {"total_marks": 1.0, "earned_marks": 1.0, "percentage": 100.0, "question_count": 1, "unanswered_count": 0}},
        bloom_scores={"Remember": {"total_marks": 1.0, "earned_marks": 1.0, "percentage": 100.0, "question_count": 1, "unanswered_count": 0}},
        question_results=[qr],
        is_complete=True
    )

    attempt_id = temp_db.save_quiz_attempt(result, lecture_id="lec_test_1")
    assert attempt_id.startswith("attempt_")

    attempts = temp_db.list_quiz_attempts()
    assert len(attempts) == 1
    assert attempts[0]["attempt_id"] == attempt_id
    assert attempts[0]["percentage"] == 100.0

    attempt_details = temp_db.get_quiz_attempt(attempt_id)
    assert attempt_details is not None
    assert attempt_details["quiz_title"] == "Machine Learning Assessment"
    assert len(attempt_details["answers"]) == 1
    assert attempt_details["answers"][0]["question_id"] == "q_db_1"
    assert attempt_details["answers"][0]["segment_id"] == 1

    # Check analytics update
    summary = temp_db.get_analytics_summary()
    assert summary["total_quizzes_attempted"] == 1
    assert summary["average_score_percentage"] == 100.0
    assert len(summary["recent_attempts"]) == 1
