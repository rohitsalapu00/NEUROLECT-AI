"""
NEUROLECT AI - Complete System Integration Test Suite
Validates the entire end-to-end pipeline:
Lecture Upload -> Content Processing -> Lecture Understanding ->
Question Generation -> Quiz Administration -> Automated Assessment ->
Database Persistence -> Learning Analytics.
"""

import pytest
import tempfile
import fitz  # PyMuPDF
from pathlib import Path
from unittest.mock import patch, MagicMock

from database.db_manager import DatabaseManager
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments
from modules.pdf_processor import PDFProcessor
from modules.text_processor import TextProcessor
from modules.audio_processor import AudioProcessor
from modules.video_processor import VideoProcessor
from modules.summarizer import LectureSummarizer
from modules.topic_detector import TopicDetector
from modules.keyword_extractor import KeywordExtractor
from modules.ner_processor import NERProcessor
from modules.concept_extractor import ConceptExtractor
from modules.question_models import GeneratedQuestion
from modules.question_generator import QuestionGenerator
from modules.quiz_engine import QuizEngine
from modules.answer_evaluator import AnswerEvaluator


@pytest.fixture
def temp_db():
    """Provides an isolated SQLite database for integration testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = DatabaseManager(db_path=db_path)
    yield db
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_complete_end_to_end_text_pipeline(temp_db):
    """
    Validates complete end-to-end execution:
    Text Ingestion -> Persistence -> Understanding -> Question Gen ->
    Quiz Session -> Evaluation -> Attempt Persistence -> Learning Analytics.
    """
    raw_lecture_text = (
        "Supervised machine learning algorithms train predictive models using labeled datasets. "
        "Classification models output discrete category labels such as spam or not spam. "
        "Regression models predict continuous real values such as house prices or temperature. "
        "Regularization techniques such as L1 Lasso and L2 Ridge penalize excessive parameter weights. "
        "L1 Lasso introduces an absolute penalty to induce parameter sparsity and perform feature selection. "
        "L2 Ridge regularization penalizes squared weight magnitudes to smoothly prevent overfitting. "
        "Backpropagation computes the gradient of the loss function with respect to model weights "
        "using the mathematical chain rule across network layers. Therefore, weights must be updated "
        "iteratively via gradient descent to minimize empirical risk."
    )

    # 1. Text Ingestion
    content = TextProcessor().process_raw_text(raw_lecture_text, title="Foundations of Machine Learning")
    assert content.total_words > 40
    assert content.total_segments >= 1

    # 2. Database Persistence
    lec_id = temp_db.save_lecture(content)
    assert lec_id is not None
    assert temp_db.get_lecture(lec_id) is not None

    # 3. Lecture Understanding Pipeline
    # a. Summarization
    summarizer = LectureSummarizer()
    concise = summarizer.generate_concise_summary(content)
    assert len(concise) > 10

    # b. Topic Detection
    detector = TopicDetector()
    topics = detector.detect_topics(content)
    assert len(topics) >= 1

    # c. Keyword & Concept Extraction
    kw_extractor = KeywordExtractor()
    keywords = kw_extractor.extract_keywords(content, top_n=5)
    assert len(keywords) >= 1

    concept_extractor = ConceptExtractor()
    principles = concept_extractor.extract_principles(content)
    assert isinstance(principles, list)

    # d. NER
    ner = NERProcessor()
    entities = ner.extract_entities(content)
    assert isinstance(entities, dict)
    assert "categories" in entities

    # 4. Question Generation
    q_gen = QuestionGenerator()
    questions = q_gen.generate_questions(
        content=content,
        question_types=["MCQ", "Short Answer"],
        total_questions=3
    )
    assert len(questions) >= 1
    for q in questions:
        assert q.source_reference is not None
        assert q.correct_answer is not None

    # Persist Questions
    saved_q_count = temp_db.save_questions(questions, lecture_id=lec_id)
    assert saved_q_count == len(questions)

    # 5. Quiz Administration
    engine = QuizEngine()
    quiz = engine.create_quiz(questions=questions, num_questions=len(questions), seed=42)
    engine.start_quiz(quiz)

    # Submit answer for question 1
    engine.submit_answer(quiz.questions[0].question_id, quiz.questions[0].correct_answer)
    # Skip question 2 if present
    if len(quiz.questions) > 1:
        engine.skip_question(quiz.questions[1].question_id)

    # 6. Evaluation and Finalization
    quiz_result = engine.finish_quiz()
    assert quiz_result.earned_marks > 0.0
    assert quiz_result.total_marks > 0.0

    # 7. Attempt Persistence
    attempt_id = temp_db.save_quiz_attempt(quiz_result, lecture_id=lec_id)
    assert attempt_id.startswith("attempt_")

    # 8. Learning Analytics Calculation
    summary = temp_db.get_analytics_summary()
    assert summary["total_lectures"] == 1
    assert summary["total_questions"] == len(questions)
    assert summary["total_quizzes_attempted"] == 1
    assert summary["average_score_percentage"] is not None
    assert summary["average_score_percentage"] > 0.0


def test_end_to_end_pdf_pipeline(temp_db):
    """
    Validates PDF generation, PyMuPDF extraction with page citations,
    and provenance retention through database storage.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)

    try:
        # Create a 2-page test PDF
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 72), "Page 1: Supervised learning models map inputs to targets using loss functions.")
        page2 = doc.new_page()
        page2.insert_text((50, 72), "Page 2: Regularization introduces parameter constraints to avoid overfitting.")
        doc.save(str(pdf_path))
        doc.close()

        # Ingest PDF
        content = PDFProcessor().extract_text(str(pdf_path), title="PDF Lecture ML")
        assert content.source_type == "pdf"
        assert len(content.segments) == 2
        assert content.segments[0].source_label == "Page 1"
        assert content.segments[1].source_label == "Page 2"

        # Save to database
        lec_id = temp_db.save_lecture(content)
        retrieved = temp_db.get_lecture(lec_id)
        assert retrieved is not None
        assert retrieved.total_segments == 2

    finally:
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception:
                pass


def test_ui_page_rendering_without_exceptions(temp_db):
    """
    Directly invokes all UI page renderers with a mocked Streamlit environment
    to ensure no page raises unexpected runtime or syntax exceptions.
    """
    import streamlit as st
    from ui.home import render_home_page
    from ui.upload import render_upload_page
    from ui.understanding import render_understanding_page
    from ui.questions import render_questions_page
    from ui.quiz import render_quiz_page
    from ui.results import render_results_page
    from ui.analytics import render_analytics_page
    from ui.history import render_history_page

    # Populate temporary DB with 1 lecture, 1 question, 1 attempt
    content = TextProcessor().process_raw_text(
        "Artificial neural networks are computational models inspired by biological neural networks. "
        "They consist of interconnected nodes organized in layers that process inputs.",
        title="Neural Networks Intro"
    )
    lec_id = temp_db.save_lecture(content)

    q = GeneratedQuestion(
        question_id="int_q1",
        question_text="What are neural networks?",
        question_type="Short Answer",
        options=[],
        correct_answer="Computational models inspired by biological neural networks",
        explanation="Brain-inspired computing.",
        topic="Neural Networks",
        subtopic="Introduction",
        bloom_level="Understand",
        difficulty="Easy",
        source_reference="Section 1"
    )
    temp_db.save_questions([q], lecture_id=lec_id)

    # Mock session state
    st.session_state["active_lecture"] = content
    st.session_state["active_lecture_id"] = lec_id
    st.session_state["generated_questions"] = [q]

    # Verify each page function executes smoothly without unhandled exceptions
    with patch("streamlit.button", return_value=False), \
         patch("streamlit.rerun"):
        render_home_page(temp_db)
        render_upload_page(temp_db)
        render_understanding_page(temp_db)
        render_questions_page(temp_db)
        render_quiz_page(temp_db)
        render_results_page(temp_db)
        render_analytics_page(temp_db)
        render_history_page(temp_db)


def test_txt_upload_and_pasted_text_integration(temp_db, tmp_path):
    """
    Validates TXT file upload and pasted text integration workflows:
    1. TXT file upload with process_txt_file.
    2. Pasted text processing with process_raw_text.
    3. LectureContent return type and chunking.
    4. Empty / insufficient text handling with TextProcessingError.
    5. Source metadata and provenance preservation throughout SQLite persistence.
    """
    from utils.file_manager import save_upload_to_disk, cleanup_temp_file
    from modules.text_processor import TextProcessor, TextProcessingError

    class MockUploadedFile:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data
            self.size = len(data)

        def getbuffer(self):
            return self._data

    text_body = (
        "Support Vector Machines find an optimal separating hyperplane in high-dimensional feature spaces.\n\n"
        "The margin is defined as the distance between the decision boundary and the closest training points.\n\n"
        "Kernel methods allow SVMs to construct non-linear decision boundaries by mapping inputs implicitly."
    )

    # 1. TXT File Upload Simulation
    mock_upload = MockUploadedFile("svm_lecture.txt", text_body.encode("utf-8"))
    saved_path = save_upload_to_disk(mock_upload, destination_dir=tmp_path)
    assert saved_path.exists()
    assert saved_path.suffix == ".txt"

    # Use the corrected API: process_txt_file
    processor = TextProcessor()
    content_from_file = processor.process_txt_file(saved_path, title="SVM Lecture")

    # 2 & 3. LectureContent return type verification
    assert isinstance(content_from_file, LectureContent)
    assert content_from_file.source_type == "text"
    assert content_from_file.title == "SVM Lecture"
    assert content_from_file.total_words > 20
    assert content_from_file.total_segments == 3
    assert content_from_file.total_chunks >= 1

    # 5. Source metadata preservation
    assert content_from_file.source_filename == saved_path.name
    assert content_from_file.metadata.get("input_method") == "file_upload"
    assert content_from_file.segments[0].source_type == "text"
    assert content_from_file.segments[0].source_label == "Section 1"

    # Verify temp file cleanup
    cleanup_success = cleanup_temp_file(saved_path)
    assert cleanup_success is True
    assert not saved_path.exists()

    # Persist to database and retrieve
    lec_id = temp_db.save_lecture(content_from_file)
    retrieved = temp_db.get_lecture(lec_id)
    assert retrieved is not None
    assert retrieved.title == "SVM Lecture"
    assert retrieved.source_type == "text"
    assert retrieved.total_segments == 3

    # 2. Pasted Text Processing
    pasted_content = processor.process_raw_text(text_body, title="Pasted SVM Notes")
    assert isinstance(pasted_content, LectureContent)
    assert pasted_content.source_type == "text"
    assert pasted_content.title == "Pasted SVM Notes"
    assert pasted_content.source_filename == "manual_input.txt"
    assert pasted_content.metadata.get("input_method") == "manual_entry"
    assert pasted_content.total_words > 20

    # 4. Empty Text and Insufficient Content Handling
    with pytest.raises(TextProcessingError):
        processor.process_raw_text("")

    with pytest.raises(TextProcessingError):
        processor.process_raw_text("   \n\t  ")

    with pytest.raises(TextProcessingError):
        processor.process_raw_text("Too short")
