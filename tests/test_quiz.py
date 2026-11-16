"""
NEUROLECT AI - Test Quiz and Assessment Engine
Comprehensive test suite validating:
- Quiz creation, topic/type/difficulty filtering, and question limits.
- Duplicate question prevention (ID and semantic similarity).
- Question and MCQ option randomization with deterministic seeds.
- Interactive state tracking (submit, skip, progress).
- Deterministic MCQ evaluation (exact, normalized, incorrect, unanswered).
- Short answer semantic evaluation, thresholding, and partial credit.
- Descriptive multi-criteria rubric scoring and missing concepts extraction.
- Boundary conditions (0.0 <= marks <= max_marks).
- Topic-wise, difficulty-wise, and Bloom-wise performance breakdowns.
- Source reference and provenance preservation.
- Model reuse via ModelLoader without redundant allocations.
- Model failure fallback (does NOT award full marks; flags for manual review).
"""

import pytest
from unittest.mock import MagicMock, patch
from modules.question_models import GeneratedQuestion
from modules.quiz_models import (
    Quiz,
    QuizQuestion,
    StudentAnswer,
    QuestionResult,
    QuizResult
)
from modules.answer_evaluator import AnswerEvaluator
from modules.quiz_engine import QuizEngine, QuizEngineError
from models.model_loader import ModelLoader


@pytest.fixture
def sample_generated_questions():
    """Builds a realistic diverse set of 6 GeneratedQuestion instances across topics and types."""
    return [
        GeneratedQuestion(
            question_id="q1",
            question_text="What do classification models predict?",
            question_type="MCQ",
            options=["Discrete category labels", "Continuous real values", "Training noise", "Voronoi clusters"],
            correct_answer="Discrete category labels",
            explanation="Classification models output discrete categories.",
            topic="Supervised Learning",
            subtopic="Classification",
            bloom_level="Remember",
            difficulty="Easy",
            source_reference="Page 1",
            generation_metadata={"segment_id": 0}
        ),
        GeneratedQuestion(
            question_id="q2",
            question_text="What does L1 Lasso regularization introduce to induce sparsity?",
            question_type="Short Answer",
            options=[],
            correct_answer="An absolute weight penalty that induces parameter sparsity",
            explanation="L1 applies an absolute value penalty.",
            topic="Supervised Learning",
            subtopic="Regularization",
            bloom_level="Understand",
            difficulty="Medium",
            source_reference="Page 2",
            generation_metadata={"segment_id": 1}
        ),
        GeneratedQuestion(
            question_id="q3",
            question_text="Explain how backpropagation computes gradients across deep neural network layers.",
            question_type="Descriptive",
            options=[],
            correct_answer="Backpropagation calculates loss gradients with respect to weights using the chain rule across layers.",
            explanation="Chain rule propagates error backwards.",
            topic="Neural Networks",
            subtopic="Optimization",
            bloom_level="Understand",
            difficulty="Medium",
            source_reference="Page 3",
            generation_metadata={"segment_id": 2}
        ),
        GeneratedQuestion(
            question_id="q4",
            question_text="What is the theoretical mechanism governing activation functions in neural networks?",
            question_type="Conceptual",
            options=[],
            correct_answer="Activation functions introduce non-linearity into network representations.",
            explanation="Non-linearity enables learning complex functions.",
            topic="Neural Networks",
            subtopic="Architecture",
            bloom_level="Understand",
            difficulty="Medium",
            source_reference="Page 3",
            generation_metadata={"segment_id": 2}
        ),
        GeneratedQuestion(
            question_id="q5",
            question_text="How can L2 Ridge regularization be applied to prevent overfitting in regression models?",
            question_type="Application",
            options=[],
            correct_answer="Penalizes squared weight magnitudes to shrink parameters smoothly.",
            explanation="L2 prevents weights from growing excessively large.",
            topic="Supervised Learning",
            subtopic="Regularization",
            bloom_level="Apply",
            difficulty="Hard",
            source_reference="Page 2",
            generation_metadata={"segment_id": 1}
        ),
        GeneratedQuestion(
            question_id="q6",
            question_text="Compare and contrast ACID transaction properties with eventual consistency in distributed systems.",
            question_type="Analytical",
            options=[],
            correct_answer="ACID guarantees immediate atomicity and consistency, whereas eventual consistency allows temporary replica divergence under partitions.",
            explanation="CAP theorem trade-offs between consistency and availability.",
            topic="Distributed Systems",
            subtopic="Transactions",
            bloom_level="Analyze",
            difficulty="Hard",
            source_reference="Page 4",
            generation_metadata={"segment_id": 3}
        )
    ]


@pytest.fixture
def evaluator():
    return AnswerEvaluator()


@pytest.fixture
def engine(evaluator):
    return QuizEngine(evaluator=evaluator)


# =========================================================================
# 1. QUIZ CREATION, FILTERING & DEDUPLICATION TESTS (DETERMINISTIC)
# =========================================================================

def test_quiz_creation_and_filtering(engine, sample_generated_questions):
    """Verify creating a quiz with topic, type, and question limits."""
    quiz = engine.create_quiz(
        questions=sample_generated_questions,
        title="Supervised Learning Quiz",
        topic_filter=["Supervised Learning"],
        num_questions=2
    )

    assert len(quiz.questions) == 2
    assert quiz.title == "Supervised Learning Quiz"
    for q in quiz.questions:
        assert q.topic == "Supervised Learning"
    assert quiz.total_marks > 0


def test_quiz_type_and_difficulty_filtering(engine, sample_generated_questions):
    """Verify filtering questions by difficulty and question type."""
    quiz = engine.create_quiz(
        questions=sample_generated_questions,
        type_filter=["MCQ", "Short Answer"],
        difficulty_filter=["Easy", "Medium"]
    )

    assert len(quiz.questions) >= 1
    for q in quiz.questions:
        assert q.question_type in ["MCQ", "Short Answer"]
        assert q.difficulty in ["Easy", "Medium"]


def test_duplicate_question_prevention(engine, sample_generated_questions):
    """Verify identical IDs and near-duplicate question texts are rejected."""
    # Duplicate ID
    q_dup_id = GeneratedQuestion.from_dict(sample_generated_questions[0].to_dict())
    
    # Near duplicate text (similarity > 0.85)
    q_dup_text = GeneratedQuestion(
        question_id="q_dup_text",
        question_text="What do classification models predict in machine learning?",  # near duplicate of q1
        question_type="MCQ",
        options=["Discrete category labels", "Continuous real values", "Training noise", "Voronoi clusters"],
        correct_answer="Discrete category labels",
        explanation="",
        topic="Supervised Learning",
        subtopic="Classification",
        bloom_level="Remember",
        difficulty="Easy",
        source_reference="Page 1"
    )

    input_pool = [sample_generated_questions[0], q_dup_id, q_dup_text, sample_generated_questions[1]]
    quiz = engine.create_quiz(questions=input_pool, randomize_questions=False)

    # Should only contain 2 questions (q1 and q2), rejecting duplicates
    assert len(quiz.questions) == 2
    assert quiz.questions[0].question_id == "q1"
    assert quiz.questions[1].question_id == "q2"


def test_question_and_option_randomization(engine, sample_generated_questions):
    """Verify that deterministic seeding controls question and option shuffling."""
    quiz1 = engine.create_quiz(
        questions=sample_generated_questions,
        randomize_questions=True,
        randomize_options=True,
        seed=42
    )
    quiz2 = engine.create_quiz(
        questions=sample_generated_questions,
        randomize_questions=True,
        randomize_options=True,
        seed=42
    )

    # Same seed yields identical order
    assert [q.question_id for q in quiz1.questions] == [q.question_id for q in quiz2.questions]
    for q1, q2 in zip(quiz1.questions, quiz2.questions):
        if q1.question_type == "MCQ":
            assert q1.options == q2.options


# =========================================================================
# 2. INTERACTIVE QUIZ STATE TRACKING TESTS (DETERMINISTIC)
# =========================================================================

def test_quiz_state_tracking(engine, sample_generated_questions):
    """Verify starting quiz, submitting answers, skipping, and progress reporting."""
    quiz = engine.create_quiz(questions=sample_generated_questions, num_questions=3, seed=1)
    engine.start_quiz(quiz)

    prog1 = engine.get_quiz_progress()
    assert prog1["active"] is True
    assert prog1["total_questions"] == 3
    assert prog1["answered_count"] == 0
    assert prog1["unanswered_count"] == 3

    # Submit answer for question 1
    engine.submit_answer(quiz.questions[0].question_id, "Discrete category labels")
    # Skip question 2
    engine.skip_question(quiz.questions[1].question_id)

    prog2 = engine.get_quiz_progress()
    assert prog2["answered_count"] == 1
    assert prog2["skipped_count"] == 1
    assert prog2["unanswered_count"] == 1


def test_submit_answer_no_active_quiz_raises(engine):
    """Verify submitting an answer without an active quiz raises QuizEngineError."""
    with pytest.raises(QuizEngineError):
        engine.submit_answer("q1", "Some answer")


# =========================================================================
# 3. MCQ EVALUATION TESTS (DETERMINISTIC)
# =========================================================================

def test_mcq_evaluation_exact_match(evaluator, sample_generated_questions):
    """Verify deterministic MCQ evaluation on correct answer."""
    q_mcq = QuizQuestion.from_generated_question(sample_generated_questions[0])
    res = evaluator.evaluate_mcq(q_mcq, "Discrete category labels")

    assert res.evaluation_status == "correct"
    assert res.marks_awarded == q_mcq.max_marks
    assert len(res.missing_concepts) == 0


def test_mcq_evaluation_with_option_letter_prefix(evaluator, sample_generated_questions):
    """Verify normalizing option letter prefixes like '[A] Discrete category labels'."""
    q_mcq = QuizQuestion.from_generated_question(sample_generated_questions[0])
    res = evaluator.evaluate_mcq(q_mcq, "[A] Discrete category labels")

    assert res.evaluation_status == "correct"
    assert res.marks_awarded == q_mcq.max_marks


def test_mcq_evaluation_incorrect_answer(evaluator, sample_generated_questions):
    """Verify deterministic MCQ evaluation on incorrect answer."""
    q_mcq = QuizQuestion.from_generated_question(sample_generated_questions[0])
    res = evaluator.evaluate_mcq(q_mcq, "Continuous real values")

    assert res.evaluation_status == "incorrect"
    assert res.marks_awarded == 0.0
    assert "Discrete category labels" in res.reference_answer


def test_mcq_evaluation_unanswered(evaluator, sample_generated_questions):
    """Verify empty answer returns 0 marks and unanswered status."""
    q_mcq = QuizQuestion.from_generated_question(sample_generated_questions[0])
    res = evaluator.evaluate_mcq(q_mcq, "")

    assert res.evaluation_status == "unanswered"
    assert res.marks_awarded == 0.0


# =========================================================================
# 4. SHORT ANSWER EVALUATION TESTS (HEURISTIC + SEMANTIC MODEL)
# =========================================================================

def test_short_answer_exact_match(evaluator, sample_generated_questions):
    """Verify exact match on short answer awards full marks."""
    q_sa = QuizQuestion.from_generated_question(sample_generated_questions[1])
    res = evaluator.evaluate_short_answer(
        q_sa,
        "An absolute weight penalty that induces parameter sparsity"
    )
    assert res.evaluation_status == "correct"
    assert res.marks_awarded == q_sa.max_marks


def test_short_answer_semantic_paraphrase(evaluator, sample_generated_questions):
    """Verify semantically equivalent paraphrase awards full or partial marks."""
    q_sa = QuizQuestion.from_generated_question(sample_generated_questions[1])
    # Paraphrased version
    res = evaluator.evaluate_short_answer(
        q_sa,
        "An absolute penalty on weights creating sparsity in parameters"
    )
    assert res.evaluation_status in ["correct", "partially_correct"]
    assert res.marks_awarded >= 0.5 * q_sa.max_marks


def test_short_answer_incorrect(evaluator, sample_generated_questions):
    """Verify completely wrong short answer receives 0 marks."""
    q_sa = QuizQuestion.from_generated_question(sample_generated_questions[1])
    res = evaluator.evaluate_short_answer(
        q_sa,
        "Gradient descent with momentum on stochastic batches"
    )
    assert res.evaluation_status == "incorrect"
    assert res.marks_awarded == 0.0


# =========================================================================
# 5. DESCRIPTIVE RUBRIC EVALUATION TESTS (HEURISTIC)
# =========================================================================

def test_descriptive_answer_rubric_scoring(evaluator, sample_generated_questions):
    """Verify multi-criteria rubric scoring for descriptive questions."""
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])
    student_ans = (
        "Backpropagation calculates gradients of the loss function using the chain rule "
        "across network layers to update weights."
    )
    res = evaluator.evaluate_descriptive(q_desc, student_ans)

    assert res.marks_awarded > 0.0
    assert res.marks_awarded <= q_desc.max_marks
    assert "concept_coverage" in res.score_breakdown
    assert "semantic_relevance" in res.score_breakdown
    assert "completeness" in res.score_breakdown
    assert "keyword_accuracy" in res.score_breakdown


def test_descriptive_partial_credit_on_incomplete_response(evaluator, sample_generated_questions):
    """Verify partial credit is awarded when response mentions only part of the concept."""
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])
    # Sparse answer missing chain rule and loss explanation
    res = evaluator.evaluate_descriptive(q_desc, "It computes gradients.")

    assert res.evaluation_status in ["partially_correct", "incorrect"]
    assert res.marks_awarded < q_desc.max_marks
    assert len(res.missing_concepts) > 0


# =========================================================================
# 6. BOUNDARY & ERROR HANDLING TESTS (DETERMINISTIC)
# =========================================================================

def test_empty_student_answer_handling(evaluator, sample_generated_questions):
    """Verify empty student answers receive 0.0 marks across all question types."""
    for gq in sample_generated_questions:
        qq = QuizQuestion.from_generated_question(gq)
        res = evaluator.evaluate_answer(qq, "   ")
        assert res.marks_awarded == 0.0
        assert res.evaluation_status == "unanswered"


def test_missing_reference_answer_handling(evaluator):
    """Verify missing reference answer flags requires_manual_review without awarding marks."""
    qq = QuizQuestion(
        question_id="q_no_ref",
        question_text="What is intelligence?",
        question_type="Short Answer",
        options=[],
        correct_answer="",  # Missing reference
        explanation="",
        topic="General",
        subtopic="General",
        difficulty="Easy",
        bloom_level="Understand",
        source_reference="Page 1",
        max_marks=2.0
    )
    res = evaluator.evaluate_answer(qq, "The ability to learn and apply knowledge.")
    assert res.evaluation_status == "requires_manual_review"
    assert res.marks_awarded == 0.0


def test_score_boundaries_strict_capping(engine, sample_generated_questions):
    """Verify marks awarded never exceed maximum marks or drop below 0.0."""
    quiz = engine.create_quiz(questions=sample_generated_questions, num_questions=3, seed=10)
    engine.start_quiz(quiz)

    # Submit an excessively wordy answer
    engine.submit_answer(quiz.questions[0].question_id, quiz.questions[0].correct_answer)
    engine.submit_answer(quiz.questions[1].question_id, "Completely irrelevant text that is long.")

    result = engine.finish_quiz()
    assert 0.0 <= result.earned_marks <= result.total_marks
    assert 0.0 <= result.percentage <= 100.0


# =========================================================================
# 7. PERFORMANCE BREAKDOWN & PROVENANCE PRESERVATION TESTS (DETERMINISTIC)
# =========================================================================

def test_topic_and_difficulty_performance_breakdowns(engine, sample_generated_questions):
    """Verify topic-wise and difficulty-wise performance breakdowns match arithmetic totals."""
    quiz = engine.create_quiz(questions=sample_generated_questions, num_questions=4, seed=5)
    engine.start_quiz(quiz)

    # Answer all questions correctly
    for q in quiz.questions:
        engine.submit_answer(q.question_id, q.correct_answer)

    result = engine.finish_quiz()

    # Check topic breakdown
    assert len(result.topic_scores) >= 1
    total_topic_marks = sum(t["total_marks"] for t in result.topic_scores.values())
    earned_topic_marks = sum(t["earned_marks"] for t in result.topic_scores.values())
    assert round(total_topic_marks, 2) == round(result.total_marks, 2)
    assert round(earned_topic_marks, 2) == round(result.earned_marks, 2)

    # Check difficulty breakdown
    assert len(result.difficulty_scores) >= 1
    total_diff_marks = sum(d["total_marks"] for d in result.difficulty_scores.values())
    assert round(total_diff_marks, 2) == round(result.total_marks, 2)


def test_source_reference_provenance_preservation(engine, sample_generated_questions):
    """Verify source references survive quiz administration and appear on question results."""
    quiz = engine.create_quiz(questions=sample_generated_questions)
    engine.start_quiz(quiz)

    for q in quiz.questions:
        engine.submit_answer(q.question_id, q.correct_answer)

    result = engine.finish_quiz()
    for qr in result.question_results:
        assert qr.source_reference in ["Page 1", "Page 2", "Page 3", "Page 4"]


# =========================================================================
# 8. MODEL REUSE & FAILURE FALLBACK TESTS
# =========================================================================

def test_model_reuse_without_duplicate_allocation(evaluator):
    """Verify AnswerEvaluator reuses the cached SentenceTransformer instance."""
    st1 = evaluator._get_embedding_model()
    st2 = ModelLoader.get_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    assert st1 is st2  # Exact same object reference in memory


def test_model_unavailability_does_not_award_full_marks(evaluator, sample_generated_questions):
    """Verify that when embedding model fails, evaluator flags manual review instead of full credit."""
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])

    with patch.object(evaluator, "_get_embedding_model", return_value=None):
        res = evaluator.evaluate_descriptive(q_desc, "Backpropagation calculates gradients.")
        assert res.evaluation_status == "requires_manual_review"
        assert res.marks_awarded < q_desc.max_marks


# =========================================================================
# 9. TARGETED AUDIT VERIFICATION TESTS
# =========================================================================

@pytest.mark.parametrize("seed", [11, 23, 42, 77, 99])
def test_mcq_option_randomization_repeated_seeds(sample_generated_questions, seed):
    """
    CRITICAL AUDIT ITEM 1:
    Verify across multiple seeds that MCQ option shuffling:
    1. Keeps options intact and distinct.
    2. Correctly matches the correct answer regardless of its randomized position.
    3. Correctly maps letter selections (A, B, C, D) to displayed option indices.
    4. Selecting an incorrect option or incorrect letter never receives marks.
    """
    engine = QuizEngine()
    evaluator = AnswerEvaluator()
    quiz = engine.create_quiz(
        questions=sample_generated_questions,
        type_filter=["MCQ"],
        num_questions=1,
        randomize_options=True,
        seed=seed
    )
    mcq_q = quiz.questions[0]
    ref_answer = mcq_q.correct_answer

    # Verify options contain the reference answer
    assert ref_answer in mcq_q.options
    assert len(mcq_q.options) == 4

    correct_idx = mcq_q.options.index(ref_answer)
    correct_letter = chr(ord("A") + correct_idx)

    # 1. Text submission of correct answer awards full marks
    res_text = evaluator.evaluate_mcq(mcq_q, ref_answer)
    assert res_text.evaluation_status == "correct"
    assert res_text.marks_awarded == mcq_q.max_marks

    # 2. Letter submission of correct option awards full marks
    for letter_fmt in [correct_letter, f"[{correct_letter}]", f"({correct_letter})", f"{correct_letter}."]:
        res_letter = evaluator.evaluate_mcq(mcq_q, letter_fmt)
        assert res_letter.evaluation_status == "correct"
        assert res_letter.marks_awarded == mcq_q.max_marks

    # 3. Incorrect option text awards 0 marks
    incorrect_idx = (correct_idx + 1) % len(mcq_q.options)
    incorrect_text = mcq_q.options[incorrect_idx]
    res_inc_text = evaluator.evaluate_mcq(mcq_q, incorrect_text)
    assert res_inc_text.evaluation_status == "incorrect"
    assert res_inc_text.marks_awarded == 0.0

    # 4. Incorrect letter choice awards 0 marks
    incorrect_letter = chr(ord("A") + incorrect_idx)
    res_inc_letter = evaluator.evaluate_mcq(mcq_q, incorrect_letter)
    assert res_inc_letter.evaluation_status == "incorrect"
    assert res_inc_letter.marks_awarded == 0.0


def test_short_answer_conservatism_matrix(evaluator, sample_generated_questions):
    """
    CRITICAL AUDIT ITEM 2:
    Short-answer evaluation conservatism matrix covering 6 required scenarios:
    1. Semantically correct paraphrase -> Full/high marks
    2. Semantically close but technically contradictory (squared vs absolute penalty) -> 0 marks
    3. High generic similarity with low concept coverage -> 0 marks
    4. Keyword overlap with completely irrelevant domain -> 0 marks
    5. Partially correct answer -> Partial credit (50%)
    6. Unanswered / empty -> 0 marks
    """
    q_sa = QuizQuestion.from_generated_question(sample_generated_questions[1])
    # Reference: "An absolute weight penalty that induces parameter sparsity"

    # Case 1: Semantically correct paraphrase
    res_corr = evaluator.evaluate_short_answer(
        q_sa,
        "It imposes an absolute penalty on model weights which forces coefficients to zero"
    )
    assert res_corr.evaluation_status in ["correct", "partially_correct"]
    assert res_corr.marks_awarded >= 0.5 * q_sa.max_marks

    # Case 2: Technically contradictory answer (L2 squared penalty for L1 question)
    res_contra = evaluator.evaluate_short_answer(
        q_sa,
        "A squared weight penalty that shrinks parameters smoothly"
    )
    assert res_contra.evaluation_status == "incorrect"
    assert res_contra.marks_awarded == 0.0

    # Case 3: Negation inversion (contradiction of core fact)
    res_neg = evaluator.evaluate_short_answer(
        q_sa,
        "It does not impose any penalty and cannot induce parameter sparsity"
    )
    assert res_neg.evaluation_status == "incorrect"
    assert res_neg.marks_awarded == 0.0

    # Case 4: High generic similarity with very low concept coverage
    res_generic = evaluator.evaluate_short_answer(
        q_sa,
        "This optimization algorithm minimizes a global cost objective using numerical methods"
    )
    assert res_generic.evaluation_status == "incorrect"
    assert res_generic.marks_awarded == 0.0

    # Case 5: Partially correct answer (mentions penalty, lacks sparsity)
    res_partial = evaluator.evaluate_short_answer(
        q_sa,
        "An absolute weight penalty applied during model training"
    )
    assert res_partial.evaluation_status == "partially_correct"
    assert res_partial.marks_awarded == round(0.5 * q_sa.max_marks, 2)
    assert "sparsity" in [c.lower() for c in res_partial.missing_concepts]

    # Case 6: Empty answer
    res_empty = evaluator.evaluate_short_answer(q_sa, "")
    assert res_empty.evaluation_status == "unanswered"
    assert res_empty.marks_awarded == 0.0


def test_descriptive_verbose_gibberish_receives_zero_marks(evaluator, sample_generated_questions):
    """
    CRITICAL AUDIT ITEM 3:
    Verify that length/completeness is strictly gated by concept coverage,
    so verbose gibberish receives 0.0 marks.
    """
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])
    # 65 words of irrelevant filler text
    gibberish = (
        "The quick brown fox jumps over the lazy dog repeatedly without knowing anything "
        "about mathematics or computer science. Furthermore, the weather today is unusually "
        "warm and sunny with blue skies across the horizon. We observed birds migrating south "
        "for the winter while leaves fell from the oak trees in picturesque fashion, continuing "
        "our leisurely walk through the countryside park with absolute serenity."
    )
    res = evaluator.evaluate_descriptive(q_desc, gibberish)
    assert res.evaluation_status == "incorrect"
    assert res.marks_awarded == 0.0
    assert len(res.missing_concepts) > 0


def test_quiz_incomplete_submission_status(engine, sample_generated_questions):
    """
    CRITICAL AUDIT ITEM 5:
    Verify that a quiz with unanswered questions receives is_complete=False
    and accurate unanswered counts.
    """
    quiz = engine.create_quiz(questions=sample_generated_questions, num_questions=3, seed=12)
    engine.start_quiz(quiz)

    # Submit answer for only 1 of the 3 questions
    engine.submit_answer(quiz.questions[0].question_id, quiz.questions[0].correct_answer)

    result = engine.finish_quiz()
    assert result.is_complete is False
    assert result.total_questions == 3
    assert result.answered_questions == 1
    assert result.unanswered_questions == 2


def test_cannot_resubmit_finished_quiz(engine, sample_generated_questions):
    """
    CRITICAL AUDIT ITEM 5:
    Verify that a completed quiz cannot be resubmitted or restarted.
    """
    quiz = engine.create_quiz(questions=sample_generated_questions, num_questions=2, seed=1)
    engine.start_quiz(quiz)
    engine.submit_answer(quiz.questions[0].question_id, quiz.questions[0].correct_answer)
    engine.finish_quiz()

    # Submitting an answer after finish raises QuizEngineError
    with pytest.raises(QuizEngineError, match="No active quiz session"):
        engine.submit_answer(quiz.questions[0].question_id, "foo")

    # Starting the already-finalized quiz raises QuizEngineError
    with pytest.raises(QuizEngineError, match="already been submitted and finalized"):
        engine.start_quiz(quiz)


# =========================================================================
# 10. FINAL APPROVAL VERIFICATION TESTS
# =========================================================================

def test_descriptive_rubric_bounds_and_length_invariance(evaluator, sample_generated_questions):
    """
    FINAL APPROVAL ITEM 1:
    Verify that each rubric component is strictly bounded in [0.0, 1.0],
    final marks do not exceed max_marks, and appending filler/length without
    adding relevant concepts does not increase score.
    """
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])

    # 1. Base response
    base_ans = "Backpropagation calculates loss gradients using the chain rule across layers."
    res_base = evaluator.evaluate_descriptive(q_desc, base_ans)

    # Check bounds
    for k, v in res_base.score_breakdown.items():
        if k != "error":
            assert 0.0 <= v <= 1.0, f"Component {k}={v} out of bounds"
    assert 0.0 <= res_base.marks_awarded <= q_desc.max_marks

    # 2. Padded response: same content plus 150 words of irrelevant filler
    padded_ans = (
        base_ans + " " +
        "Furthermore we took a walk in the sunny park observing birds and trees with absolute serenity. "
        "The weather was pleasant and calm, creating a relaxed atmosphere for reading books and listening "
        "to classical music while drinking hot tea on a Sunday morning. No additional technical concepts "
        "or mathematical definitions are being introduced in this paragraph whatsoever."
    )
    res_padded = evaluator.evaluate_descriptive(q_desc, padded_ans)

    # Adding filler without concepts must NOT increase marks
    assert res_padded.marks_awarded <= res_base.marks_awarded


def test_lexical_overlap_not_treated_as_factual_correctness(evaluator, sample_generated_questions):
    """
    FINAL APPROVAL ITEM 1:
    Verify that high lexical overlap with scrambled/contradictory meaning
    triggers manual review and does not blindly award full marks.
    """
    q_desc = QuizQuestion.from_generated_question(sample_generated_questions[2])
    # Jumbled words from reference forming non-meaningful text
    scrambled = "Respect rule calculates layers across chain the weights loss with backpropagation."
    res = evaluator.evaluate_descriptive(q_desc, scrambled)

    # Must not receive full marks; high discrepancy flags manual review or incorrect
    assert res.marks_awarded < q_desc.max_marks
    if res.score_breakdown.get("concept_coverage", 0) >= 0.50 and res.score_breakdown.get("semantic_relevance", 0) < 0.35:
        assert res.evaluation_status == "requires_manual_review"


def test_model_failure_handling_comprehensive(evaluator, sample_generated_questions):
    """
    FINAL APPROVAL ITEM 2:
    Verify all 5 model failure modes:
    1. Missing reference answer.
    2. SentenceTransformer loading failure.
    3. Embedding generation exception.
    4. Invalid or empty embedding output.
    5. Unexpected evaluation exception.
    Guarantees failure never silently succeeds or awards full marks.
    """
    # 1. Missing reference answer
    q_no_ref = QuizQuestion.from_generated_question(sample_generated_questions[1])
    q_no_ref.correct_answer = ""
    res1 = evaluator.evaluate_answer(q_no_ref, "Valid student text")
    assert res1.evaluation_status == "requires_manual_review"
    assert res1.marks_awarded == 0.0

    # 2. SentenceTransformer loading failure (returns None)
    q_sa = QuizQuestion.from_generated_question(sample_generated_questions[1])
    with patch.object(evaluator, "_get_embedding_model", return_value=None):
        res2 = evaluator.evaluate_short_answer(q_sa, "An absolute weight penalty parameter sparsity")
        assert res2.evaluation_status == "requires_manual_review"
        assert res2.marks_awarded <= 0.5 * q_sa.max_marks
        assert res2.score_breakdown.get("lexical_fallback") == 1.0

    # 3. Embedding generation exception
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("CUDA out of memory exception")
    with patch.object(evaluator, "_get_embedding_model", return_value=mock_model):
        res3 = evaluator.evaluate_descriptive(q_sa, "An absolute penalty inducing sparsity")
        assert res3.evaluation_status == "requires_manual_review"
        assert res3.marks_awarded < q_sa.max_marks

    # 4. Invalid or empty embedding output
    mock_model_empty = MagicMock()
    mock_model_empty.encode.return_value = []
    with patch.object(evaluator, "_get_embedding_model", return_value=mock_model_empty):
        res4 = evaluator.evaluate_short_answer(q_sa, "An absolute penalty inducing sparsity")
        assert res4.evaluation_status == "requires_manual_review"
        assert res4.marks_awarded < q_sa.max_marks

    # 5. Unexpected evaluation exception
    with patch.object(evaluator, "evaluate_short_answer", side_effect=ValueError("Corrupted memory pointer")):
        res5 = evaluator.evaluate_answer(q_sa, "Any text")
        assert res5.evaluation_status == "requires_manual_review"
        assert res5.marks_awarded == 0.0


def test_source_provenance_complete_pipeline_flow(engine):
    """
    FINAL APPROVAL ITEM 3:
    Verify that source reference and segment ID survive through every stage:
    GeneratedQuestion -> QuizQuestion -> QuestionResult -> QuizResult.
    """
    gq = GeneratedQuestion(
        question_id="provenance_q1",
        question_text="What is gradient descent?",
        question_type="Short Answer",
        options=[],
        correct_answer="An iterative optimization algorithm for minimizing objective functions.",
        explanation="Iterative first-order optimization.",
        topic="Optimization",
        subtopic="Gradient Descent",
        bloom_level="Understand",
        difficulty="Easy",
        source_reference="Page 12, Paragraph 3",
        generation_metadata={"segment_id": 42}
    )

    # 1. GeneratedQuestion -> QuizQuestion
    qq = QuizQuestion.from_generated_question(gq, max_marks=2.0)
    assert qq.source_reference == "Page 12, Paragraph 3"
    assert qq.segment_id == 42
    assert qq.to_dict()["segment_id"] == 42

    # 2. QuizQuestion -> Quiz
    quiz = engine.create_quiz(questions=[gq], num_questions=1)
    engine.start_quiz(quiz)
    engine.submit_answer(quiz.questions[0].question_id, "An iterative optimization algorithm.")

    # 3. Quiz -> QuizResult -> QuestionResult
    quiz_result = engine.finish_quiz()
    assert len(quiz_result.question_results) == 1
    qr = quiz_result.question_results[0]

    assert qr.source_reference == "Page 12, Paragraph 3"
    assert qr.segment_id == 42
    assert qr.to_dict()["source_reference"] == "Page 12, Paragraph 3"
    assert qr.to_dict()["segment_id"] == 42


def test_topic_and_difficulty_analytics_with_all_outcome_types(engine):
    """
    FINAL APPROVAL ITEM 4:
    Verify topic-wise and difficulty-wise analytics on a quiz containing:
    - Correct answer
    - Partially correct answer
    - Incorrect answer
    - Skipped / unanswered question
    Verifies question counts, marks earned, total marks, unanswered counts,
    and ensures unanswered questions are included in denominators.
    """
    gqs = [
        GeneratedQuestion(
            question_id="qa_1",
            question_text="What do classification models predict?",
            question_type="MCQ",
            options=["Discrete category labels", "Continuous real values", "Noise", "Weights"],
            correct_answer="Discrete category labels",
            explanation="Outputs discrete labels.",
            topic="Supervised Learning",
            subtopic="Classification",
            bloom_level="Remember",
            difficulty="Easy",
            source_reference="Page 1",
            generation_metadata={"segment_id": 1}
        ),
        GeneratedQuestion(
            question_id="qa_2",
            question_text="What does L1 Lasso regularization introduce?",
            question_type="Short Answer",
            options=[],
            correct_answer="An absolute weight penalty that induces parameter sparsity",
            explanation="L1 imposes absolute penalty.",
            topic="Supervised Learning",
            subtopic="Regularization",
            bloom_level="Understand",
            difficulty="Medium",
            source_reference="Page 2",
            generation_metadata={"segment_id": 2}
        ),
        GeneratedQuestion(
            question_id="qa_3",
            question_text="What is backpropagation?",
            question_type="Short Answer",
            options=[],
            correct_answer="Calculates loss gradients across layers using chain rule",
            explanation="Chain rule propagates errors.",
            topic="Neural Networks",
            subtopic="Optimization",
            bloom_level="Understand",
            difficulty="Medium",
            source_reference="Page 3",
            generation_metadata={"segment_id": 3}
        ),
        GeneratedQuestion(
            question_id="qa_4",
            question_text="How does an autoencoder compress data?",
            question_type="Short Answer",
            options=[],
            correct_answer="Encodes inputs into a lower-dimensional latent bottleneck representation",
            explanation="Bottleneck forces compression.",
            topic="Neural Networks",
            subtopic="Architectures",
            bloom_level="Apply",
            difficulty="Hard",
            source_reference="Page 4",
            generation_metadata={"segment_id": 4}
        )
    ]

    # Create quiz with marks_per_type = 2.0 for all questions
    marks_map = {"MCQ": 2.0, "Short Answer": 2.0}
    quiz = engine.create_quiz(
        questions=gqs,
        marks_per_type=marks_map,
        randomize_questions=False,
        seed=1
    )
    engine.start_quiz(quiz)

    # 1. Answer Q1 correctly (2.0 / 2.0)
    engine.submit_answer("qa_1", "Discrete category labels")

    # 2. Answer Q2 partially (1.0 / 2.0)
    engine.submit_answer("qa_2", "An absolute weight penalty applied during model training")

    # 3. Answer Q3 incorrectly (0.0 / 2.0)
    engine.submit_answer("qa_3", "Completely unrelated text about gardening and soil nutrients")

    # 4. Skip Q4 (0.0 / 2.0)
    engine.skip_question("qa_4")

    result = engine.finish_quiz()

    # Overall Metrics
    assert result.total_questions == 4
    assert result.answered_questions == 3
    assert result.unanswered_questions == 1
    assert result.total_marks == 8.0
    assert result.earned_marks == 3.0
    assert result.correct_count == 1
    assert result.partial_count == 1
    assert result.incorrect_count == 1
    assert result.is_complete is False

    # Topic Analytics: Supervised Learning (Q1 + Q2)
    sl_stats = result.topic_scores["Supervised Learning"]
    assert sl_stats["total_marks"] == 4.0
    assert sl_stats["earned_marks"] == 3.0
    assert sl_stats["question_count"] == 2
    assert sl_stats["unanswered_count"] == 0
    assert sl_stats["percentage"] == 75.0

    # Topic Analytics: Neural Networks (Q3 + Q4)
    # Crucial: Total marks must be 4.0 (includes the skipped question Q4!)
    nn_stats = result.topic_scores["Neural Networks"]
    assert nn_stats["total_marks"] == 4.0
    assert nn_stats["earned_marks"] == 0.0
    assert nn_stats["question_count"] == 2
    assert nn_stats["unanswered_count"] == 1
    assert nn_stats["percentage"] == 0.0

    # Difficulty Analytics: Hard (Q4 only, which was skipped)
    hard_stats = result.difficulty_scores["Hard"]
    assert hard_stats["total_marks"] == 2.0
    assert hard_stats["earned_marks"] == 0.0
    assert hard_stats["question_count"] == 1
    assert hard_stats["unanswered_count"] == 1
    assert hard_stats["percentage"] == 0.0

    # Conservation of marks across categories
    assert sum(t["total_marks"] for t in result.topic_scores.values()) == result.total_marks
    assert sum(t["earned_marks"] for t in result.topic_scores.values()) == result.earned_marks


