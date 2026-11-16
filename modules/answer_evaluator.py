"""
NEUROLECT AI - Automated Answer Evaluation Engine
Evaluates student responses across all 6 question types:
- Deterministic string/option matching for Multiple Choice Questions (MCQs).
- Conservative semantic + concept-matching for Short Answer questions.
- Multi-criteria transparent rubric evaluation for Descriptive/Analytical responses:
  * Concept Coverage (40%)
  * Semantic Relevance (30%)
  * Response Completeness (20%)
  * Technical Vocabulary Accuracy (10%)
Generates constructive pedagogical feedback with explicit missing concepts.
Reuses existing SentenceTransformer instance without duplicate memory allocation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from difflib import SequenceMatcher
from config import MODELS
from models.model_loader import ModelLoader, ModelLoadingError
from modules.quiz_models import QuizQuestion, QuestionResult

logger = logging.getLogger("NEUROLECT.AnswerEvaluator")

# Domain antonyms and conceptual contrasts to detect factually contradictory answers
TECHNICAL_ANTONYMS = {
    "absolute": "squared",
    "squared": "absolute",
    "l1": "l2",
    "l2": "l1",
    "supervised": "unsupervised",
    "unsupervised": "supervised",
    "discrete": "continuous",
    "continuous": "discrete",
    "linear": "non-linear",
    "non-linear": "linear",
    "convex": "non-convex",
    "non-convex": "convex",
    "deterministic": "stochastic",
    "stochastic": "deterministic",
    "increase": "decrease",
    "decrease": "increase",
    "maximize": "minimize",
    "minimize": "maximize",
    "sparsity": "density",
    "density": "sparsity",
    "underfitting": "overfitting",
    "overfitting": "underfitting"
}


class AnswerEvaluationError(Exception):
    """Raised when an answer evaluation pipeline encounters unrecoverable failure."""
    pass


class AnswerEvaluator:
    """
    Evaluates student answers against lecture-grounded reference answers.
    Combines deterministic string verification with SentenceTransformers semantic scoring.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        concept_weight: float = 0.40,
        semantic_weight: float = 0.30,
        completeness_weight: float = 0.20,
        keyword_weight: float = 0.10
    ):
        self.model_name = model_name or MODELS.get("semantic_evaluator", "sentence-transformers/all-MiniLM-L6-v2")
        self.concept_weight = concept_weight
        self.semantic_weight = semantic_weight
        self.completeness_weight = completeness_weight
        self.keyword_weight = keyword_weight

        # Normalization of rubric weights to sum to 1.0
        total_w = self.concept_weight + self.semantic_weight + self.completeness_weight + self.keyword_weight
        if total_w > 0:
            self.concept_weight /= total_w
            self.semantic_weight /= total_w
            self.completeness_weight /= total_w
            self.keyword_weight /= total_w

    def _get_embedding_model(self) -> Any:
        """Retrieves cached SentenceTransformer instance via ModelLoader."""
        try:
            return ModelLoader.get_sentence_transformer(self.model_name)
        except ModelLoadingError as e:
            logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Normalizes casing, removes punctuation and redundant whitespace without stripping words."""
        if not text:
            return ""
        t = text.strip()
        # Strip standard option prefixes like '[A]', '(A)', 'A.'
        t = re.sub(r'^\[([A-Da-d])\]\s*', '', t)
        t = re.sub(r'^\(([A-Da-d])\)\s*', '', t)
        t = re.sub(r'^([A-Da-d])[\.\)]\s+', '', t)
        # Strip surrounding punctuation and redundant whitespace
        t = re.sub(r'[^\w\s]', ' ', t)
        return " ".join(t.split()).lower()

    def _extract_content_tokens(self, text: str) -> List[str]:
        """Extracts substantive content words (>= 4 characters, excluding stopwords)."""
        stop = {
            "what", "which", "where", "when", "does", "from", "with", "that", "this",
            "have", "been", "were", "explain", "describe", "state", "their", "there",
            "into", "about", "more", "other", "than", "then", "some", "such", "only"
        }
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return [w for w in words if w not in stop]

    def _compute_semantic_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Computes cosine similarity between two texts using cached SentenceTransformer.
        Returns float similarity between 0.0 and 1.0, or None if model unavailable.
        """
        model = self._get_embedding_model()
        if model is None:
            return None

        try:
            embs = model.encode([text1, text2], convert_to_tensor=False)
            if embs is None or len(embs) < 2 or embs[0] is None or embs[1] is None:
                return None
            norm0 = np.linalg.norm(embs[0])
            norm1 = np.linalg.norm(embs[1])
            if norm0 == 0 or norm1 == 0 or np.isnan(norm0) or np.isnan(norm1):
                return 0.0
            cos_sim = float(np.dot(embs[0], embs[1]) / (norm0 * norm1))
            if np.isnan(cos_sim):
                return 0.0
            return max(0.0, min(1.0, cos_sim))
        except Exception as e:
            logger.error(f"Semantic similarity computation error: {e}")
            return None

    def evaluate_mcq(self, question: QuizQuestion, student_answer: str) -> QuestionResult:
        """
        Evaluates Multiple Choice Questions deterministically.
        Exact match awards full marks; mismatch awards 0 marks.
        Supports full option text, prefixed text ('[A] Option'), or standalone letter choice ('A', '[A]', '(A)', 'A.').
        """
        student_raw = student_answer.strip() if student_answer else ""

        # Handle unanswered (empty string or whitespace)
        if not student_raw:
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type="MCQ",
                student_answer="",
                reference_answer=question.correct_answer,
                marks_awarded=0.0,
                max_marks=question.max_marks,
                evaluation_status="unanswered",
                feedback=f"No answer was provided. Correct option: '{question.correct_answer}'.",
                missing_concepts=[question.correct_answer],
                score_breakdown={"exact_match": 0.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )

        clean_ref = self._clean_text(question.correct_answer)

        # Handle standalone letter-based selection (e.g. 'A', '[A]', '(A)', 'A.', 'a)')
        letter_match = re.match(r'^\(?\[?([A-Da-d])[\)\]\.]?$', student_raw)
        if letter_match and question.options:
            letter = letter_match.group(1).upper()
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(question.options):
                chosen_opt = question.options[idx]
                is_correct = (
                    self._clean_text(chosen_opt) == clean_ref or
                    chosen_opt.strip().lower() == question.correct_answer.strip().lower()
                )
            else:
                is_correct = False
        else:
            clean_stud = self._clean_text(student_answer)
            if not clean_stud:
                return QuestionResult(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    question_type="MCQ",
                    student_answer=student_answer,
                    reference_answer=question.correct_answer,
                    marks_awarded=0.0,
                    max_marks=question.max_marks,
                    evaluation_status="unanswered",
                    feedback=f"No substantive answer was provided. Correct option: '{question.correct_answer}'.",
                    missing_concepts=[question.correct_answer],
                    score_breakdown={"exact_match": 0.0},
                    source_reference=question.source_reference,
                    topic=question.topic,
                    difficulty=question.difficulty,
                    bloom_level=question.bloom_level,
                    metadata=dict(question.metadata)
                )
            is_correct = (clean_stud == clean_ref)

        marks = question.max_marks if is_correct else 0.0
        status = "correct" if is_correct else "incorrect"

        if is_correct:
            feedback = f"Correct! You selected the right answer: '{question.correct_answer}'."
            missing = []
        else:
            feedback = (
                f"Incorrect. Your selection '{student_answer}' does not match the correct option. "
                f"Correct answer: '{question.correct_answer}'. {question.explanation}"
            )
            missing = [question.correct_answer]

        return QuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            question_type="MCQ",
            student_answer=student_answer,
            reference_answer=question.correct_answer,
            marks_awarded=marks,
            max_marks=question.max_marks,
            evaluation_status=status,
            feedback=feedback,
            missing_concepts=missing,
            score_breakdown={"exact_match": 1.0 if is_correct else 0.0},
            source_reference=question.source_reference,
            topic=question.topic,
            difficulty=question.difficulty,
            bloom_level=question.bloom_level,
            metadata=dict(question.metadata)
        )

    def evaluate_short_answer(self, question: QuizQuestion, student_answer: str) -> QuestionResult:
        """
        Evaluates short-answer questions using semantic similarity and concept coverage.
        Awards full credit, partial credit (50%), or zero credit with conservative thresholds.
        """
        clean_stud = self._clean_text(student_answer)
        clean_ref = self._clean_text(question.correct_answer)

        # Handle unanswered
        if not clean_stud:
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer="",
                reference_answer=question.correct_answer,
                marks_awarded=0.0,
                max_marks=question.max_marks,
                evaluation_status="unanswered",
                feedback=f"No answer was provided. Reference answer: '{question.correct_answer}'.",
                missing_concepts=self._extract_content_tokens(question.correct_answer),
                score_breakdown={"semantic_similarity": 0.0, "concept_coverage": 0.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )

        ref_tokens = self._extract_content_tokens(question.correct_answer)
        stud_tokens = self._extract_content_tokens(student_answer)

        # Token intersection & missing concepts
        matched_tokens = [w for w in ref_tokens if w in stud_tokens or any(SequenceMatcher(None, w, s).ratio() > 0.85 for s in stud_tokens)]
        missing = [w for w in ref_tokens if w not in matched_tokens]
        token_coverage = len(matched_tokens) / max(1, len(ref_tokens))

        # Check exact string match
        if clean_stud == clean_ref:
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer=student_answer,
                reference_answer=question.correct_answer,
                marks_awarded=question.max_marks,
                max_marks=question.max_marks,
                evaluation_status="correct",
                feedback=f"Fully accurate response. Matches reference answer exactly.",
                missing_concepts=[],
                score_breakdown={"semantic_similarity": 1.0, "concept_coverage": 1.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )

        # Semantic cosine similarity
        sim = self._compute_semantic_similarity(student_answer, question.correct_answer)

        if sim is None:
            # Model unavailable fallback: conservative partial credit capped at 50% max_marks
            # Guarantees that model failure/exception cannot accidentally award full marks
            fallback_marks = round(min(0.50, token_coverage * 0.50) * question.max_marks, 2)
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer=student_answer,
                reference_answer=question.correct_answer,
                marks_awarded=fallback_marks,
                max_marks=question.max_marks,
                evaluation_status="requires_manual_review",
                feedback="Semantic evaluator model was unavailable. Lexical fallback applied; flagged for manual faculty review.",
                missing_concepts=missing,
                score_breakdown={"semantic_similarity": 0.0, "concept_coverage": round(token_coverage, 2), "lexical_fallback": 1.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )

        # Check for technical contradictions (e.g. 'squared' when ref is 'absolute', 'unsupervised' when ref is 'supervised')
        has_contradiction = False
        for rw in ref_tokens:
            if rw in TECHNICAL_ANTONYMS and TECHNICAL_ANTONYMS[rw] in stud_tokens:
                has_contradiction = True
                break

        # Check for negation inversion (reference has no negation, but student introduces negation)
        negation_pattern = re.compile(r'\b(no|not|never|neither|nor|without any|does not|cannot)\b', re.IGNORECASE)
        ref_has_negation = bool(negation_pattern.search(question.correct_answer))
        stud_has_negation = bool(negation_pattern.search(student_answer))
        has_negation_inversion = stud_has_negation and not ref_has_negation

        # Conservative short-answer scoring logic
        if has_contradiction or has_negation_inversion:
            marks = 0.0
            status = "incorrect"
            feedback = f"Incorrect response. Contains conflicting technical concepts or inverts the reference concept: '{question.correct_answer}'."
        # High lexical overlap but low semantic similarity: do not treat lexical overlap as proof of factual correctness
        elif token_coverage >= 0.50 and sim < 0.35:
            marks = 0.0
            status = "requires_manual_review"
            feedback = (
                f"Discrepancy detected: response shares keywords but diverges semantically from the reference answer. "
                f"Flagged for manual review."
            )
        # 1. High similarity + strong concept presence -> Full Marks
        elif sim >= 0.75 and token_coverage >= 0.45:
            marks = question.max_marks
            status = "correct"
            feedback = "Accurate response. Successfully conveys the required concept."
        # 2. Moderate similarity AND minimum concept coverage -> Partial Credit (50%)
        # Strictly requires BOTH sim >= 0.55 AND token_coverage >= 0.30 (no credit on similarity alone)
        elif sim >= 0.55 and token_coverage >= 0.30:
            marks = round(0.5 * question.max_marks, 2)
            status = "partially_correct"
            missing_str = ", ".join(missing[:3]) if missing else "additional details"
            feedback = f"Partially correct. Captures the general idea, but missing key concepts: {missing_str}."
        # 3. Low similarity OR insufficient concept coverage -> Incorrect (0.0 marks)
        else:
            marks = 0.0
            status = "incorrect"
            feedback = f"Incorrect response. Does not sufficiently match reference concept: '{question.correct_answer}'."

        return QuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            question_type=question.question_type,
            student_answer=student_answer,
            reference_answer=question.correct_answer,
            marks_awarded=marks,
            max_marks=question.max_marks,
            evaluation_status=status,
            feedback=feedback,
            missing_concepts=missing,
            score_breakdown={"semantic_similarity": round(sim, 2), "concept_coverage": round(token_coverage, 2)},
            source_reference=question.source_reference,
            topic=question.topic,
            difficulty=question.difficulty,
            bloom_level=question.bloom_level,
            metadata=dict(question.metadata)
        )

    def evaluate_descriptive(self, question: QuizQuestion, student_answer: str) -> QuestionResult:
        """
        Evaluates open-ended questions (Descriptive, Conceptual, Application, Analytical)
        using a multi-criteria transparent rubric:
        - Concept Coverage (40%)
        - Semantic Relevance (30%)
        - Completeness (20%)
        - Technical Vocabulary Accuracy (10%)
        """
        clean_stud = self._clean_text(student_answer)
        clean_ref = self._clean_text(question.correct_answer)

        # Handle unanswered
        if not clean_stud:
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer="",
                reference_answer=question.correct_answer,
                marks_awarded=0.0,
                max_marks=question.max_marks,
                evaluation_status="unanswered",
                feedback=f"No answer was provided. Reference explanation: '{question.correct_answer}'.",
                missing_concepts=self._extract_content_tokens(question.correct_answer),
                score_breakdown={"rubric_total": 0.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level
            )

        ref_tokens = self._extract_content_tokens(question.correct_answer)
        stud_tokens = self._extract_content_tokens(student_answer)

        # 1. Concept Coverage (Weight: 40%)
        matched_tokens = [w for w in ref_tokens if w in stud_tokens or any(SequenceMatcher(None, w, s).ratio() > 0.85 for s in stud_tokens)]
        missing = [w for w in ref_tokens if w not in matched_tokens]
        concept_coverage = max(0.0, min(1.0, len(matched_tokens) / max(1, len(ref_tokens))))

        # 2. Semantic Relevance (Weight: 30%)
        sim = self._compute_semantic_similarity(student_answer, question.correct_answer)
        if sim is None:
            # Model unavailable fallback: flag for manual review
            sim = 0.0
            model_missing = True
        else:
            sim = max(0.0, min(1.0, sim))
            model_missing = False

        # 3. Completeness (Weight: 20%)
        # Gated by concept coverage and relevance: irrelevant verbose answers receive 0.0 completeness
        ref_words = len(question.correct_answer.split())
        stud_words = len(student_answer.split())
        if concept_coverage < 0.15 and (sim < 0.40):
            completeness = 0.0
        else:
            completeness = min(1.0, stud_words / max(8, ref_words * 0.70)) * min(1.0, 1.5 * concept_coverage)
        completeness = max(0.0, min(1.0, completeness))

        # 4. Keyword Accuracy (Weight: 10%)
        ref_keywords = [w for w in ref_tokens if len(w) >= 6]
        if ref_keywords:
            kw_matches = sum(1 for kw in ref_keywords if kw in stud_tokens)
            keyword_accuracy = kw_matches / len(ref_keywords)
        else:
            keyword_accuracy = concept_coverage
        keyword_accuracy = max(0.0, min(1.0, keyword_accuracy))

        # Rubric Composite Ratio (0.0 to 1.0)
        rubric_ratio = (
            self.concept_weight * concept_coverage +
            self.semantic_weight * sim +
            self.completeness_weight * completeness +
            self.keyword_weight * keyword_accuracy
        )
        rubric_ratio = max(0.0, min(1.0, rubric_ratio))

        # Irrelevant responses (< 0.25) receive 0.0 marks
        if rubric_ratio < 0.25:
            rubric_ratio = 0.0

        # Calculate marks awarded (strictly bounded between 0.0 and max_marks)
        marks_awarded = round(rubric_ratio * question.max_marks, 2)
        marks_awarded = min(question.max_marks, max(0.0, marks_awarded))

        # Lexical overlap is not treated as proof of factual correctness; uncertain evaluations trigger manual review
        is_discrepancy = (concept_coverage >= 0.50 and sim < 0.35)

        # Determine status
        if model_missing:
            status = "requires_manual_review"
            feedback = "Automated embedding evaluation was unavailable; partial score assigned based on lexical fallback. Flagged for manual faculty review."
        elif is_discrepancy:
            status = "requires_manual_review"
            feedback = "High discrepancy detected: significant keyword overlap but low semantic alignment with reference answer. Flagged for manual faculty review."
        elif rubric_ratio >= 0.75:
            status = "correct"
            feedback = "Comprehensive and well-structured response demonstrating sound conceptual understanding."
        elif rubric_ratio >= 0.35:
            status = "partially_correct"
            missing_str = ", ".join(missing[:4]) if missing else "further technical depth"
            feedback = f"Partially correct answer ({int(rubric_ratio*100)}% rubric score). Missing key components: {missing_str}."
        else:
            status = "incorrect"
            feedback = f"Insufficient explanation. Did not adequately address the core concepts of the reference answer."

        return QuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            question_type=question.question_type,
            student_answer=student_answer,
            reference_answer=question.correct_answer,
            marks_awarded=marks_awarded,
            max_marks=question.max_marks,
            evaluation_status=status,
            feedback=feedback,
            missing_concepts=missing,
            score_breakdown={
                "concept_coverage": round(concept_coverage, 2),
                "semantic_relevance": round(sim, 2),
                "completeness": round(completeness, 2),
                "keyword_accuracy": round(keyword_accuracy, 2),
                "composite_ratio": round(rubric_ratio, 2),
                "lexical_fallback": 1.0 if model_missing else 0.0
            },
            source_reference=question.source_reference,
            topic=question.topic,
            difficulty=question.difficulty,
            bloom_level=question.bloom_level,
            metadata=dict(question.metadata)
        )

    def evaluate_answer(self, question: QuizQuestion, student_answer: str) -> QuestionResult:
        """
        Dispatches evaluation to the appropriate strategy based on question type.
        Safely traps unexpected evaluation errors to prevent silent grading inflation.
        """
        if not question.correct_answer or not question.correct_answer.strip():
            logger.warning(f"Question '{question.question_id}' has missing reference answer.")
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer=student_answer,
                reference_answer="",
                marks_awarded=0.0,
                max_marks=question.max_marks,
                evaluation_status="requires_manual_review",
                feedback="Reference answer was missing for this question. Flagged for manual faculty review.",
                missing_concepts=[],
                score_breakdown={},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )

        q_type = question.question_type.upper()

        try:
            if q_type == "MCQ":
                return self.evaluate_mcq(question, student_answer)
            elif q_type in ["SHORT ANSWER", "SHORT_ANSWER"]:
                return self.evaluate_short_answer(question, student_answer)
            else:
                # Descriptive, Conceptual, Application, Analytical
                return self.evaluate_descriptive(question, student_answer)
        except Exception as e:
            logger.error(f"Unexpected evaluation error for question '{question.question_id}': {e}", exc_info=True)
            return QuestionResult(
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                student_answer=student_answer,
                reference_answer=question.correct_answer,
                marks_awarded=0.0,
                max_marks=question.max_marks,
                evaluation_status="requires_manual_review",
                feedback=f"An unexpected evaluation error occurred ({type(e).__name__}). Flagged for manual faculty review.",
                missing_concepts=[],
                score_breakdown={"error": 1.0},
                source_reference=question.source_reference,
                topic=question.topic,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                metadata=dict(question.metadata)
            )
