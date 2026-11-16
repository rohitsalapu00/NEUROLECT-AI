"""
NEUROLECT AI - Quiz Execution & Assessment Engine
Manages quiz generation, filtering, interactive state tracking, scoring,
and comprehensive performance analytics from Phase 4 GeneratedQuestion objects.
Guarantees deduplication, strict score bounding, and topic/difficulty breakdown.
Completely RAG-free.
"""

import uuid
import random
import logging
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from modules.question_models import GeneratedQuestion
from modules.quiz_models import (
    Quiz,
    QuizQuestion,
    StudentAnswer,
    QuestionResult,
    QuizResult
)
from modules.answer_evaluator import AnswerEvaluator

logger = logging.getLogger("NEUROLECT.QuizEngine")


class QuizEngineError(Exception):
    """Raised when quiz creation, state management, or evaluation fails."""
    pass


class QuizEngine:
    """
    Orchestrates quiz session lifecycle: creation, filtering, deduplication,
    answer recording, evaluation dispatch, and multi-dimensional analytics.
    """

    def __init__(self, evaluator: Optional[AnswerEvaluator] = None):
        self.evaluator = evaluator or AnswerEvaluator()
        self.active_quiz: Optional[Quiz] = None
        self.student_answers: Dict[str, StudentAnswer] = {}
        self.current_question_idx: int = 0
        self.completed_quiz_ids: set = set()

    def create_quiz(
        self,
        questions: List[GeneratedQuestion],
        title: str = "NEUROLECT Educational Assessment",
        num_questions: Optional[int] = None,
        topic_filter: Optional[List[str]] = None,
        type_filter: Optional[List[str]] = None,
        difficulty_filter: Optional[List[str]] = None,
        bloom_filter: Optional[List[str]] = None,
        randomize_questions: bool = True,
        randomize_options: bool = True,
        seed: Optional[int] = None,
        marks_per_type: Optional[Dict[str, float]] = None
    ) -> Quiz:
        """
        Creates a structured Quiz instance from existing GeneratedQuestion objects.
        Applies filtering, deduplication, and deterministic randomization if requested.
        """
        if not questions:
            raise QuizEngineError("Cannot create quiz: question candidate list is empty.")

        filtered: List[GeneratedQuestion] = []
        seen_ids = set()
        seen_texts: List[str] = []

        # 1. Filter and Deduplicate
        for q in questions:
            # ID deduplication
            if q.question_id in seen_ids:
                continue

            # Near-identical text deduplication (ratio >= 0.75 or word containment >= 0.75)
            q_clean = q.question_text.lower().strip()
            q_words = set(q_clean.split())
            is_dup = any(
                SequenceMatcher(None, q_clean, st).ratio() >= 0.75 or
                (len(q_words & set(st.split())) / max(1, min(len(q_words), len(st.split()))) >= 0.75)
                for st in seen_texts
            )
            if is_dup:
                continue

            # Topic Filter
            if topic_filter and q.topic not in topic_filter:
                continue

            # Question Type Filter
            if type_filter and q.question_type not in type_filter:
                continue

            # Difficulty Filter
            if difficulty_filter and q.difficulty not in difficulty_filter:
                continue

            # Bloom's Filter
            if bloom_filter and q.bloom_level not in bloom_filter:
                continue

            filtered.append(q)
            seen_ids.add(q.question_id)
            seen_texts.append(q_clean)

        if not filtered:
            raise QuizEngineError("No questions matched the specified filters.")

        # 2. Randomize question order if requested
        rng = random.Random(seed) if seed is not None else random.Random()
        if randomize_questions:
            rng.shuffle(filtered)

        # 3. Limit count
        if num_questions is not None and num_questions > 0:
            filtered = filtered[:num_questions]

        # 4. Convert to QuizQuestion instances and randomize MCQ options
        quiz_questions: List[QuizQuestion] = []
        total_quiz_marks = 0.0

        for idx, gq in enumerate(filtered):
            # Determine marks
            if marks_per_type and gq.question_type in marks_per_type:
                max_marks = marks_per_type[gq.question_type]
            else:
                max_marks = None  # Use model default

            qq = QuizQuestion.from_generated_question(gq, max_marks=max_marks)

            # Randomize MCQ options if requested
            if randomize_options and qq.question_type == "MCQ" and qq.options:
                opt_copy = list(qq.options)
                rng.shuffle(opt_copy)
                qq.options = opt_copy

            quiz_questions.append(qq)
            total_quiz_marks += qq.max_marks

        quiz = Quiz(
            quiz_id=f"quiz_{uuid.uuid4().hex[:8]}",
            title=title,
            questions=quiz_questions,
            total_marks=round(total_quiz_marks, 2),
            metadata={
                "topic_filter": topic_filter,
                "type_filter": type_filter,
                "difficulty_filter": difficulty_filter,
                "question_count": len(quiz_questions),
                "seed": seed
            }
        )

        logger.info(
            f"Created quiz '{quiz.title}' with {len(quiz.questions)} questions "
            f"(Total marks: {quiz.total_marks})."
        )
        return quiz

    def start_quiz(self, quiz: Quiz) -> None:
        """Initializes a new interactive quiz session."""
        if quiz.quiz_id in self.completed_quiz_ids:
            raise QuizEngineError(f"Quiz '{quiz.quiz_id}' has already been submitted and finalized. Cannot resubmit.")
        self.active_quiz = quiz
        self.student_answers = {}
        self.current_question_idx = 0
        logger.info(f"Quiz '{quiz.quiz_id}' session started.")

    def submit_answer(self, question_id: str, answer_text: str) -> None:
        """Records a student's answer for a question in the active quiz."""
        if not self.active_quiz:
            raise QuizEngineError("No active quiz session. Call start_quiz() first.")

        q = self.active_quiz.get_question(question_id)
        if not q:
            raise QuizEngineError(f"Question '{question_id}' not found in active quiz.")

        clean_ans = answer_text.strip() if answer_text else ""
        self.student_answers[question_id] = StudentAnswer(
            question_id=question_id,
            answer_text=clean_ans,
            is_skipped=len(clean_ans) == 0
        )

    def skip_question(self, question_id: str) -> None:
        """Explicitly records a question as skipped."""
        if not self.active_quiz:
            raise QuizEngineError("No active quiz session. Call start_quiz() first.")

        self.student_answers[question_id] = StudentAnswer(
            question_id=question_id,
            answer_text="",
            is_skipped=True
        )

    def get_quiz_progress(self) -> Dict[str, Any]:
        """Returns the current state and completion statistics of the active quiz."""
        if not self.active_quiz:
            return {"active": False}

        total = len(self.active_quiz.questions)
        answered = sum(1 for sa in self.student_answers.values() if not sa.is_skipped and len(sa.answer_text) > 0)
        skipped = sum(1 for sa in self.student_answers.values() if sa.is_skipped)
        unanswered = total - len(self.student_answers)

        return {
            "active": True,
            "quiz_id": self.active_quiz.quiz_id,
            "total_questions": total,
            "answered_count": answered,
            "skipped_count": skipped,
            "unanswered_count": unanswered,
            "current_index": self.current_question_idx,
            "is_complete": len(self.student_answers) == total
        }

    def finish_quiz(
        self,
        evaluator: Optional[AnswerEvaluator] = None
    ) -> QuizResult:
        """
        Evaluates all submitted answers, calculates marks, builds topic and difficulty
        breakdowns, and generates the final QuizResult.
        """
        if not self.active_quiz:
            raise QuizEngineError("Cannot finish quiz: no active quiz session.")

        active_evaluator = evaluator or self.evaluator
        question_results: List[QuestionResult] = []

        earned_marks = 0.0
        correct_count = 0
        partial_count = 0
        incorrect_count = 0
        unanswered_count = 0

        # Topic, difficulty, and bloom trackers
        # Structure: {'category': {'earned': 0.0, 'total': 0.0, 'count': 0, 'unanswered': 0}}
        topic_tracker: Dict[str, Dict[str, float]] = {}
        diff_tracker: Dict[str, Dict[str, float]] = {}
        bloom_tracker: Dict[str, Dict[str, float]] = {}

        for q in self.active_quiz.questions:
            # Initialize trackers
            if q.topic not in topic_tracker:
                topic_tracker[q.topic] = {"earned": 0.0, "total": 0.0, "count": 0, "unanswered": 0}
            if q.difficulty not in diff_tracker:
                diff_tracker[q.difficulty] = {"earned": 0.0, "total": 0.0, "count": 0, "unanswered": 0}
            if q.bloom_level not in bloom_tracker:
                bloom_tracker[q.bloom_level] = {"earned": 0.0, "total": 0.0, "count": 0, "unanswered": 0}

            topic_tracker[q.topic]["total"] += q.max_marks
            topic_tracker[q.topic]["count"] += 1
            diff_tracker[q.difficulty]["total"] += q.max_marks
            diff_tracker[q.difficulty]["count"] += 1
            bloom_tracker[q.bloom_level]["total"] += q.max_marks
            bloom_tracker[q.bloom_level]["count"] += 1

            # Retrieve student submission
            sub = self.student_answers.get(q.question_id)
            if sub is None or sub.is_skipped or not sub.answer_text.strip():
                # Unanswered Question
                res = active_evaluator.evaluate_answer(q, "")
                unanswered_count += 1
                topic_tracker[q.topic]["unanswered"] += 1
                diff_tracker[q.difficulty]["unanswered"] += 1
                bloom_tracker[q.bloom_level]["unanswered"] += 1
            else:
                # Answered Question
                res = active_evaluator.evaluate_answer(q, sub.answer_text)
                if res.evaluation_status == "correct":
                    correct_count += 1
                elif res.evaluation_status == "partially_correct":
                    partial_count += 1
                else:
                    incorrect_count += 1

            # Strict bounds checking: 0.0 <= marks <= max_marks
            bounded_marks = max(0.0, min(q.max_marks, res.marks_awarded))
            res.marks_awarded = bounded_marks

            earned_marks += bounded_marks
            topic_tracker[q.topic]["earned"] += bounded_marks
            diff_tracker[q.difficulty]["earned"] += bounded_marks
            bloom_tracker[q.bloom_level]["earned"] += bounded_marks

            question_results.append(res)

        total_marks = self.active_quiz.total_marks
        earned_marks = round(max(0.0, min(total_marks, earned_marks)), 2)
        percentage = round((earned_marks / total_marks * 100.0) if total_marks > 0 else 0.0, 1)

        # Compute percentages for breakdowns
        def _calc_percentages(tracker: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
            out = {}
            for k, v in tracker.items():
                tot = round(v["total"], 2)
                ear = round(v["earned"], 2)
                pct = round((ear / tot * 100.0) if tot > 0 else 0.0, 1)
                out[k] = {
                    "total_marks": tot,
                    "earned_marks": ear,
                    "percentage": pct,
                    "question_count": int(v["count"]),
                    "unanswered_count": int(v.get("unanswered", 0))
                }
            return out

        topic_scores = _calc_percentages(topic_tracker)
        difficulty_scores = _calc_percentages(diff_tracker)
        bloom_scores = _calc_percentages(bloom_tracker)

        quiz_result = QuizResult(
            quiz_id=self.active_quiz.quiz_id,
            quiz_title=self.active_quiz.title,
            total_questions=len(self.active_quiz.questions),
            answered_questions=len(self.active_quiz.questions) - unanswered_count,
            unanswered_questions=unanswered_count,
            total_marks=total_marks,
            earned_marks=earned_marks,
            percentage=percentage,
            correct_count=correct_count,
            partial_count=partial_count,
            incorrect_count=incorrect_count,
            topic_scores=topic_scores,
            difficulty_scores=difficulty_scores,
            bloom_scores=bloom_scores,
            question_results=question_results,
            is_complete=(unanswered_count == 0)
        )

        logger.info(
            f"Finished quiz '{quiz_result.quiz_id}': Earned {earned_marks}/{total_marks} "
            f"({percentage}%). Correct: {correct_count}, Partial: {partial_count}, Incorrect: {incorrect_count}."
        )

        # Mark quiz finalized and clear active session
        self.completed_quiz_ids.add(self.active_quiz.quiz_id)
        self.active_quiz = None
        self.student_answers = {}
        self.current_question_idx = 0

        return quiz_result
