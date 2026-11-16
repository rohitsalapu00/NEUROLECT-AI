"""
NEUROLECT AI - Quiz Data Models
Defines standardized data models for quizzes, questions, student submissions,
evaluation outcomes, and comprehensive performance analytics.
Preserves full question provenance and source citations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from modules.question_models import GeneratedQuestion


@dataclass
class QuizQuestion:
    """
    Represents an individual question in an active quiz session.
    Carries complete source provenance, Bloom's level, difficulty, and scoring metadata.
    """
    question_id: str
    question_text: str
    question_type: str                         # 'MCQ', 'Short Answer', 'Descriptive', 'Conceptual', 'Application', 'Analytical'
    options: List[str]                        # For MCQs: 4 options; empty for open-ended questions
    correct_answer: str                        # Ground-truth reference answer supported by lecture source
    explanation: str                          # Conceptual rationale
    topic: str                                # Associated lecture topic
    subtopic: str                             # Associated subtopic
    difficulty: str                           # 'Easy', 'Medium', 'Hard'
    bloom_level: str                          # 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'
    source_reference: str                     # Page number (e.g. "Page 2") or timestamp (e.g. "[01:30 - 02:15]")
    max_marks: float = 1.0                    # Maximum possible marks for this question
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def segment_id(self) -> Optional[int]:
        """Convenience property extracting segment_id from metadata."""
        return self.metadata.get("segment_id")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes QuizQuestion to dictionary."""
        d = asdict(self)
        d["segment_id"] = self.segment_id
        return d

    @classmethod
    def from_generated_question(
        cls,
        gq: GeneratedQuestion,
        max_marks: Optional[float] = None
    ) -> "QuizQuestion":
        """Factory constructor converting a Phase 4 GeneratedQuestion into a QuizQuestion."""
        default_marks = {
            "MCQ": 1.0,
            "Short Answer": 2.0,
            "Conceptual": 3.0,
            "Descriptive": 5.0,
            "Application": 5.0,
            "Analytical": 5.0,
        }
        marks = max_marks if max_marks is not None else default_marks.get(gq.question_type, 1.0)
        return cls(
            question_id=gq.question_id,
            question_text=gq.question_text,
            question_type=gq.question_type,
            options=list(gq.options),
            correct_answer=gq.correct_answer,
            explanation=gq.explanation,
            topic=gq.topic,
            subtopic=gq.subtopic,
            difficulty=gq.difficulty,
            bloom_level=gq.bloom_level,
            source_reference=gq.source_reference,
            max_marks=float(marks),
            metadata=dict(gq.generation_metadata)
        )


@dataclass
class StudentAnswer:
    """
    Represents an answer submitted by a student for a specific quiz question.
    """
    question_id: str
    answer_text: str
    is_skipped: bool = False
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionResult:
    """
    Evaluation outcome and feedback for an individual question response.
    """
    question_id: str
    question_text: str
    question_type: str
    student_answer: str
    reference_answer: str
    marks_awarded: float
    max_marks: float
    evaluation_status: str                     # 'correct', 'partially_correct', 'incorrect', 'unanswered', 'requires_manual_review'
    feedback: str                             # Constructive pedagogical explanation
    missing_concepts: List[str]               # Important terms/concepts not mentioned in student's response
    score_breakdown: Dict[str, float]         # Rubric components: concept_coverage, semantic_relevance, etc.
    source_reference: str                     # Verifiable provenance citation
    topic: str
    difficulty: str
    bloom_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def segment_id(self) -> Optional[int]:
        """Convenience property extracting segment_id from metadata."""
        return self.metadata.get("segment_id")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["segment_id"] = self.segment_id
        return d


@dataclass
class Quiz:
    """
    Represents a synthesized quiz ready for interactive administration.
    """
    quiz_id: str
    title: str
    questions: List[QuizQuestion]
    total_marks: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quiz_id": self.quiz_id,
            "title": self.title,
            "total_marks": self.total_marks,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "questions": [q.to_dict() for q in self.questions]
        }

    def get_question(self, question_id: str) -> Optional[QuizQuestion]:
        """Retrieves a question by its unique identifier."""
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None


@dataclass
class QuizResult:
    """
    Comprehensive performance evaluation and analytics for a completed quiz session.
    """
    quiz_id: str
    quiz_title: str
    total_questions: int
    answered_questions: int
    unanswered_questions: int
    total_marks: float
    earned_marks: float
    percentage: float
    correct_count: int
    partial_count: int
    incorrect_count: int
    topic_scores: Dict[str, Dict[str, float]]          # e.g. {'Topic': {'earned': 4.0, 'total': 5.0, 'percentage': 80.0}}
    difficulty_scores: Dict[str, Dict[str, float]]     # e.g. {'Easy': {'earned': 2.0, 'total': 2.0, 'percentage': 100.0}}
    bloom_scores: Dict[str, Dict[str, float]]          # e.g. {'Understand': {'earned': 3.0, 'total': 5.0, 'percentage': 60.0}}
    question_results: List[QuestionResult]
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quiz_id": self.quiz_id,
            "quiz_title": self.quiz_title,
            "total_questions": self.total_questions,
            "answered_questions": self.answered_questions,
            "unanswered_questions": self.unanswered_questions,
            "total_marks": self.total_marks,
            "earned_marks": self.earned_marks,
            "percentage": self.percentage,
            "correct_count": self.correct_count,
            "partial_count": self.partial_count,
            "incorrect_count": self.incorrect_count,
            "topic_scores": self.topic_scores,
            "difficulty_scores": self.difficulty_scores,
            "bloom_scores": self.bloom_scores,
            "completed_at": self.completed_at,
            "is_complete": self.is_complete,
            "question_results": [r.to_dict() for r in self.question_results]
        }
