"""
NEUROLECT AI - Question Data Models
Defines the standardized GeneratedQuestion dataclass supporting 6 question types,
Bloom's Taxonomy, difficulty tiers, and source provenance tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime


@dataclass
class GeneratedQuestion:
    """
    Standardized question representation for NEUROLECT AI.
    Contains full assessment metadata, Bloom's level, difficulty, and source citations.
    """
    question_id: str
    question_text: str
    question_type: str                         # 'MCQ', 'Short Answer', 'Descriptive', 'Conceptual', 'Application', 'Analytical'
    options: List[str]                        # For MCQs: exactly 4 options; empty list for open-ended
    correct_answer: str                        # Ground-truth answer supported by source
    explanation: str                          # Conceptual rationale for the answer
    topic: str                                # Associated lecture topic
    subtopic: str                             # Associated subtopic
    bloom_level: str                          # 'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'
    difficulty: str                           # 'Easy', 'Medium', 'Hard'
    source_reference: str                     # Source page (e.g. "Page 2") or timestamp (e.g. "[01:15 - 01:45]")
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes question instance to dictionary for storage and UI."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedQuestion":
        """Reconstructs GeneratedQuestion from dictionary."""
        return cls(
            question_id=data.get("question_id", str(uuid.uuid4())[:8]),
            question_text=data.get("question_text", ""),
            question_type=data.get("question_type", "Short Answer"),
            options=data.get("options", []),
            correct_answer=data.get("correct_answer", ""),
            explanation=data.get("explanation", ""),
            topic=data.get("topic", "General"),
            subtopic=data.get("subtopic", "General"),
            bloom_level=data.get("bloom_level", "Understand"),
            difficulty=data.get("difficulty", "Medium"),
            source_reference=data.get("source_reference", "General"),
            generation_metadata=data.get("generation_metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
