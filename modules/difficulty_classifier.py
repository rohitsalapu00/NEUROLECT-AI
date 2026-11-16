"""
NEUROLECT AI - Question Difficulty Classification Engine
Classifies educational questions into: Easy, Medium, Hard.
Analyzes composite features: cognitive depth (Bloom's tier), structural complexity,
reasoning conditionals, and domain technical vocabulary density.
"""

import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger("NEUROLECT.DifficultyClassifier")


class DifficultyClassifier:
    """
    Multi-feature heuristic classifier estimating question difficulty.
    Transparently reports reasoning metrics and confidence limitations.
    """

    BLOOM_WEIGHTS = {
        "Remember": 1.0,
        "Understand": 1.6,
        "Apply": 2.5,
        "Analyze": 3.2,
        "Evaluate": 3.6,
        "Create": 4.0,
    }

    # Complex reasoning conjunctions, comparative markers, and condition indicators
    REASONING_MARKERS = [
        "assuming", "given that", "in contrast to", "trade-off", "provided that",
        "under what conditions", "why would you prefer", "consequence of",
        "whereas", "relationship between", "how would you justify",
        "difference between", "compare", "contrast", "differentiate", "versus"
    ]

    def classify(
        self,
        question_text: str,
        bloom_level: str = "Understand",
        correct_answer: str = ""
    ) -> Tuple[str, float, str]:
        """
        Calculates a multi-feature composite difficulty score.
        Returns: (difficulty: 'Easy'|'Medium'|'Hard', composite_score: float, rationale: str).
        """
        if not question_text or not question_text.strip():
            return "Easy", 1.0, "Empty question text defaulted to Easy."

        clean_q = question_text.strip()
        words = clean_q.split()
        word_count = len(words)

        # Feature 1: Cognitive level weight (Base: 1.0 to 4.0)
        cognitive_score = self.BLOOM_WEIGHTS.get(bloom_level, 1.8)

        # Feature 2: Syntactic & reasoning complexity
        reasoning_score = 0.0
        q_lower = clean_q.lower()
        for marker in self.REASONING_MARKERS:
            if marker in q_lower:
                reasoning_score += 0.8
        reasoning_score = min(2.5, reasoning_score)

        # Feature 3: Structural depth and length
        if word_count < 10:
            length_score = 0.3
        elif word_count <= 22:
            length_score = 0.8
        else:
            length_score = 1.6

        # Feature 4: Technical vocabulary density (words >= 9 characters)
        complex_vocab_count = sum(1 for w in words if len(w) >= 9)
        vocab_score = min(1.5, complex_vocab_count * 0.4)

        # Feature 5: Answer complexity (if provided)
        answer_score = 0.0
        if correct_answer:
            ans_words = len(correct_answer.split())
            if ans_words > 25:
                answer_score = 1.0
            elif ans_words > 12:
                answer_score = 0.5

        # Composite score calculation
        composite = round(cognitive_score + reasoning_score + length_score + vocab_score + answer_score, 2)

        # Threshold calibration
        if composite < 3.2:
            difficulty = "Easy"
        elif composite < 5.4:
            difficulty = "Medium"
        else:
            difficulty = "Hard"

        rationale = (
            f"Difficulty rated '{difficulty}' (composite score {composite:.2f}) based on "
            f"Bloom's weight ({bloom_level}={cognitive_score}), "
            f"reasoning complexity (+{reasoning_score:.1f}), "
            f"length (+{length_score:.1f}), and technical vocab (+{vocab_score:.1f}). "
            f"Note: This is an estimated heuristic score and may vary across curricula."
        )

        return difficulty, composite, rationale
