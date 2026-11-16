"""
NEUROLECT AI - Bloom's Taxonomy Classification Engine
Classifies educational questions into Bloom's six cognitive domains:
1. Remember, 2. Understand, 3. Apply, 4. Analyze, 5. Evaluate, 6. Create.
Uses multi-feature linguistic and contextual pattern matching beyond simple question words.
"""

import re
import logging
from typing import Tuple, Dict, Any
from config import BLOOMS_TAXONOMY

logger = logging.getLogger("NEUROLECT.BloomClassifier")


class BloomClassifier:
    """
    Contextual heuristic classifier for Bloom's Taxonomy cognitive domains.
    Analyzes action verbs, syntactic depth, comparative markers, and operational phrasing.
    """

    # Primary cognitive action indicators
    COGNITIVE_MARKERS = {
        "Remember": [
            "define", "recall", "list", "name", "state", "identify", "recognize",
            "which of the following", "what is the definition", "what term refers to",
            "what does the term", "mention", "duplicate", "label"
        ],
        "Understand": [
            "explain", "describe", "summarize", "interpret", "clarify", "discuss",
            "in your own words", "what is the purpose", "what is the role", "illustrate",
            "what does it mean", "why does", "how does", "paraphrase", "outline"
        ],
        "Apply": [
            "calculate", "solve", "implement", "execute", "apply", "demonstrate",
            "how would you use", "given the scenario", "how would you compute", "utilize",
            "employ", "operate", "determine the output", "modify", "adapt"
        ],
        "Analyze": [
            "compare", "contrast", "differentiate", "distinguish", "trade-off",
            "difference between", "how does x compare", "versus", "in contrast to",
            "break down", "categorize", "examine", "investigate", "dissect", "analyze"
        ],
        "Evaluate": [
            "assess", "critique", "judge", "justify", "defend", "rate", "prioritize",
            "which is better", "evaluate the effectiveness", "most suitable", "validity",
            "recommend", "argue for", "weigh the pros"
        ],
        "Create": [
            "design", "develop", "construct", "formulate", "synthesize", "propose",
            "how would you design", "create an architecture", "invent", "devise",
            "compose", "integrate", "reorganize", "plan a system"
        ]
    }

    # Leading imperative cognitive action verbs
    LEADING_IMPERATIVES = {
        "Create": re.compile(r"^(design|develop|construct|formulate|synthesize|devise)\b", re.IGNORECASE),
        "Evaluate": re.compile(r"^(assess|critique|judge|justify|defend|rate|prioritize)\b", re.IGNORECASE),
        "Analyze": re.compile(r"^(compare|contrast|differentiate|distinguish|analyze|examine|dissect)\b", re.IGNORECASE),
        "Apply": re.compile(r"^(calculate|solve|implement|execute|apply|demonstrate|compute)\b", re.IGNORECASE),
        "Understand": re.compile(r"^(explain|describe|summarize|interpret|clarify|discuss)\b", re.IGNORECASE),
        "Remember": re.compile(r"^(define|recall|list|name|state|identify|label)\b", re.IGNORECASE),
    }

    # Syntactic pattern boosters
    SYNTACTIC_BOOSTERS = {
        "Analyze": re.compile(r"\b(differ|difference|compare|contrast|versus|vs|trade-?off|distinguish between)\b", re.IGNORECASE),
        "Apply": re.compile(r"\b(how would you (use|apply|solve|compute|implement)|calculate the|determine the output)\b", re.IGNORECASE),
        "Evaluate": re.compile(r"\b(justify|critique|most effective|best suited|better suited|and why|assess the validity)\b", re.IGNORECASE),
        "Create": re.compile(r"\b(how would you (design|develop|formulate|architect|construct)|design a (schema|pipeline|system|architecture))\b", re.IGNORECASE),
        "Understand": re.compile(r"\b(why does|what causes|mechanism behind|explain how|what is the purpose)\b", re.IGNORECASE),
        "Remember": re.compile(r"\b(which (of the following|one)|what is the term|define|what is the definition)\b", re.IGNORECASE),
    }

    def classify(self, question_text: str, question_type: str = "") -> Tuple[str, float, str]:
        """
        Classifies a question into a Bloom's Taxonomy level using cognitive task analysis.
        Returns: (bloom_level, heuristic_confidence_score, rationale).
        """
        if not question_text or not question_text.strip():
            return "Remember", 0.50, "Empty question text defaulted to Remember (heuristic fallback)."

        text_lower = question_text.lower().strip()
        scores: Dict[str, float] = {level: 0.1 for level in BLOOMS_TAXONOMY}

        # 1. Leading imperative action verb boost (strongest indicator of intended cognitive demand)
        for level, pattern in self.LEADING_IMPERATIVES.items():
            if pattern.search(text_lower):
                scores[level] += 3.5

        # 2. Action verb and phrase pattern matching
        for level, markers in self.COGNITIVE_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    # Multi-word markers carry stronger weight than single common words
                    weight = 2.5 if " " in marker else 1.2
                    scores[level] += weight

        # 3. Syntactic construction boosters
        for level, pattern in self.SYNTACTIC_BOOSTERS.items():
            if pattern.search(text_lower):
                scores[level] += 2.0

        # 4. Question type contextual prior (soft prior; does not override explicit verbs)
        q_type_lower = question_type.lower()
        if "analytical" in q_type_lower:
            scores["Analyze"] += 0.8
        elif "application" in q_type_lower:
            scores["Apply"] += 0.8
        elif "conceptual" in q_type_lower:
            scores["Understand"] += 0.8
        elif "descriptive" in q_type_lower:
            scores["Understand"] += 0.6
        elif "mcq" in q_type_lower and scores["Remember"] > 1.0:
            scores["Remember"] += 0.5

        # Determine winner
        best_level = max(scores, key=scores.get)
        total_score = sum(scores.values())
        max_score = scores[best_level]

        # Normalized heuristic match score (not a statistical posterior probability)
        confidence = round(min(0.95, max(0.55, max_score / (total_score * 0.45))), 2)

        rationale = (
            f"Classified as '{best_level}' (heuristic match score: {confidence:.2f}) based on "
            f"cognitive action verbs and syntactic phrasing. "
            f"Note: Heuristic rule-based estimate; not an empirically calibrated posterior probability."
        )

        return best_level, confidence, rationale
