"""
NEUROLECT AI - MCQ Distractor Generation Module
Generates exactly three plausible, unambiguously incorrect distractor options for MCQs.
Combines domain concept contrast, model assistance, and strict deduplication.
Stores correct answers separately from randomized option lists.
"""

import re
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
from difflib import SequenceMatcher
from utils.helpers import clean_text, split_into_sentences

logger = logging.getLogger("NEUROLECT.DistractorGenerator")


class DistractorGenerationError(Exception):
    """Raised when distractor generation fails or produces invalid options."""
    pass


class DistractorGenerator:
    """
    Generates and validates plausible multiple-choice distractors for educational assessments.
    Uses lecture domain context, semantic contrast, and string distance filtering.
    """

    # Domain vocabulary antonyms with word boundaries to avoid partial-word replacement
    TECHNICAL_ANTONYMS = {
        "supervised": "unsupervised",
        "unsupervised": "supervised",
        "labeled": "unlabeled",
        "unlabeled": "labeled",
        "linear": "non-linear",
        "non-linear": "linear",
        "discrete": "continuous",
        "continuous": "discrete",
        "convex": "non-convex",
        "non-convex": "convex",
        "deterministic": "stochastic",
        "stochastic": "deterministic",
        "probabilistic": "deterministic",
        "increase": "decrease",
        "decrease": "increase",
        "increases": "decreases",
        "decreases": "increases",
        "maximize": "minimize",
        "minimize": "maximize",
        "maximizes": "minimizes",
        "minimizes": "maximizes",
        "maximum": "minimum",
        "minimum": "maximum",
        "overfitting": "underfitting",
        "underfitting": "overfitting",
        "shallow": "deep",
        "deep": "shallow",
        "symmetric": "asymmetric",
        "asymmetric": "symmetric",
        "synchronous": "asynchronous",
        "asynchronous": "synchronous",
        "positive": "negative",
        "negative": "positive",
        "empirical": "theoretical",
        "theoretical": "empirical",
        "local": "global",
        "global": "local",
        "static": "dynamic",
        "dynamic": "static",
        "loss": "accuracy",
        "cost": "reward",
        "sparsity": "density"
    }

    def __init__(self):
        pass

    def _is_word_subphrase(self, a_words: List[str], b_words: List[str]) -> bool:
        """Checks if sequence of words in list a is an exact consecutive sublist of b."""
        if not a_words or not b_words or len(a_words) >= len(b_words):
            return False
        for i in range(len(b_words) - len(a_words) + 1):
            if b_words[i:i+len(a_words)] == a_words:
                return True
        return False

    def _similarity(self, a: str, b: str) -> float:
        """Computes normalized string similarity ratio."""
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    def _is_competing_correct_answer(
        self,
        candidate: str,
        question_text: str,
        context: str
    ) -> bool:
        """
        Checks if candidate distractor is also factually supported as a correct answer
        to question_text in the source context (avoiding multiple defensible answers).
        """
        if not candidate or not context or not question_text:
            return False

        c_lower = candidate.lower().strip()
        stop = {"what", "which", "where", "when", "does", "from", "with", "that", "this", "have", "been", "were", "explain", "describe", "state"}
        c_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', c_lower)) - stop
        q_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question_text.lower())) - stop

        if not c_words or not q_words:
            return False

        sentences = split_into_sentences(clean_text(context))
        q_sentences = [s for s in sentences if any(w in s.lower() for w in q_words)]

        for s in q_sentences:
            s_lower = s.lower()
            matches = [w for w in c_words if w in s_lower]
            if len(matches) >= max(1, len(c_words) // 2):
                # Candidate appears in the grounding sentence; check if presented affirmatively without contrast
                if " whereas " not in s_lower and " in contrast " not in s_lower and " rather than " not in s_lower and " unlike " not in s_lower:
                    return True

        return False

    def _is_structurally_compatible(self, candidate: str, target: str) -> bool:
        """
        Verifies grammatical/length compatibility between candidate distractor and target answer.
        Prevents mixing short noun phrases with long multi-sentence explanations.
        """
        c_words = len(candidate.split())
        t_words = len(target.split())
        if t_words <= 4:
            return 1 <= c_words <= 8
        elif t_words <= 12:
            return 3 <= c_words <= 18
        else:
            return c_words >= 5

    def _clean_option(self, text: str, target_ends_with_period: bool) -> str:
        """Standardizes option formatting and punctuation."""
        t = text.strip()
        if not target_ends_with_period:
            t = t.rstrip(". ")
        elif not t.endswith("."):
            t += "."
        # Capitalize first letter if target is capitalized
        if t and t[0].isalpha():
            t = t[0].upper() + t[1:]
        return t

    def generate_distractors(
        self,
        question_text: str,
        correct_answer: str,
        context: str,
        domain_keywords: Optional[List[str]] = None,
        all_concepts: Optional[List[str]] = None
    ) -> Tuple[List[str], bool]:
        """
        Generates exactly 3 distinct, plausible distractors for the given question and correct answer.
        Returns: (distractors: List[str], is_validated: bool).
        """
        clean_ans = clean_text(correct_answer).strip()
        if not clean_ans:
            raise DistractorGenerationError("Correct answer cannot be empty.")

        target_ends_with_period = clean_ans.endswith(".")
        clean_ans_norm = self._clean_option(clean_ans, target_ends_with_period)
        ans_words_count = len(clean_ans_norm.split())

        candidates: List[str] = []

        # 1. Antonym and Contrast Variations (High Plausibility)
        contrasts = self._generate_contrastive_fallbacks(clean_ans_norm, question_text)
        contrast_set = {c.lower().strip() for c in contrasts}
        candidates.extend(contrasts)

        # 2. Candidate Pool from KeyBERT Domain Terms
        if domain_keywords:
            for kw in domain_keywords:
                kw_clean = clean_text(kw).strip()
                if kw_clean and len(kw_clean) > 2:
                    candidates.append(kw_clean)

        # 3. Candidate Pool from Technical Concepts
        if all_concepts:
            for c in all_concepts:
                c_clean = clean_text(c).strip()
                if c_clean:
                    candidates.append(c_clean)

        # 4. Contextual extraction (sentences and phrases)
        if context:
            sentences = split_into_sentences(clean_text(context))
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                candidates.append(s_clean)
                # If target is a short phrase, extract sub-clauses and phrases
                if ans_words_count <= 6:
                    sub_parts = re.split(r'[,;]|\band\b|\bor\b|\bwhereas\b|\bwhile\b', s_clean, flags=re.IGNORECASE)
                    for sp in sub_parts:
                        sp_clean = sp.strip()
                        if 1 <= len(sp_clean.split()) <= 6:
                            candidates.append(sp_clean)

        # 5. Strict Filtering against correct answer and duplicates
        valid_distractors: List[str] = []
        seen = {clean_ans_norm.lower()}

        for cand in candidates:
            cand_norm = self._clean_option(cand, target_ends_with_period)
            cand_lower = cand_norm.lower().strip()

            if not cand_lower or cand_lower in seen:
                continue
            # Must not be identical or too similar to correct answer (unless validated antonym)
            if cand_lower not in contrast_set and self._similarity(cand_norm, clean_ans_norm) > 0.65:
                continue
            # Must not be a consecutive word subphrase of correct answer or vice versa
            cw = cand_lower.split()
            aw = clean_ans_norm.lower().split()
            if self._is_word_subphrase(cw, aw) or self._is_word_subphrase(aw, cw):
                continue
            # Check structural / length compatibility
            if not self._is_structurally_compatible(cand_norm, clean_ans_norm):
                continue
            # Must not be a competing valid/defensible answer to the question
            if self._is_competing_correct_answer(cand_norm, question_text, context):
                continue
            # Must not be too similar to existing selected distractors or share subphrases
            if any(self._similarity(cand_norm, existing) > 0.65 for existing in valid_distractors):
                continue
            if any(self._is_word_subphrase(cw, ex.lower().split()) or self._is_word_subphrase(ex.lower().split(), cw) for ex in valid_distractors):
                continue

            valid_distractors.append(cand_norm)
            seen.add(cand_lower)
            if len(valid_distractors) == 3:
                break

        # 6. Fallback generation if pool has fewer than 3 valid options
        if len(valid_distractors) < 3:
            # Generate grounded contextual contrasts based on question text
            q_words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', question_text) if w.lower() not in {"what", "which", "where", "does", "from", "with", "that"}]
            subject = q_words[0].title() if q_words else "Model Parameter"

            if ans_words_count <= 4:
                contextual_short_contrasts = [
                    f"Stochastic {subject} Estimation",
                    f"Unconstrained {subject} Variance",
                    f"Deterministic {subject} Heuristic",
                    f"Static {subject} Partition"
                ]
                for sc in contextual_short_contrasts:
                    sc_norm = self._clean_option(sc, target_ends_with_period)
                    if sc_norm.lower() not in seen:
                        valid_distractors.append(sc_norm)
                        seen.add(sc_norm.lower())
                    if len(valid_distractors) == 3:
                        break
            else:
                contextual_long_contrasts = [
                    f"Applies empirical heuristics without updating {subject.lower()} configurations",
                    f"Evaluates static baseline distributions rather than active training dynamics",
                    f"Restricts parameter optimizations to unconstrained local gradient approximations"
                ]
                for lc in contextual_long_contrasts:
                    lc_norm = self._clean_option(lc, target_ends_with_period)
                    if lc_norm.lower() not in seen:
                        valid_distractors.append(lc_norm)
                        seen.add(lc_norm.lower())
                    if len(valid_distractors) == 3:
                        break

        selected_distractors = valid_distractors[:3]
        is_validated = len(selected_distractors) == 3 and all(d.lower() != clean_ans_norm.lower() for d in selected_distractors)

        return selected_distractors, is_validated

    def build_shuffled_mcq(
        self,
        question_text: str,
        correct_answer: str,
        context: str,
        domain_keywords: Optional[List[str]] = None,
        all_concepts: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> Tuple[List[str], str]:
        """
        Builds exactly 4 MCQ options (1 correct answer + 3 distractors), randomly shuffled.
        Returns: (shuffled_options, correct_answer).
        """
        distractors, _ = self.generate_distractors(
            question_text=question_text,
            correct_answer=correct_answer,
            context=context,
            domain_keywords=domain_keywords,
            all_concepts=all_concepts
        )

        clean_ans = clean_text(correct_answer).strip()
        target_ends_with_period = clean_ans.endswith(".")
        clean_ans_norm = self._clean_option(clean_ans, target_ends_with_period)

        options = [clean_ans_norm] + [self._clean_option(d, target_ends_with_period) for d in distractors]
        
        # Shuffle deterministically if seed provided, else randomly
        rng = random.Random(seed) if seed is not None else random.Random()
        rng.shuffle(options)

        return options, clean_ans_norm

    def _generate_contrastive_fallbacks(self, answer: str, question: str) -> List[str]:
        """
        Generates plausible syntactic/semantic contrast options using technical antonyms.
        """
        fallbacks = []
        ans_lower = answer.lower()

        for word, antonym in self.TECHNICAL_ANTONYMS.items():
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            if pattern.search(answer):
                def _replace(match):
                    m = match.group(0)
                    if m.isupper():
                        return antonym.upper()
                    elif m[0].isupper():
                        return antonym.capitalize()
                    return antonym.lower()
                contrast = pattern.sub(_replace, answer)
                if contrast.lower() != ans_lower:
                    fallbacks.append(contrast)

        return fallbacks
