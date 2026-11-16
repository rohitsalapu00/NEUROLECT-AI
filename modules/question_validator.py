"""
NEUROLECT AI - Question Quality Validation Engine
Performs multi-criteria quality control on generated questions:
- Detects duplicate questions and near-identical paraphrases.
- Validates MCQ integrity (exactly 4 options, no duplicate options, correct answer present).
- Verifies source grounding without vector retrieval (direct lexical & concept checking).
- Flags malformed outputs, empty fields, and unsupported question types.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from modules.question_models import GeneratedQuestion
from modules.content_models import LectureContent
from config import QUESTION_TYPES

logger = logging.getLogger("NEUROLECT.QuestionValidator")


class ValidationResult:
    """Encapsulates the validation outcome for a single question or question suite."""
    def __init__(self, is_valid: bool, issues: List[str], warnings: List[str], quality_score: float):
        self.is_valid = is_valid
        self.issues = issues          # Blocking defects
        self.warnings = warnings      # Non-blocking advisories
        self.quality_score = quality_score  # 0.0 to 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "quality_score": self.quality_score
        }


class QuestionValidator:
    """
    Validates question structural integrity, distractor quality, and source grounding.
    Uses direct sequence matching and lexical cross-referencing without RAG or vector retrieval.
    """

    def check_mcq_multiple_defensible_answers(
        self,
        question_text: str,
        correct_answer: str,
        options: List[str],
        source_text: str
    ) -> Tuple[bool, List[str]]:
        """
        Verifies whether any distractor in the MCQ is also factually supported as a correct answer
        to question_text based on the source text (detecting ambiguous MCQs with multiple defensible options).
        """
        if not question_text or not options or not source_text:
            return False, []

        stop = {"what", "which", "where", "when", "does", "from", "with", "that", "this", "have", "been", "were", "explain", "describe", "state", "model", "system"}
        q_focus_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question_text.lower())) - stop
        if not q_focus_words:
            return False, []

        sentences = re.split(r'(?<=[.?!])\s+', source_text)
        grounding_sentences = [s for s in sentences if any(w in s.lower() for w in q_focus_words)]

        ambiguous_distractors = []
        ans_lower = correct_answer.lower().strip()
        distractors = [opt for opt in options if opt.lower().strip() != ans_lower]

        for dist in distractors:
            d_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', dist.lower())) - stop
            if not d_words:
                continue
            for s in grounding_sentences:
                s_lower = s.lower()
                matches = [w for w in d_words if w in s_lower]
                if len(matches) >= max(1, len(d_words) // 2):
                    if " whereas " not in s_lower and " in contrast " not in s_lower and " rather than " not in s_lower and " unlike " not in s_lower:
                        ambiguous_distractors.append(dist)
                        break

        return len(ambiguous_distractors) > 0, ambiguous_distractors

    def validate_question(
        self,
        question: GeneratedQuestion,
        source_content: Optional[LectureContent] = None
    ) -> ValidationResult:
        """
        Validates an individual GeneratedQuestion object.
        """
        issues: List[str] = []
        warnings: List[str] = []
        quality_score = 100.0

        # 1. Non-empty and substantive check
        if not question.question_text or not question.question_text.strip():
            issues.append("Question text is empty.")
            quality_score -= 50.0
        elif len(question.question_text.strip().split()) < 4 or len(question.question_text.strip()) < 12:
            issues.append("Question text is too brief to form a substantive assessment item.")
            quality_score -= 40.0

        if not question.correct_answer or not question.correct_answer.strip():
            issues.append("Correct answer is empty or missing.")
            quality_score -= 40.0
        elif len(question.correct_answer.strip()) < 2:
            issues.append("Correct answer is too short to be substantive.")
            quality_score -= 30.0

        # 2. Source reference provenance check
        if not question.source_reference or not question.source_reference.strip():
            warnings.append("Question is missing source reference provenance (page/timestamp).")
            quality_score -= 10.0

        # 3. Valid question type check
        if question.question_type not in QUESTION_TYPES:
            issues.append(f"Invalid question type '{question.question_type}'. Must be one of {QUESTION_TYPES}")
            quality_score -= 30.0

        # 4. Punctuation sanity
        if question.question_text and not question.question_text.strip().endswith("?"):
            warnings.append("Question does not end with a question mark.")
            quality_score -= 5.0

        # 5. MCQ-specific validation
        if question.question_type == "MCQ":
            if not question.options or len(question.options) != 4:
                issues.append(f"MCQ must contain exactly 4 options. Found {len(question.options)} options.")
                quality_score -= 35.0
            else:
                # Check for duplicate options
                option_lowers = [opt.strip().lower() for opt in question.options]
                if len(set(option_lowers)) < len(option_lowers):
                    issues.append("MCQ contains duplicate answer options.")
                    quality_score -= 30.0

                # Check if correct answer is present in options
                ans_lower = question.correct_answer.strip().lower()
                if not any(ans_lower == opt.strip().lower() for opt in question.options):
                    issues.append("Correct answer is not present in the MCQ options list.")
                    quality_score -= 35.0

                # Check that distractors are distinct from correct answer
                distractors = [opt for opt in question.options if opt.strip().lower() != ans_lower]
                if len(distractors) != 3:
                    issues.append("MCQ does not contain exactly three incorrect distractors.")
                    quality_score -= 20.0

                # Check for multiple defensible answers in MCQs
                if source_content:
                    is_ambig, ambig_dist = self.check_mcq_multiple_defensible_answers(
                        question_text=question.question_text,
                        correct_answer=question.correct_answer,
                        options=question.options,
                        source_text=source_content.full_text or ""
                    )
                    if is_ambig:
                        issues.append(
                            f"MCQ contains multiple defensible answers: distractor(s) {ambig_dist} "
                            f"are also factually supported as correct answers by the source text."
                        )
                        quality_score -= 35.0

        # 6. Source grounding and question-answering verification (without vector search)
        if source_content and question.question_text:
            # Check targeted segment first if available
            target_text = ""
            seg_id = question.generation_metadata.get("segment_id") if question.generation_metadata else None
            if seg_id is not None and source_content.segments:
                for s in source_content.segments:
                    if s.segment_id == seg_id:
                        target_text = s.text
                        break
            if not target_text and question.source_reference and source_content.segments:
                for s in source_content.segments:
                    if s.source_label == question.source_reference:
                        target_text = s.text
                        break
            if not target_text:
                target_text = source_content.full_text or ""

            if target_text:
                stop = {"what", "which", "where", "when", "does", "from", "with", "that", "this", "have", "been", "were", "explain", "describe", "state"}
                q_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.question_text.lower())) - stop
                ans_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.correct_answer.lower())) - stop

                # Check 1: Answer vocabulary supported by source
                if ans_keywords and not any(w in target_text.lower() for w in ans_keywords):
                    issues.append("Correct answer is not supported by the source text.")
                    quality_score -= 40.0

                # Check 2: Answer addresses the question (topical co-occurrence in source sentences)
                sentences = re.split(r'(?<=[.?!])\s+', target_text)
                q_sentences = [s for s in sentences if any(w in s.lower() for w in q_keywords)]
                if q_sentences and ans_keywords:
                    in_grounding = any(any(w in s.lower() for w in ans_keywords) for s in q_sentences)
                    if not in_grounding:
                        issues.append(
                            "Answer does not address the question: answer concepts belong to an "
                            "unrelated context with no co-occurrence in the source text."
                        )
                        quality_score -= 40.0

        is_valid = len(issues) == 0
        final_score = max(0.0, round(quality_score, 1))

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            quality_score=final_score
        )

    def validate_question_suite(
        self,
        questions: List[GeneratedQuestion],
        source_content: Optional[LectureContent] = None,
        similarity_threshold: float = 0.80
    ) -> Dict[str, Any]:
        """
        Validates an entire suite of generated questions, checking for duplicates across items.
        """
        results = []
        duplicate_pairs = []
        valid_count = 0

        # 1. Validate each question individually
        for q in questions:
            res = self.validate_question(q, source_content=source_content)
            if res.is_valid:
                valid_count += 1
            results.append({
                "question_id": q.question_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "bloom_level": q.bloom_level,
                "difficulty": q.difficulty,
                "source_reference": q.source_reference,
                "validation": res.to_dict()
            })

        # 2. Check for duplicate/near-identical questions across the suite
        n = len(questions)
        for i in range(n):
            for j in range(i + 1, n):
                q1 = questions[i].question_text.lower().strip()
                q2 = questions[j].question_text.lower().strip()
                sim = SequenceMatcher(None, q1, q2).ratio()
                if sim >= similarity_threshold:
                    duplicate_pairs.append({
                        "question_id_1": questions[i].question_id,
                        "question_id_2": questions[j].question_id,
                        "similarity": round(sim, 2),
                        "text_1": questions[i].question_text,
                        "text_2": questions[j].question_text
                    })

        total = len(questions)
        avg_quality = round(sum(r["validation"]["quality_score"] for r in results) / max(1, total), 1)

        return {
            "total_questions": total,
            "valid_questions": valid_count,
            "invalid_questions": total - valid_count,
            "duplicate_count": len(duplicate_pairs),
            "duplicate_pairs": duplicate_pairs,
            "average_quality_score": avg_quality,
            "question_evaluations": results
        }
