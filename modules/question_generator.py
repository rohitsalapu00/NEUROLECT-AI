"""
NEUROLECT AI - AI Question Generation Engine
Generates genuine transformer-based educational assessments across six question types:
1. Multiple Choice Questions (MCQ) with 3 plausible distractors
2. Short Answer
3. Descriptive
4. Conceptual
5. Application-Based
6. Analytical
Grounded directly in LectureContent with exact page number and timestamp provenance.
Completely RAG-free: operates directly on lecture segments and topic structures.
"""

import uuid
import re
import logging
from typing import List, Dict, Any, Optional
from config import MODELS, QUESTION_TYPES
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent, ContentSegment
from modules.question_models import GeneratedQuestion
from modules.distractor_generator import DistractorGenerator
from modules.bloom_classifier import BloomClassifier
from modules.difficulty_classifier import DifficultyClassifier
from utils.helpers import clean_text, split_into_sentences

logger = logging.getLogger("NEUROLECT.QuestionGenerator")


class QuestionGenerationError(Exception):
    """Raised when question generation pipeline fails."""
    pass


class QuestionGenerator:
    """
    Orchestrates transformer-based question generation, MCQ distractor synthesis,
    Bloom's Taxonomy classification, and difficulty grading across lecture topics.
    """

    # Instruction templates for the 6 educational question categories
    PROMPT_TEMPLATES = {
        "MCQ": "Generate a clear, objective question testing this concept: {sentence}\nContext: {context}\nQuestion:",
        "Short Answer": "Ask a direct, factual short-answer question based on: {sentence}\nContext: {context}\nQuestion:",
        "Descriptive": "Ask an open-ended descriptive question that requires explaining this concept thoroughly: {sentence}\nContext: {context}\nQuestion:",
        "Conceptual": "Ask a conceptual question testing understanding of: {sentence}\nContext: {context}\nQuestion:",
        "Application": "Ask an application question about how an engineer or researcher applies this concept: {sentence}\nContext: {context}\nQuestion:",
        "Analytical": "Ask an analytical question that compares, contrasts, or analyzes: {sentence}\nContext: {context}\nQuestion:",
    }

    FALLBACK_QUESTION_FORMATS = {
        "MCQ": "What is the key principle underlying {concept}?",
        "Short Answer": "What defines the operation of {concept}?",
        "Descriptive": "How would you explain the role and significance of {concept} in educational systems?",
        "Conceptual": "What is the theoretical mechanism governing {concept}?",
        "Application": "How can the principle of {concept} be implemented in practical engineering?",
        "Analytical": "How does {concept} compare and interact with related methodologies?",
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or MODELS.get("question_generator", "google/flan-t5-small")
        self.distractor_gen = DistractorGenerator()
        self.bloom_classifier = BloomClassifier()
        self.difficulty_classifier = DifficultyClassifier()

    def _verify_answer_relevance(
        self,
        question: str,
        answer: str,
        segment_text: str,
        target_sentence: str = ""
    ) -> bool:
        """
        Verifies that an answer:
        1. Is non-empty, substantive, and distinct from the question.
        2. Is lexically grounded in the source segment.
        3. Topically co-occurs in a source sentence related to the question's premise.
        """
        if not answer or not answer.strip():
            return False

        ans_clean = answer.strip()
        q_clean = question.strip()

        # Reject degenerate or trivial outputs
        if (
            ans_clean.lower() == q_clean.lower()
            or len(ans_clean.split()) < 1
            or ans_clean.startswith("(")
            or len(ans_clean) < 2
        ):
            return False

        stop = {
            "what", "which", "where", "when", "does", "from", "with", "that", "this",
            "have", "been", "were", "explain", "describe", "state", "give", "their", "there"
        }
        q_focus_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', q_clean.lower())) - stop
        ans_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ans_clean.lower())) - stop

        # Lexical grounding: answer content words must appear in segment_text
        if ans_words:
            seg_lower = segment_text.lower()
            if not any(w in seg_lower for w in ans_words):
                return False

        # Topical alignment: answer words must appear in a sentence relevant to the question
        sentences = split_into_sentences(segment_text)
        if target_sentence and target_sentence not in sentences:
            sentences.append(target_sentence)

        grounding_sentences = [s for s in sentences if any(w in s.lower() for w in q_focus_words)]
        if not grounding_sentences:
            grounding_sentences = sentences

        ans_in_grounding = any(
            any(w in s.lower() for w in ans_words) for s in grounding_sentences
        ) if ans_words else True

        return ans_in_grounding

    def _extract_relevant_predicate(self, question: str, sentence: str) -> Optional[str]:
        """Extracts the direct predicate or complement answering the question from the target sentence."""
        s = sentence.strip()
        match = re.search(
            r'\b(is|are|predicts|introduces|calculates|computes|measures|refers to|equals)\s+(.+)$',
            s,
            re.IGNORECASE
        )
        if match:
            pred = match.group(2).strip().rstrip(".")
            if len(pred.split()) >= 2:
                return pred
        return None

    def _generate_text(self, prompt: str, max_tokens: int = 64) -> str:
        """Executes forward generation pass using FLAN-T5."""
        try:
            tokenizer, model = ModelLoader.get_question_generator_model(self.model_name)
        except ModelLoadingError as e:
            raise QuestionGenerationError(f"Question generation model unavailable: {e}") from e

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        try:
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=max_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"FLAN-T5 question generation error: {e}")
            raise QuestionGenerationError(f"Inference failed: {e}") from e

    def generate_questions(
        self,
        content: LectureContent,
        topics: Optional[List[Dict[str, Any]]] = None,
        question_types: Optional[List[str]] = None,
        num_questions: int = 6,
        domain_keywords: Optional[List[str]] = None,
        **kwargs
    ) -> List[GeneratedQuestion]:
        """
        Generates a structured assessment suite spanning the requested question types.
        Preserves segment provenance (page numbers or timestamps) for every question.
        """
        if not content.segments:
            raise QuestionGenerationError("LectureContent has no segments available for question generation.")

        actual_num_questions = kwargs.get("total_questions", num_questions)
        selected_types = question_types or QUESTION_TYPES
        generated_questions: List[GeneratedQuestion] = []
        seen_questions = set()

        # Gather lecture segments with substantial text
        viable_segments = [s for s in content.segments if len(s.text.split()) >= 6]
        if not viable_segments:
            viable_segments = content.segments

        # If topics are explicitly provided, filter viable segments to those belonging to the topics
        if topics:
            topic_seg_ids = {sid for t in topics for sid in t.get("segment_ids", []) if sid is not None}
            if topic_seg_ids:
                topic_matched_segments = [s for s in viable_segments if s.segment_id in topic_seg_ids]
                if topic_matched_segments:
                    viable_segments = topic_matched_segments

        # Collect technical concepts from segments to feed distractor pool
        all_concepts_pool = [
            s.text.split(".")[0].strip()
            for s in viable_segments
            if len(s.text.split(".")[0].strip().split()) <= 10
        ]

        # Track sentence pointer per segment for diverse focus anchors
        seg_sentences_map = {}
        seg_ptr_map = {}
        for s in viable_segments:
            sents = split_into_sentences(clean_text(s.text))
            seg_sentences_map[s.segment_id] = sents if sents else [clean_text(s.text)]
            seg_ptr_map[s.segment_id] = 0

        segment_idx = 0
        type_idx = 0
        max_attempts = max(36, actual_num_questions * 5)
        attempts = 0

        while len(generated_questions) < actual_num_questions and attempts < max_attempts:
            attempts += 1
            current_seg = viable_segments[segment_idx % len(viable_segments)]
            current_type = selected_types[type_idx % len(selected_types)]

            segment_idx += 1
            type_idx += 1

            seg_text = clean_text(current_seg.text)
            if len(seg_text.split()) < 4:
                continue

            # Select target focus sentence from current segment
            sentences = seg_sentences_map.get(current_seg.segment_id, [seg_text])
            curr_s_idx = seg_ptr_map.get(current_seg.segment_id, 0)
            target_sentence = sentences[curr_s_idx % len(sentences)]
            seg_ptr_map[current_seg.segment_id] = curr_s_idx + 1

            # 1. Generate Question using FLAN-T5
            template = self.PROMPT_TEMPLATES.get(current_type, self.PROMPT_TEMPLATES["Conceptual"])
            q_prompt = template.format(sentence=target_sentence, context=seg_text[:400])
            raw_q = self._generate_text(q_prompt, max_tokens=64)

            # Clean and validate question string
            q_text = re.sub(r'^(Question|Q):\s*', '', raw_q.strip(), flags=re.IGNORECASE).strip().rstrip('. ')
            if not q_text.endswith("?"):
                q_text += "?"

            # Handle duplicate or degenerate questions with intelligent grounding fallback
            q_lower = q_text.lower()
            if q_lower in seen_questions or len(q_text.split()) < 4:
                # Derive concept noun phrase from target sentence
                words = [w for w in target_sentence.replace(".", "").split() if len(w) > 3]
                concept_name = " ".join(words[:4]) if words else target_sentence[:40]
                fallback_fmt = self.FALLBACK_QUESTION_FORMATS.get(
                    current_type, "What is the primary role of {concept}?"
                )
                q_text = fallback_fmt.format(concept=concept_name).rstrip('. ')
                if not q_text.endswith("?"):
                    q_text += "?"
                q_lower = q_text.lower()
                if q_lower in seen_questions:
                    # Append differentiator
                    q_text = q_text[:-1] + f" (Part {len(generated_questions) + 1})?"
                    q_lower = q_text.lower()

            # 2. Generate Correct Answer Grounded in Context
            ans_prompt = f"Answer this question concisely based on the context:\nContext: {seg_text[:400]}\nQuestion: {q_text}\nAnswer:"
            raw_ans = self._generate_text(ans_prompt, max_tokens=64)
            correct_ans = re.sub(r'^(Answer|A):\s*', '', raw_ans.strip(), flags=re.IGNORECASE).strip()

            # 2. Verify and Ground Correct Answer
            if not self._verify_answer_relevance(q_text, correct_ans, seg_text, target_sentence):
                # Attempt regeneration 1: Focused prompt strictly on target sentence
                retry_prompt = f"Context sentence: {target_sentence}\nQuestion: {q_text}\nAnswer the question concisely based on the context sentence:\nAnswer:"
                retry_raw = self._generate_text(retry_prompt, max_tokens=48)
                retry_ans = re.sub(r'^(Answer|A):\s*', '', retry_raw.strip(), flags=re.IGNORECASE).strip()
                if self._verify_answer_relevance(q_text, retry_ans, seg_text, target_sentence):
                    correct_ans = retry_ans
                else:
                    # Attempt regeneration 2: Extract relevant predicate from target sentence
                    predicate_ans = self._extract_relevant_predicate(q_text, target_sentence)
                    if predicate_ans and self._verify_answer_relevance(q_text, predicate_ans, seg_text, target_sentence):
                        correct_ans = predicate_ans
                    else:
                        # If answer cannot be verified as answering the question, reject the question
                        logger.warning(
                            f"Question '{q_text}' rejected: generated answer could not be verified "
                            f"as answering the question from {current_seg.source_label}."
                        )
                        continue

            # 3. Generate Explanation
            explanation = (
                f"Directly derived from {current_seg.source_label}: '{seg_text[:120]}...'. "
                f"This explains why '{correct_ans}' is the correct response."
            )

            # 4. Handle MCQ Distractors
            options: List[str] = []
            if current_type == "MCQ":
                options, correct_ans = self.distractor_gen.build_shuffled_mcq(
                    question_text=q_text,
                    correct_answer=correct_ans,
                    context=seg_text,
                    domain_keywords=domain_keywords,
                    all_concepts=all_concepts_pool,
                    seed=len(generated_questions)
                )

            # 5. Classify Bloom's Level
            bloom_level, bloom_conf, bloom_rationale = self.bloom_classifier.classify(
                question_text=q_text,
                question_type=current_type
            )

            # 6. Classify Difficulty
            diff_level, diff_score, diff_rationale = self.difficulty_classifier.classify(
                question_text=q_text,
                bloom_level=bloom_level,
                correct_answer=correct_ans
            )

            # 7. Map Topic and Subtopic
            topic_name = "General Lecture"
            subtopic_name = "Core Concept"
            if topics:
                for t in topics:
                    if current_seg.segment_id in t.get("segment_ids", []):
                        topic_name = t.get("name", topic_name)
                        subtopics = t.get("subtopics", [])
                        if subtopics:
                            subtopic_name = subtopics[0]
                        break
                # If single topic was requested and fallback wasn't triggered
                if topic_name == "General Lecture" and len(topics) == 1:
                    topic_name = topics[0].get("name", topic_name)
                    subtopics = topics[0].get("subtopics", [])
                    if subtopics:
                        subtopic_name = subtopics[0]

            # 8. Assemble Standardized Question
            question_obj = GeneratedQuestion(
                question_id=f"q_{uuid.uuid4().hex[:8]}",
                question_text=q_text,
                question_type=current_type,
                options=options,
                correct_answer=correct_ans,
                explanation=explanation,
                topic=topic_name,
                subtopic=subtopic_name,
                bloom_level=bloom_level,
                difficulty=diff_level,
                source_reference=current_seg.source_label,
                generation_metadata={
                    "bloom_confidence": bloom_conf,
                    "difficulty_composite_score": diff_score,
                    "segment_id": current_seg.segment_id,
                    "model_used": self.model_name
                }
            )

            generated_questions.append(question_obj)
            seen_questions.add(q_lower)

        logger.info(
            f"Successfully generated {len(generated_questions)} questions from '{content.title}' "
            f"covering {len(set(q.question_type for q in generated_questions))} question types."
        )
        return generated_questions
