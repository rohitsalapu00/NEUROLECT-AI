"""
NEUROLECT AI - Lecture Summarization Engine
Generates transformer-based concise, detailed, topic-wise summaries and structured lecture notes.
Handles long lectures through token-safe hierarchical chunk-and-combine processing without RAG or vector DBs.
Guarantees that no trailing chunks or paragraphs are silently discarded.
"""

import logging
from typing import List, Dict, Any, Optional
from config import MODELS
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent, ContentChunk
from utils.helpers import clean_text

logger = logging.getLogger("NEUROLECT.Summarizer")


class SummarizationError(Exception):
    """Raised when transformer summarization inference fails."""
    pass


class LectureSummarizer:
    """
    Transforms LectureContent into multi-tier educational summaries using pretrained Seq2Seq transformers.
    Directly processes content chunks hierarchically without vector databases or document retrieval.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or MODELS.get("summarizer", "google/flan-t5-small")

    def _format_prompt(self, text: str) -> str:
        """Formats the input prompt appropriate for the active model family."""
        if "flan" in self.model_name.lower():
            return f"Summarize the following educational text:\n{text}"
        return f"summarize: {text}"

    def _summarize_single_chunk(self, text: str, max_tokens: int = 100, min_tokens: int = 20) -> str:
        """
        Executes genuine transformer generation for a single text chunk.
        """
        cleaned = clean_text(text)
        if len(cleaned.split()) < 15:
            return cleaned

        try:
            tokenizer, model = ModelLoader.get_summarizer_model(self.model_name)
        except ModelLoadingError as e:
            raise SummarizationError(f"Summarizer model unavailable: {e}") from e

        input_prompt = self._format_prompt(cleaned)
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        try:
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_tokens,
                min_length=min_tokens,
                length_penalty=1.5,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Transformer generation error: {e}")
            raise SummarizationError(f"Summarization inference failed: {e}") from e

    def generate_concise_summary(self, content: LectureContent) -> str:
        """
        Generates a concise summary (1-2 paragraphs) capturing core concepts across the entire lecture.
        Guarantees that the beginning, middle, and end chunks are all processed hierarchically.
        """
        if not content.full_text or content.total_words == 0:
            raise SummarizationError("LectureContent has no text to summarize.")

        # Ensure chunks exist and are properly sized for transformer context limits
        chunks = content.chunks
        if (not chunks or len(chunks) <= 1) and content.total_words > 250:
            from modules.content_models import build_chunks_from_segments
            chunks = build_chunks_from_segments(content.segments, content.full_text)

        if not chunks:
            return self._summarize_single_chunk(content.full_text, max_tokens=100, min_tokens=25)

        if len(chunks) == 1:
            return self._summarize_single_chunk(chunks[0].text, max_tokens=100, min_tokens=25)

        # 1. Summarize EVERY chunk without discarding tail segments
        intermediate_summaries = []
        for chunk in chunks:
            if chunk.word_count >= 15:
                chunk_sum = self._summarize_single_chunk(chunk.text, max_tokens=70, min_tokens=15)
                intermediate_summaries.append(chunk_sum)
            else:
                # Retain short tail text directly
                intermediate_summaries.append(chunk.text)

        # 2. Hierarchical reduction: combine in batches of <= 3 to stay strictly within 512 token limits
        current_level = intermediate_summaries
        while len(current_level) > 1:
            next_level = []
            batch_size = 3
            for i in range(0, len(current_level), batch_size):
                batch = current_level[i:i + batch_size]
                if len(batch) == 1 and len(current_level) > batch_size:
                    next_level.append(batch[0])
                else:
                    batch_text = " ".join(batch)
                    batch_sum = self._summarize_single_chunk(batch_text, max_tokens=100, min_tokens=25)
                    next_level.append(batch_sum)
            current_level = next_level

        return current_level[0]

    def generate_detailed_summary(self, content: LectureContent) -> str:
        """
        Generates a comprehensive detailed summary covering all lecture segments.
        Guarantees every chunk is summarized and cited with its source label.
        """
        if not content.full_text:
            raise SummarizationError("LectureContent has no text to summarize.")

        chunks = content.chunks
        if (not chunks or len(chunks) <= 1) and content.total_words > 250:
            from modules.content_models import build_chunks_from_segments
            chunks = build_chunks_from_segments(content.segments, content.full_text)

        if not chunks:
            return self._summarize_single_chunk(content.full_text, max_tokens=200, min_tokens=50)

        if len(chunks) == 1:
            return self._summarize_single_chunk(chunks[0].text, max_tokens=200, min_tokens=50)

        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            label = chunk.source_label or f"Part {idx + 1}"
            if chunk.word_count >= 15:
                chunk_sum = self._summarize_single_chunk(chunk.text, max_tokens=120, min_tokens=25)
            else:
                chunk_sum = chunk.text
            chunk_summaries.append(f"**[{label}]**: {chunk_sum}")

        return "\n\n".join(chunk_summaries)

    def generate_topic_summaries(
        self,
        content: LectureContent,
        topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates dedicated summaries for each detected lecture topic.
        """
        topic_summaries = []
        for topic in topics:
            topic_name = topic.get("name", "General Topic")
            associated_text = topic.get("associated_text", "")
            
            if not associated_text or len(associated_text.split()) < 15:
                summary = associated_text or f"Covers concepts related to {topic_name}."
            else:
                summary = self._summarize_single_chunk(associated_text, max_tokens=90, min_tokens=20)

            topic_summaries.append({
                "topic": topic_name,
                "summary": summary,
                "subtopics": topic.get("subtopics", []),
                "source_label": topic.get("source_label", "Lecture Content")
            })

        return topic_summaries

    def generate_structured_notes(self, content: LectureContent) -> Dict[str, Any]:
        """
        Generates formatted educational lecture notes including executive summary,
        key takeaways, and bulleted concept breakdowns spanning the entire lecture.
        """
        concise = self.generate_concise_summary(content)
        
        sentences = []
        for seg in content.segments:
            for s in seg.text.split(". "):
                clean_s = s.strip()
                if len(clean_s.split()) >= 6 and len(clean_s) < 200:
                    sentences.append((clean_s if clean_s.endswith(".") else clean_s + ".", seg.source_label))

        step = max(1, len(sentences) // 6)
        takeaways = [
            {"point": sent, "source": src}
            for sent, src in sentences[::step][:6]
        ]

        return {
            "title": content.title,
            "source_type": content.source_type,
            "total_words": content.total_words,
            "executive_summary": concise,
            "key_takeaways": takeaways,
        }
