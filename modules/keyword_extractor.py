"""
NEUROLECT AI - Keyword and Keyphrase Extraction Module
Extracts and ranks educational keywords and technical terms using KeyBERT.
Employs Maximal Marginal Relevance (MMR) for deduplication and associates keywords with topics.
"""

import logging
from typing import List, Dict, Any, Optional
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent
from utils.helpers import clean_text

logger = logging.getLogger("NEUROLECT.KeywordExtractor")


class KeywordExtractionError(Exception):
    """Raised when KeyBERT keyword extraction fails."""
    pass


class KeywordExtractor:
    """
    Extracts educational keywords and technical terms from LectureContent.
    Uses KeyBERT with Maximal Marginal Relevance (MMR) to balance semantic
    relevance and lexical diversity while eliminating duplicates.
    """

    def __init__(self):
        pass

    def extract_keywords(
        self,
        content: LectureContent,
        top_n: int = 15,
        ngram_range: tuple = (1, 2),
        diversity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extracts top ranked technical terms from the full lecture content.
        Uses MMR (Maximal Marginal Relevance) to prune redundant terms.
        """
        cleaned = clean_text(content.full_text)
        if not cleaned or len(cleaned.split()) < 5:
            raise KeywordExtractionError("Lecture content is too short for keyword extraction.")

        try:
            kw_model = ModelLoader.get_keybert_model()
        except ModelLoadingError as e:
            raise KeywordExtractionError(f"KeyBERT model unavailable: {e}") from e

        try:
            # Extract keyphrases with MMR to avoid duplicate variants
            raw_keywords = kw_model.extract_keywords(
                cleaned,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                use_mmr=True,
                diversity=diversity,
                top_n=top_n
            )
        except Exception as e:
            logger.error(f"KeyBERT extraction failed: {e}")
            raise KeywordExtractionError(f"Keyword extraction failed: {e}") from e

        formatted_keywords = []
        seen = set()

        for kw, score in raw_keywords:
            clean_kw = kw.strip().lower()
            if clean_kw not in seen and len(clean_kw) > 2:
                seen.add(clean_kw)
                formatted_keywords.append({
                    "keyword": kw.title(),
                    "score": round(float(score), 4),
                    "relevance_percent": int(round(float(score) * 100)),
                    "type": "Keyphrase" if " " in kw else "Term"
                })

        logger.info(f"Extracted {len(formatted_keywords)} unique keywords from '{content.title}'.")
        return formatted_keywords

    def extract_topic_keywords(
        self,
        topics: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extracts specific technical keywords for each detected lecture topic.
        """
        try:
            kw_model = ModelLoader.get_keybert_model()
        except ModelLoadingError as e:
            raise KeywordExtractionError(f"KeyBERT model unavailable: {e}") from e

        topic_keyword_map = []
        for topic in topics:
            text = topic.get("associated_text", "")
            if not text or len(text.split()) < 10:
                topic_keyword_map.append({
                    "topic_id": topic.get("topic_id", 0),
                    "topic_name": topic.get("name", "Topic"),
                    "keywords": []
                })
                continue

            try:
                raw_kws = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    use_mmr=True,
                    diversity=0.4,
                    top_n=top_n
                )
                kws = [kw.title() for kw, _ in raw_kws]
            except Exception:
                kws = []

            topic_keyword_map.append({
                "topic_id": topic.get("topic_id", 0),
                "topic_name": topic.get("name", "Topic"),
                "keywords": kws
            })

        return topic_keyword_map
