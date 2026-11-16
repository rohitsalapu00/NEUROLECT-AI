"""
NEUROLECT AI - Named Entity Recognition (NER) Module
Extracts and categorizes named entities (People, Organizations, Locations, Miscellaneous)
using pretrained BERT-NER (dslim/bert-base-NER) from Hugging Face.
Preserves segment-level source tracking (page numbers / timestamps).
"""

import logging
from typing import List, Dict, Any, Optional
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent
from utils.helpers import clean_text

logger = logging.getLogger("NEUROLECT.NERProcessor")


class NERProcessingError(Exception):
    """Raised when named entity recognition pipeline fails."""
    pass


class NERProcessor:
    """
    Identifies people, organizations, locations, and technical entities from LectureContent.
    Uses Hugging Face dslim/bert-base-NER with simple aggregation to consolidate subword tokens.
    """

    # Human-readable category mappings
    ENTITY_LABELS = {
        "PER": "Persons & Researchers",
        "ORG": "Organizations & Institutions",
        "LOC": "Locations & Facilities",
        "MISC": "Technical Terms, Datasets & Miscellaneous",
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name

    def extract_entities(self, content: LectureContent) -> Dict[str, Any]:
        """
        Runs token classification across content segments, preserving source provenance.
        Returns categorized entities with frequency, confidence score, and source citations.
        """
        if not content.segments:
            raise NERProcessingError("LectureContent has no segments for entity extraction.")

        try:
            ner_pipe = ModelLoader.get_ner_pipeline(self.model_name)
        except ModelLoadingError as e:
            raise NERProcessingError(f"NER pipeline unavailable: {e}") from e

        # Structured container for entities
        categorized_entities: Dict[str, Dict[str, Dict[str, Any]]] = {
            "PER": {},
            "ORG": {},
            "LOC": {},
            "MISC": {},
        }

        total_extracted = 0

        from utils.helpers import split_into_sentences

        for seg in content.segments:
            text = clean_text(seg.text)
            if not text or len(text.split()) < 3:
                continue

            # Split oversized segments at sentence boundaries so NO text is discarded
            sentences = split_into_sentences(text)
            text_blocks = []
            curr_block = []
            curr_len = 0
            for s in sentences:
                if curr_len + len(s) > 1000 and curr_block:
                    text_blocks.append(" ".join(curr_block))
                    curr_block = [s]
                    curr_len = len(s)
                else:
                    curr_block.append(s)
                    curr_len += len(s)
            if curr_block:
                text_blocks.append(" ".join(curr_block))

            for block in text_blocks:
                try:
                    results = ner_pipe(block)
                except Exception as e:
                    logger.warning(f"NER extraction error on segment {seg.segment_id}: {e}")
                    continue

                for item in results:
                    entity_group = item.get("entity_group", "MISC")
                    word = item.get("word", "").strip()
                    score = float(item.get("score", 0.0))

                    # Filter out single-character or punctuation artifacts
                    if len(word) <= 1 or word.startswith("##") or entity_group not in categorized_entities:
                        continue

                    clean_word = word.replace(" ##", "").replace("##", "").strip()
                    if not clean_word:
                        continue

                    cat_dict = categorized_entities[entity_group]
                    if clean_word not in cat_dict:
                        cat_dict[clean_word] = {
                            "name": clean_word,
                            "category": entity_group,
                            "category_label": self.ENTITY_LABELS.get(entity_group, entity_group),
                            "count": 1,
                            "confidence": round(score, 3),
                            "sources": [seg.source_label]
                        }
                    else:
                        cat_dict[clean_word]["count"] += 1
                        cat_dict[clean_word]["confidence"] = max(cat_dict[clean_word]["confidence"], round(score, 3))
                        if seg.source_label not in cat_dict[clean_word]["sources"]:
                            cat_dict[clean_word]["sources"].append(seg.source_label)

                    total_extracted += 1

        # Format and sort by occurrence frequency
        formatted_output = {
            "total_entities_detected": total_extracted,
            "categories": {}
        }

        for group_code, group_label in self.ENTITY_LABELS.items():
            items = list(categorized_entities[group_code].values())
            # Sort by frequency descending, then confidence descending
            items.sort(key=lambda x: (x["count"], x["confidence"]), reverse=True)
            formatted_output["categories"][group_code] = {
                "label": group_label,
                "count": len(items),
                "entities": items
            }

        logger.info(
            f"NER extraction complete for '{content.title}': "
            f"{total_extracted} total occurrences across {len(content.segments)} segments."
        )
        return formatted_output
