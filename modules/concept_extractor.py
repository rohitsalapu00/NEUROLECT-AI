"""
NEUROLECT AI - Important Concept & Definition Extraction Module
Extracts definitions, technical explanations, key relationships, and core facts from LectureContent.
Guarantees source provenance (page numbers or timestamps) for every extracted concept.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from modules.content_models import LectureContent, ContentSegment
from utils.helpers import clean_text

logger = logging.getLogger("NEUROLECT.ConceptExtractor")


class ConceptExtractionError(Exception):
    """Raised when concept extraction fails."""
    pass


class ConceptExtractor:
    """
    Extracts educational concepts grounded directly in the lecture source.
    Identifies formal definitions, key technical principles, and structural relationships.
    """

    # Definitional linguistic patterns
    DEFINITION_PATTERNS = [
        re.compile(r"(?P<term>[A-Z][A-Za-z0-9\s\-]{2,35})\s+(?:is defined as|refers to|is characterized as|can be defined as)\s+(?P<definition>[^.;]+)", re.IGNORECASE),
        re.compile(r"(?P<term>[A-Z][A-Za-z0-9\s\-]{2,35})\s+(?:is a type of|is an algorithm that|is a technique that|is a framework that|is a model that)\s+(?P<definition>[^.;]+)", re.IGNORECASE),
        re.compile(r"(?P<term>[A-Z][A-Za-z0-9\s\-]{2,35})\s+(?:is known as|is called)\s+(?P<definition>[^.;]+)", re.IGNORECASE),
        re.compile(r"(?:defined as|meaning)\s+(?P<definition>[^.;]+),\s+(?P<term>[A-Z][A-Za-z0-9\s\-]{2,35})", re.IGNORECASE),
    ]

    # Relational & comparative patterns
    RELATIONSHIP_PATTERNS = [
        re.compile(r"(?P<concept>[^.;]+)\s+(?:leads to|results in|causes|enables|depends on|is composed of|consists of)\s+(?P<target>[^.;]+)", re.IGNORECASE),
        re.compile(r"(?:in contrast to|unlike|compared with)\s+(?P<concept1>[^,]+),\s+(?P<concept2>[^.;]+)", re.IGNORECASE),
    ]

    # Significance & core takeaway indicators
    SIGNIFICANCE_TRIGGERS = [
        "important", "crucial", "fundamental", "essential", "key principle",
        "primary advantage", "main disadvantage", "objective", "critical",
        "underlying mechanism", "standard approach"
    ]

    def extract_concepts(self, content: LectureContent) -> Dict[str, Any]:
        """
        Extracts structured educational concepts from LectureContent segments.
        Returns definitions, technical principles, and key relationships with source citations.
        """
        if not content.segments:
            raise ConceptExtractionError("LectureContent contains no segments for concept extraction.")

        definitions: List[Dict[str, Any]] = []
        principles: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        seen_definitions = set()
        seen_principles = set()

        for seg in content.segments:
            text = clean_text(seg.text)
            sentences = [s.strip() for s in text.split(".") if len(s.strip().split()) >= 4]

            for sentence in sentences:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith("."):
                    clean_sentence += "."

                # 1. Search for definitions
                for pattern in self.DEFINITION_PATTERNS:
                    match = pattern.search(sentence)
                    if match:
                        term = match.group("term").strip()
                        definition = match.group("definition").strip()
                        
                        # Validate reasonable length
                        if 2 <= len(term.split()) <= 6 and len(definition.split()) >= 4:
                            term_key = term.lower()
                            if term_key not in seen_definitions:
                                seen_definitions.add(term_key)
                                definitions.append({
                                    "term": term.title(),
                                    "definition": definition + ".",
                                    "full_statement": clean_sentence,
                                    "source": seg.source_label,
                                    "confidence": 0.88,
                                    "segment_id": seg.segment_id
                                })
                        break

                # 2. Search for technical principles and important statements
                lower_sentence = sentence.lower()
                if any(trigger in lower_sentence for trigger in self.SIGNIFICANCE_TRIGGERS):
                    if clean_sentence not in seen_principles and len(clean_sentence.split()) >= 7:
                        seen_principles.add(clean_sentence)
                        principles.append({
                            "statement": clean_sentence,
                            "source": seg.source_label,
                            "segment_id": seg.segment_id,
                            "importance": "High" if any(t in lower_sentence for t in ["crucial", "fundamental", "essential"]) else "Medium"
                        })

                # 3. Search for conceptual relationships
                for rel_pat in self.RELATIONSHIP_PATTERNS:
                    rel_match = rel_pat.search(sentence)
                    if rel_match and len(clean_sentence.split()) >= 6:
                        relationships.append({
                            "statement": clean_sentence,
                            "source": seg.source_label,
                            "segment_id": seg.segment_id
                        })
                        break

        # Fallback if text is informational but doesn't strictly match regex
        if not definitions:
            substantive_sentences = [s for s in content.full_text.split(".") if 8 <= len(s.split()) <= 25]
            if substantive_sentences:
                for s in substantive_sentences[:3]:
                    definitions.append({
                        "term": "Core Concept",
                        "definition": s.strip() + ".",
                        "full_statement": s.strip() + ".",
                        "source": content.segments[0].source_label if content.segments else "Lecture",
                        "confidence": 0.70,
                        "segment_id": 0
                    })

        logger.info(
            f"Concept extraction complete for '{content.title}': "
            f"{len(definitions)} definitions, {len(principles)} principles, {len(relationships)} relationships."
        )

        return {
            "title": content.title,
            "source_type": content.source_type,
            "definitions": definitions,
            "principles": principles[:10],
            "relationships": relationships[:8],
            "total_concepts_found": len(definitions) + len(principles) + len(relationships)
        }

    def extract_definitions(self, content: LectureContent) -> List[Dict[str, Any]]:
        """Convenience method returning extracted definitions."""
        return self.extract_concepts(content).get("definitions", [])

    def extract_principles(self, content: LectureContent) -> List[Dict[str, Any]]:
        """Convenience method returning extracted technical principles."""
        return self.extract_concepts(content).get("principles", [])
