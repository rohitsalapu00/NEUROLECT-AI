"""
NEUROLECT AI - Test Concept Extractor
Validates educational definition extraction, technical principle identification,
and source citation tracking.
"""

import pytest
from modules.concept_extractor import ConceptExtractor, ConceptExtractionError
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def concept_extractor():
    return ConceptExtractor()


@pytest.fixture
def concept_sample_content():
    seg1 = ContentSegment(
        segment_id=0,
        text="Backpropagation is defined as an algorithm for computing partial derivatives of the loss function. "
             "A fundamental principle in neural optimization is adjusting weights in the negative gradient direction.",
        source_type="pdf",
        page_number=3
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="Batch Normalization refers to a technique that standardizes layer inputs across mini-batches. "
             "Unlike standard SGD, Adam optimizer computes adaptive learning rates for each parameter.",
        source_type="pdf",
        page_number=4
    )
    full_text = f"{seg1.text}\n\n{seg2.text}"

    return LectureContent(
        title="Optimization Techniques",
        source_type="pdf",
        source_filename="optimization.pdf",
        full_text=full_text,
        segments=[seg1, seg2]
    )


def test_extract_definitions(concept_extractor, concept_sample_content):
    """Verify definitions are parsed and linked to source page numbers."""
    result = concept_extractor.extract_concepts(concept_sample_content)

    assert isinstance(result, dict)
    assert "definitions" in result
    assert "principles" in result
    assert len(result["definitions"]) >= 1

    terms = [d["term"].lower() for d in result["definitions"]]
    assert any("backpropagation" in t or "batch normalization" in t for t in terms)

    # Verify source citation
    assert any(d["source"] in ["Page 3", "Page 4"] for d in result["definitions"])


def test_extract_principles(concept_extractor, concept_sample_content):
    """Verify technical principles triggered by significance terms are extracted."""
    result = concept_extractor.extract_concepts(concept_sample_content)
    assert len(result["principles"]) >= 1
    assert any("fundamental" in p["statement"].lower() for p in result["principles"])


def test_concept_extractor_empty_raises(concept_extractor):
    """Verify empty segments raise ConceptExtractionError."""
    empty = LectureContent(
        title="Empty",
        source_type="text",
        source_filename="empty.txt",
        full_text="",
        segments=[]
    )
    with pytest.raises(ConceptExtractionError):
        concept_extractor.extract_concepts(empty)
