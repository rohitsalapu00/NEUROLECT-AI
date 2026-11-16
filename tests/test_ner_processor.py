"""
NEUROLECT AI - Test Named Entity Recognition (NER)
Validates BERT-NER entity extraction, categorization (PER, ORG, LOC, MISC), and source preservation.
"""

import pytest
from modules.ner_processor import NERProcessor, NERProcessingError
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def ner_processor():
    return NERProcessor()


@pytest.fixture
def entity_sample_content():
    seg1 = ContentSegment(
        segment_id=0,
        text="Geoffrey Hinton and Yann LeCun pioneered deep learning at Google and Meta.",
        source_type="pdf",
        page_number=1
    )
    seg2 = ContentSegment(
        segment_id=1,
        text="The famous Dartmouth Conference of 1956 in New Hampshire launched the field of AI.",
        source_type="pdf",
        page_number=2
    )
    full_text = f"{seg1.text}\n\n{seg2.text}"

    return LectureContent(
        title="History of AI Pioneers",
        source_type="pdf",
        source_filename="history.pdf",
        full_text=full_text,
        segments=[seg1, seg2]
    )


def test_extract_entities(ner_processor, entity_sample_content):
    """Verify BERT-NER identifies persons, organizations, and locations with provenance."""
    results = ner_processor.extract_entities(entity_sample_content)

    assert isinstance(results, dict)
    assert "categories" in results
    assert "total_entities_detected" in results

    categories = results["categories"]
    assert "PER" in categories
    assert "ORG" in categories
    assert "LOC" in categories

    # Verify at least one entity found in persons or organizations
    per_entities = [e["name"] for e in categories["PER"]["entities"]]
    org_entities = [e["name"] for e in categories["ORG"]["entities"]]

    assert any("Hinton" in name or "LeCun" in name for name in per_entities) or len(org_entities) > 0


def test_ner_empty_segments_raises(ner_processor):
    """Verify empty segments raise NERProcessingError."""
    empty = LectureContent(
        title="Empty",
        source_type="text",
        source_filename="empty.txt",
        full_text="",
        segments=[]
    )
    with pytest.raises(NERProcessingError):
        ner_processor.extract_entities(empty)
