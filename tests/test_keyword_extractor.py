"""
NEUROLECT AI - Test Keyword Extractor
Validates KeyBERT keyword extraction, deduplication, and topic keyword mapping.
"""

import pytest
from modules.keyword_extractor import KeywordExtractor, KeywordExtractionError
from modules.content_models import LectureContent, ContentSegment


@pytest.fixture
def keyword_extractor():
    return KeywordExtractor()


@pytest.fixture
def keyword_sample_content():
    text = (
        "In artificial intelligence, reinforcement learning agents interact with dynamic environments. "
        "The agent selects actions to maximize cumulative reward through exploration and exploitation. "
        "Q-learning is a model-free temporal difference algorithm updating state-action value functions."
    )
    seg = ContentSegment(segment_id=0, text=text, source_type="text")
    return LectureContent(
        title="Reinforcement Learning Fundamentals",
        source_type="text",
        source_filename="rl.txt",
        full_text=text,
        segments=[seg]
    )


def test_extract_keywords(keyword_extractor, keyword_sample_content):
    """Verify KeyBERT extracts, ranks, and scores technical keywords."""
    keywords = keyword_extractor.extract_keywords(keyword_sample_content, top_n=5)
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert len(keywords) <= 5

    for kw in keywords:
        assert "keyword" in kw
        assert "score" in kw
        assert "relevance_percent" in kw
        assert 0.0 <= kw["score"] <= 1.0


def test_extract_topic_keywords(keyword_extractor):
    """Verify keyword extraction for topic dictionaries."""
    topics = [
        {
            "topic_id": 1,
            "name": "Reinforcement Learning",
            "associated_text": "Q-learning agents maximize cumulative rewards using Markov decision processes."
        }
    ]
    topic_kws = keyword_extractor.extract_topic_keywords(topics, top_n=3)
    assert len(topic_kws) == 1
    assert "keywords" in topic_kws[0]


def test_extract_keywords_short_content_raises(keyword_extractor):
    """Verify insufficient content raises KeywordExtractionError."""
    short_content = LectureContent(
        title="Short",
        source_type="text",
        source_filename="short.txt",
        full_text="Two words."
    )
    with pytest.raises(KeywordExtractionError):
        keyword_extractor.extract_keywords(short_content)
