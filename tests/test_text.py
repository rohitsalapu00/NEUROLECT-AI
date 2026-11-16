"""
NEUROLECT AI - Test Text Processor
Validates plain text processing, TXT file reading, validation, and error handling.
"""

import pytest
from pathlib import Path
from modules.text_processor import TextProcessor, TextProcessingError
from modules.content_models import LectureContent


@pytest.fixture
def text_processor():
    return TextProcessor()


def test_process_raw_text_valid(text_processor):
    """Verify processing of valid multi-paragraph educational text."""
    sample_text = (
        "Natural Language Processing is a subfield of artificial intelligence.\n\n"
        "Transformers utilize self-attention mechanisms to process sequential tokens in parallel.\n\n"
        "Popular architectures include BERT for encoder representations and GPT for autoregressive generation."
    )
    result = text_processor.process_raw_text(sample_text, title="NLP Overview")

    assert isinstance(result, LectureContent)
    assert result.source_type == "text"
    assert result.title == "NLP Overview"
    assert result.total_words > 20
    assert result.total_segments == 3
    assert result.total_chunks >= 1
    assert "Transformers" in result.full_text


def test_process_raw_text_empty(text_processor):
    """Verify that empty or whitespace-only text raises TextProcessingError."""
    with pytest.raises(TextProcessingError):
        text_processor.process_raw_text("   \n   \t  ")


def test_process_raw_text_too_short(text_processor):
    """Verify that insufficient content raises TextProcessingError."""
    with pytest.raises(TextProcessingError):
        text_processor.process_raw_text("Too short.")


def test_process_txt_file(tmp_path, text_processor):
    """Verify TXT file loading and processing from disk."""
    file_path = tmp_path / "lecture_notes.txt"
    content = (
        "Supervised learning algorithms train on labeled datasets.\n\n"
        "Common algorithms include linear regression, support vector machines, and random forests.\n\n"
        "Evaluation metrics include precision, recall, F1 score, and area under the ROC curve."
    )
    file_path.write_text(content, encoding="utf-8")

    result = text_processor.process_txt_file(file_path)
    assert isinstance(result, LectureContent)
    assert result.source_filename == "lecture_notes.txt"
    assert result.total_segments == 3
    assert result.total_words > 15


def test_process_txt_file_nonexistent(text_processor):
    """Verify that non-existent TXT path raises TextProcessingError."""
    with pytest.raises(TextProcessingError):
        text_processor.process_txt_file("non_existent_file.txt")


def test_process_txt_file_invalid_extension(tmp_path, text_processor):
    """Verify that invalid extension raises TextProcessingError."""
    invalid_file = tmp_path / "document.docx"
    invalid_file.write_text("Some text", encoding="utf-8")
    with pytest.raises(TextProcessingError):
        text_processor.process_txt_file(invalid_file)
