"""
NEUROLECT AI - Test Bloom's Taxonomy Classifier
Validates contextual and linguistic classification across Bloom's 6 cognitive levels:
Remember, Understand, Apply, Analyze, Evaluate, Create.
"""

import pytest
from modules.bloom_classifier import BloomClassifier


@pytest.fixture
def classifier():
    return BloomClassifier()


def test_classify_remember(classifier):
    """Verify Remember classification for definitional questions."""
    level, conf, _ = classifier.classify("What is the definition of supervised learning?")
    assert level == "Remember"
    assert 0.5 <= conf <= 1.0


def test_classify_understand(classifier):
    """Verify Understand classification for conceptual explanatory questions."""
    level, conf, _ = classifier.classify("Explain why vanishing gradients occur in recurrent networks.")
    assert level == "Understand"
    assert 0.5 <= conf <= 1.0


def test_classify_apply(classifier):
    """Verify Apply classification for implementation and calculation questions."""
    level, conf, _ = classifier.classify("How would you calculate the loss given predicted and actual values?")
    assert level == "Apply"
    assert 0.5 <= conf <= 1.0


def test_classify_analyze(classifier):
    """Verify Analyze classification for comparative questions."""
    level, conf, _ = classifier.classify("Compare and contrast convolutional networks versus recurrent networks.")
    assert level == "Analyze"
    assert 0.5 <= conf <= 1.0


def test_classify_evaluate(classifier):
    """Verify Evaluate classification for critique and assessment questions."""
    level, conf, _ = classifier.classify("Assess and critique the effectiveness of L1 regularization over L2.")
    assert level == "Evaluate"
    assert 0.5 <= conf <= 1.0


def test_classify_create(classifier):
    """Verify Create classification for synthesis and architectural design questions."""
    level, conf, _ = classifier.classify("How would you design a novel neural pipeline for multimodal learning?")
    assert level == "Create"
    assert 0.5 <= conf <= 1.0


def test_classify_empty(classifier):
    """Verify empty question text returns safe fallback."""
    level, conf, _ = classifier.classify("")
    assert level == "Remember"
    assert conf == 0.50
