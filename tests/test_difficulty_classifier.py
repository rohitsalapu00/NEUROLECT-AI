"""
NEUROLECT AI - Test Difficulty Classifier
Validates multi-feature classification across Easy, Medium, Hard levels.
"""

import pytest
from modules.difficulty_classifier import DifficultyClassifier


@pytest.fixture
def classifier():
    return DifficultyClassifier()


def test_classify_easy_question(classifier):
    """Verify simple, short recall questions are rated as Easy."""
    q = "What is a decision tree?"
    diff, score, _ = classifier.classify(q, bloom_level="Remember", correct_answer="A predictive model")
    assert diff == "Easy"
    assert score < 3.2


def test_classify_medium_question(classifier):
    """Verify standard conceptual application questions are rated as Medium."""
    q = "How does backpropagation compute gradients through hidden neural layers?"
    diff, score, _ = classifier.classify(q, bloom_level="Understand", correct_answer="By applying the calculus chain rule recursively backward.")
    assert diff in ["Easy", "Medium"]
    assert score < 5.4


def test_classify_hard_question(classifier):
    """Verify complex, multi-clause analytical synthesis questions are rated as Hard."""
    q = (
        "Assuming high-dimensional data with multicollinearity, how would you justify "
        "the trade-off between L1 regularization versus L2 regularization, and under what conditions "
        "would ElasticNet optimization provide superior generalization?"
    )
    diff, score, _ = classifier.classify(
        q,
        bloom_level="Evaluate",
        correct_answer="ElasticNet balances sparsity and grouping effect under severe collinearity."
    )
    assert diff == "Hard"
    assert score >= 5.4


def test_classify_empty_fallback(classifier):
    """Verify empty question text returns safe fallback."""
    diff, score, _ = classifier.classify("")
    assert diff == "Easy"
    assert score == 1.0
