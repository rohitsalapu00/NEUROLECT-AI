"""
NEUROLECT AI - Test Distractor Generator
Validates MCQ distractor generation, option shuffling, deduplication, and answer isolation.
"""

import pytest
from modules.distractor_generator import DistractorGenerator, DistractorGenerationError


@pytest.fixture
def generator():
    return DistractorGenerator()


def test_generate_distractors_count_and_uniqueness(generator):
    """Verify exactly 3 distinct distractors are generated and none match the answer."""
    q = "What is the primary objective of gradient descent?"
    ans = "Minimizing the objective loss function"
    context = (
        "Gradient descent minimizes the objective loss function. "
        "Support vector machines maximize the geometric margin between classes. "
        "Principal component analysis reduces feature dimensionality. "
        "K-means partitions observations into Voronoi clusters."
    )
    distractors, is_valid = generator.generate_distractors(
        question_text=q,
        correct_answer=ans,
        context=context,
        domain_keywords=["Support Vector Machines", "Principal Component Analysis", "K-Means Clustering"]
    )

    assert len(distractors) == 3
    assert is_valid is True
    # None of the distractors can equal the correct answer
    assert all(d.lower() != ans.lower() for d in distractors)
    # Distractors must all be mutually distinct
    assert len(set(d.lower() for d in distractors)) == 3


def test_build_shuffled_mcq(generator):
    """Verify exactly 4 options are returned with correct answer present."""
    q = "Which gate in an LSTM decides what information to discard?"
    ans = "Forget gate"
    context = "LSTMs contain an input gate, a forget gate, and an output gate."

    options, correct_ans = generator.build_shuffled_mcq(
        question_text=q,
        correct_answer=ans,
        context=context,
        domain_keywords=["Input Gate", "Output Gate", "Update Gate"]
    )

    assert len(options) == 4
    assert correct_ans == ans
    assert ans in options
    # Exactly one occurrence of correct answer
    assert options.count(ans) == 1


def test_empty_answer_raises(generator):
    """Verify empty answer raises DistractorGenerationError."""
    with pytest.raises(DistractorGenerationError):
        generator.generate_distractors("Question?", "", "Context")
