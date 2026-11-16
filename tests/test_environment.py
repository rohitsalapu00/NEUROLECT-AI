"""
NEUROLECT AI - Test Environment & Architecture
Validates packages, paths, configuration, and hardware compatibility.
"""

import os
import sys
import pytest
from pathlib import Path


def test_python_version():
    """Verify Python version is 3.10+."""
    assert sys.version_info >= (3, 10), "Python 3.10 or higher is required."


def test_required_packages():
    """Verify that all core dependencies can be imported successfully."""
    import torch
    import transformers
    import sentence_transformers
    import keybert
    import nltk
    import fitz
    import streamlit
    import plotly
    import whisper
    import imageio_ffmpeg
    
    assert torch.__version__ is not None
    assert transformers.__version__ is not None
    assert sentence_transformers.__version__ is not None
    assert fitz.__version__ is not None


def test_directories_exist():
    """Verify required project directories exist."""
    from config import BASE_DIR, DATA_DIR, TEMP_DIR, UPLOADS_DIR, ASSETS_DIR
    
    for directory in [BASE_DIR, DATA_DIR, TEMP_DIR, UPLOADS_DIR, ASSETS_DIR]:
        assert directory.exists(), f"Directory {directory} does not exist"


def test_ffmpeg_availability():
    """Verify FFmpeg binary is discoverable via imageio_ffmpeg."""
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    assert os.path.exists(exe), f"FFmpeg binary not found at {exe}"


def test_config_models_defined():
    """Verify configured model names are properly set in config."""
    from config import MODELS, QUESTION_TYPES, DIFFICULTY_LEVELS, BLOOMS_TAXONOMY
    
    assert "whisper" in MODELS
    assert "summarizer" in MODELS
    assert "question_generator" in MODELS
    assert "ner" in MODELS
    assert "semantic_evaluator" in MODELS
    assert len(QUESTION_TYPES) == 6
    assert len(DIFFICULTY_LEVELS) == 3
    assert len(BLOOMS_TAXONOMY) == 6
