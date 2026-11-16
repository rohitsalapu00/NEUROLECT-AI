"""
NEUROLECT AI - Configuration Module
Intelligent Lecture Understanding, Question Generation and Automated Assessment System
"""

import os
from pathlib import Path

# Silence harmless Windows HF symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
UPLOADS_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "neurolect.db"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
for path in [DATA_DIR, TEMP_DIR, UPLOADS_DIR, ASSETS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Application Information
APP_NAME = "NEUROLECT AI"
APP_TAGLINE = "Intelligent Lecture Understanding, Question Generation & Assessment System"
APP_VERSION = "1.0.0"

# Supported Input Formats
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".mkv"]
SUPPORTED_DOC_FORMATS = [".pdf", ".txt"]
MAX_UPLOAD_SIZE_MB = 200

# Text Processing & Chunking Configuration
DEFAULT_CHUNK_SIZE = 1200          # Characters per chunk for transformer input limits
DEFAULT_CHUNK_OVERLAP = 150        # Overlap characters to maintain context
MAX_SUMMARY_INPUT_TOKENS = 1024

# Pretrained Hugging Face Models (Optimized for CPU & Offline Inference)
MODELS = {
    # Speech Recognition
    "whisper": "base",  # Options: 'tiny', 'base', 'small'
    
    # Summarization (genuine instruction-tuned Flan-T5 transformer)
    "summarizer": "google/flan-t5-small",
    
    # Question Generation (genuine instruction-tuned Flan-T5)
    "question_generator": "google/flan-t5-small",
    
    # Named Entity Recognition
    "ner": "dslim/bert-base-NER",
    
    # Semantic Answer Evaluation & Embedding
    "semantic_evaluator": "sentence-transformers/all-MiniLM-L6-v2",
}

# Question Generation Configurations
QUESTION_TYPES = [
    "MCQ",
    "Short Answer",
    "Descriptive",
    "Conceptual",
    "Application",
    "Analytical",
]

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

BLOOMS_TAXONOMY = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

# Scoring Weights for Answer Evaluation
SEMANTIC_SIMILARITY_WEIGHT = 0.50
KEYWORD_COVERAGE_WEIGHT = 0.30
CONCEPT_MATCH_WEIGHT = 0.20
