"""
NEUROLECT AI - Test Model Loader & Memory Management
Validates lazy loading, safe model unloading, memory status reporting, and shared embedding usage.
"""

import pytest
from models.model_loader import ModelLoader


def test_shared_embedding_instance():
    """Verify KeyBERT and SentenceTransformer share the exact same underlying model instance in memory."""
    st_model = ModelLoader.get_sentence_transformer()
    kw_model = ModelLoader.get_keybert_model()

    # In KeyBERT, the SentenceTransformer is stored in kw_model.model.embedding_model
    backend_model = getattr(kw_model.model, "embedding_model", kw_model.model)
    assert backend_model == st_model, "KeyBERT must reuse the cached SentenceTransformer to prevent RAM bloat"


def test_memory_status_reporting():
    """Verify that get_memory_status reports loaded models and RAM metrics."""
    status = ModelLoader.get_memory_status()
    assert isinstance(status, dict)
    assert "loaded_models" in status
    assert "process_ram_mb" in status
    assert "system_available_ram_mb" in status
    assert status["process_ram_mb"] > 0


def test_safe_model_unloading():
    """Verify that models can be safely evicted from memory cache."""
    # Ensure a model key is loaded
    _ = ModelLoader.get_sentence_transformer()
    cache_key = "sentence_transformer_sentence-transformers/all-MiniLM-L6-v2"
    assert ModelLoader.is_model_loaded(cache_key)

    # Unload specific model
    unloaded = ModelLoader.unload_model(cache_key)
    assert unloaded is True
    assert not ModelLoader.is_model_loaded(cache_key)

    # Calling unload on non-existent model returns False safely
    assert ModelLoader.unload_model("non_existent_model") is False


def test_model_reloading_after_unload():
    """Verify that a model can be seamlessly reloaded into cache after being unloaded."""
    # 1. Load model
    m1 = ModelLoader.get_sentence_transformer()
    cache_key = "sentence_transformer_sentence-transformers/all-MiniLM-L6-v2"
    assert ModelLoader.is_model_loaded(cache_key)

    # 2. Unload model
    ModelLoader.unload_model(cache_key)
    assert not ModelLoader.is_model_loaded(cache_key)

    # 3. Reload model
    m2 = ModelLoader.get_sentence_transformer()
    assert m2 is not None
    assert ModelLoader.is_model_loaded(cache_key)
