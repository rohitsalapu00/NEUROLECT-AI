"""
NEUROLECT AI - Model Loader & Cache Registry
Implements lazy-loading singleton patterns and memory management for all transformer models.
Models are initialized on CPU only when requested by the user, with explicit unloading support.
"""

import os
import gc
import logging
from typing import Any, Dict, Optional, Tuple, List
import torch
import psutil
from config import MODELS
from utils.file_manager import ensure_ffmpeg_on_path

logger = logging.getLogger("NEUROLECT.ModelLoader")


class ModelLoadingError(Exception):
    """Raised when an NLP or speech recognition model fails to download or initialize."""
    pass


class ModelLoader:
    """
    Central registry for lazily loaded pretrained models.
    Guarantees no heavy model downloads happen on application startup.
    Caches model instances in memory for fast subsequent requests with safe eviction.
    """
    _cache: Dict[str, Any] = {}

    @classmethod
    def get_whisper_model(cls, model_name: Optional[str] = None) -> Any:
        """
        Lazily loads and returns the OpenAI Whisper speech recognition model.
        Optimized for CPU inference. Ensures FFmpeg binary is on system PATH.
        """
        if model_name is None:
            model_name = MODELS.get("whisper", "base")

        cache_key = f"whisper_{model_name}"
        if cache_key in cls._cache:
            logger.info(f"Using cached Whisper model: {model_name}")
            return cls._cache[cache_key]

        logger.info(f"Loading Whisper model '{model_name}' on CPU...")
        ensure_ffmpeg_on_path()

        try:
            import whisper
            model = whisper.load_model(model_name, device="cpu")
            cls._cache[cache_key] = model
            logger.info(f"Whisper model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}")
            raise ModelLoadingError(
                f"Could not load Whisper model '{model_name}'. Details: {e}"
            ) from e

    @classmethod
    def get_summarizer_model(cls, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Lazily loads and returns the tokenizer and AutoModelForSeq2SeqLM for summarization.
        Uses google/flan-t5-small on CPU by default.
        """
        if model_name is None:
            model_name = MODELS.get("summarizer", "google/flan-t5-small")

        cache_key = f"summarizer_{model_name}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        logger.info(f"Loading sequence-to-sequence summarizer model '{model_name}' on CPU...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.eval()
            cls._cache[cache_key] = (tokenizer, model)
            logger.info(f"Summarizer model '{model_name}' loaded successfully.")
            return cls._cache[cache_key]
        except Exception as e:
            logger.error(f"Failed to load summarizer model '{model_name}': {e}")
            raise ModelLoadingError(
                f"Could not load summarizer model '{model_name}'. Details: {e}"
            ) from e

    @classmethod
    def get_question_generator_model(cls, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Lazily loads and returns the tokenizer and AutoModelForSeq2SeqLM for question generation.
        Reuses the cached FLAN-T5 model from summarization if the model checkpoint matches,
        guaranteeing zero duplicate model instances in memory.
        """
        if model_name is None:
            model_name = MODELS.get("question_generator", "google/flan-t5-small")

        summarizer_key = f"summarizer_{model_name}"
        if summarizer_key in cls._cache:
            return cls._cache[summarizer_key]

        return cls.get_summarizer_model(model_name)

    @classmethod
    def get_ner_pipeline(cls, model_name: Optional[str] = None) -> Any:
        """
        Lazily loads and returns the Hugging Face token-classification NER pipeline.
        Uses dslim/bert-base-NER on CPU by default.
        """
        if model_name is None:
            model_name = MODELS.get("ner", "dslim/bert-base-NER")

        cache_key = f"ner_{model_name}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        logger.info(f"Loading token-classification NER pipeline '{model_name}' on CPU...")
        try:
            from transformers import pipeline
            ner_pipe = pipeline(
                "token-classification",
                model=model_name,
                aggregation_strategy="simple",
                device="cpu"
            )
            cls._cache[cache_key] = ner_pipe
            logger.info(f"NER pipeline '{model_name}' loaded successfully.")
            return ner_pipe
        except Exception as e:
            logger.error(f"Failed to load NER pipeline '{model_name}': {e}")
            raise ModelLoadingError(
                f"Could not load NER pipeline '{model_name}'. Details: {e}"
            ) from e

    @classmethod
    def get_sentence_transformer(cls, model_name: Optional[str] = None) -> Any:
        """
        Lazily loads and returns the SentenceTransformer model.
        Uses sentence-transformers/all-MiniLM-L6-v2 on CPU by default.
        """
        if model_name is None:
            model_name = MODELS.get("semantic_evaluator", "sentence-transformers/all-MiniLM-L6-v2")

        cache_key = f"sentence_transformer_{model_name}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        logger.info(f"Loading SentenceTransformer '{model_name}' on CPU...")
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(model_name, device="cpu")
            cls._cache[cache_key] = st_model
            logger.info(f"SentenceTransformer '{model_name}' loaded successfully.")
            return st_model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer '{model_name}': {e}")
            raise ModelLoadingError(
                f"Could not load SentenceTransformer '{model_name}'. Details: {e}"
            ) from e

    @classmethod
    def get_keybert_model(cls) -> Any:
        """
        Lazily initializes KeyBERT reusing the cached SentenceTransformer model.
        Guarantees zero duplicate embedding model downloads.
        """
        cache_key = "keybert_instance"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        logger.info("Initializing KeyBERT model with shared SentenceTransformer...")
        try:
            from keybert import KeyBERT
            st_model = cls.get_sentence_transformer()
            kw_model = KeyBERT(model=st_model)
            cls._cache[cache_key] = kw_model
            logger.info("KeyBERT model initialized successfully.")
            return kw_model
        except Exception as e:
            logger.error(f"Failed to initialize KeyBERT: {e}")
            raise ModelLoadingError(f"Could not initialize KeyBERT. Details: {e}") from e

    @classmethod
    def is_model_loaded(cls, model_key: str) -> bool:
        """Checks if a model is currently loaded in the memory cache."""
        return model_key in cls._cache

    @classmethod
    def unload_model(cls, model_key: str) -> bool:
        """
        Safely unloads a specific model from the memory cache and triggers garbage collection.
        Explicitly detaches underlying model and tokenizer references to prevent circular leaks.
        Returns True if the model was present and removed, False otherwise.
        """
        if model_key in cls._cache:
            model_obj = cls._cache.pop(model_key)
            # Break internal pipeline / tuple references to allow Python CRT heap cleanup
            if hasattr(model_obj, "model"):
                try:
                    del model_obj.model
                except Exception:
                    pass
            if hasattr(model_obj, "tokenizer"):
                try:
                    del model_obj.tokenizer
                except Exception:
                    pass
            if isinstance(model_obj, tuple):
                for item in model_obj:
                    del item
            del model_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model '{model_key}' from memory cache.")
            return True
        return False

    @classmethod
    def free_heavy_models(cls, keep_shared_embeddings: bool = True) -> List[str]:
        """
        Evicts heavy models (Whisper, NER, T5) to free RAM while optionally
        preserving lightweight shared sentence embeddings used by KeyBERT and topic detection.
        """
        unloaded = []
        keys_to_remove = []

        for key in list(cls._cache.keys()):
            if keep_shared_embeddings and ("sentence_transformer" in key or "keybert" in key):
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            if cls.unload_model(key):
                unloaded.append(key)

        logger.info(f"Freed heavy models from memory: {unloaded}")
        return unloaded
        return unloaded

    @classmethod
    def get_memory_status(cls) -> Dict[str, Any]:
        """
        Reports currently loaded models and system RAM usage statistics.
        """
        vm = psutil.virtual_memory()
        process = psutil.Process()
        return {
            "loaded_models": list(cls._cache.keys()),
            "total_loaded_count": len(cls._cache),
            "process_ram_mb": round(process.memory_info().rss / (1024 * 1024), 1),
            "system_available_ram_mb": round(vm.available / (1024 * 1024), 1),
            "system_percent_ram_used": vm.percent,
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Frees all cached models from RAM."""
        cls._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared all models from memory cache.")
