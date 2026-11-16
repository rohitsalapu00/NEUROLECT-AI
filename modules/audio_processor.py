"""
NEUROLECT AI - Audio Processing Module
Handles audio lecture validation, Whisper transcription with timestamps, and conversion to LectureContent.
Supported formats: MP3, WAV, M4A.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union
from config import SUPPORTED_AUDIO_FORMATS, MAX_UPLOAD_SIZE_MB, MODELS
from utils.validators import validate_file_extension, validate_file_size, ValidationError
from utils.file_manager import ensure_ffmpeg_on_path
from utils.helpers import clean_text
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments

logger = logging.getLogger("NEUROLECT.AudioProcessor")


class AudioProcessingError(Exception):
    """Raised when audio validation, decoding, or transcription fails."""
    pass


class AudioProcessor:
    """
    Transforms audio lecture recordings (MP3, WAV, M4A) into unified LectureContent.
    Uses pretrained OpenAI Whisper on CPU, extracting segment-level timestamps.
    """

    def __init__(self, whisper_model_name: Optional[str] = None):
        self.model_name = whisper_model_name or MODELS.get("whisper", "base")
        ensure_ffmpeg_on_path()

    def validate_audio_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validates that the file exists, has a supported audio extension, and is within size limits.
        """
        path = Path(file_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            validate_file_extension(path.name, SUPPORTED_AUDIO_FORMATS)
            validate_file_size(path, MAX_UPLOAD_SIZE_MB)
        except ValidationError as e:
            raise AudioProcessingError(str(e)) from e

        return path

    def transcribe(
        self,
        audio_path: Union[str, Path],
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        language: Optional[str] = "en"
    ) -> LectureContent:
        """
        Processes an audio file through Whisper and returns a standardized LectureContent object.
        Preserves segment timestamps (start, end) for downstream provenance citations.
        """
        valid_path = self.validate_audio_file(audio_path)
        audio_title = title or valid_path.stem.replace("_", " ").title()

        if progress_callback:
            progress_callback(0.10, "Initializing Whisper speech recognition model...")

        try:
            model = ModelLoader.get_whisper_model(self.model_name)
        except ModelLoadingError as e:
            raise AudioProcessingError(f"Speech recognition model unavailable: {e}") from e

        if progress_callback:
            progress_callback(0.30, f"Transcribing lecture audio from '{valid_path.name}'...")

        try:
            # Transcribe with Whisper (fp16=False for CPU inference)
            result = model.transcribe(
                str(valid_path),
                fp16=False,
                language=language,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed for {valid_path}: {e}")
            raise AudioProcessingError(
                f"Failed to transcribe audio recording. The file may be corrupted or unreadable. Details: {e}"
            ) from e

        if progress_callback:
            progress_callback(0.75, "Structuring transcript segments and timestamps...")

        raw_text = result.get("text", "")
        normalized_full_text = clean_text(raw_text)

        if not normalized_full_text:
            raise AudioProcessingError(
                "Audio transcription produced empty text. The recording may be silent, indistinct, or corrupted."
            )

        # Extract timestamped segments
        segments = []
        raw_segments = result.get("segments", [])
        
        if raw_segments:
            for idx, seg in enumerate(raw_segments):
                seg_text = clean_text(seg.get("text", ""))
                if seg_text:
                    segments.append(ContentSegment(
                        segment_id=idx,
                        text=seg_text,
                        source_type="audio",
                        start_time=float(seg.get("start", 0.0)),
                        end_time=float(seg.get("end", 0.0)),
                        metadata={
                            "temperature": seg.get("temperature", 0.0),
                            "avg_logprob": seg.get("avg_logprob", 0.0),
                            "no_speech_prob": seg.get("no_speech_prob", 0.0),
                        }
                    ))
        else:
            # Fallback single segment if Whisper returned flat text without segment breakdown
            segments.append(ContentSegment(
                segment_id=0,
                text=normalized_full_text,
                source_type="audio",
                start_time=0.0,
                end_time=0.0
            ))

        if progress_callback:
            progress_callback(0.90, "Generating transformer-sized content chunks...")

        # Build chunks respecting transformer input token boundaries
        chunks = build_chunks_from_segments(segments, normalized_full_text)

        if progress_callback:
            progress_callback(1.0, "Audio processing complete.")

        lecture_content = LectureContent(
            title=audio_title,
            source_type="audio",
            source_filename=valid_path.name,
            full_text=normalized_full_text,
            segments=segments,
            chunks=chunks,
            metadata={
                "detected_language": result.get("language", language),
                "whisper_model": self.model_name,
                "file_size_bytes": valid_path.stat().st_size,
                "duration_seconds": segments[-1].end_time if segments and segments[-1].end_time else None
            }
        )

        logger.info(
            f"Successfully processed audio '{valid_path.name}': "
            f"{len(segments)} segments, {len(chunks)} chunks, {lecture_content.total_words} words."
        )
        return lecture_content
