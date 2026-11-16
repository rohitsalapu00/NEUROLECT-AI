"""
NEUROLECT AI - Video Processing Module
First-class video pipeline: extracts audio track using FFmpeg, transcribes via Whisper,
and standardizes into unified LectureContent.
Supported formats: MP4, MOV, MKV.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union
from config import SUPPORTED_VIDEO_FORMATS, MAX_UPLOAD_SIZE_MB, MODELS
from utils.validators import validate_file_extension, validate_file_size, ValidationError
from utils.file_manager import ensure_ffmpeg_on_path, get_temp_filepath, cleanup_temp_file
from modules.content_models import LectureContent, ContentSegment, build_chunks_from_segments
from modules.audio_processor import AudioProcessor, AudioProcessingError

logger = logging.getLogger("NEUROLECT.VideoProcessor")


class VideoProcessingError(Exception):
    """Raised when video file validation, FFmpeg extraction, or transcription fails."""
    pass


class VideoProcessor:
    """
    Extracts speech from video lecture recordings (MP4, MOV, MKV) using FFmpeg,
    then processes the audio through Whisper into unified LectureContent.
    Temporary extracted audio files are guaranteed to be cleaned up after processing.
    """

    def __init__(self, whisper_model_name: Optional[str] = None):
        self.whisper_model_name = whisper_model_name or MODELS.get("whisper", "base")
        self.ffmpeg_exe = ensure_ffmpeg_on_path()
        if not self.ffmpeg_exe:
            logger.warning("FFmpeg executable not immediately located. Video processing may require system FFmpeg.")

    def validate_video_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validates file existence, supported video extension, and size limits.
        """
        path = Path(file_path)
        if not path.exists():
            raise VideoProcessingError(f"Video file not found: {path}")

        try:
            validate_file_extension(path.name, SUPPORTED_VIDEO_FORMATS)
            validate_file_size(path, MAX_UPLOAD_SIZE_MB)
        except ValidationError as e:
            raise VideoProcessingError(str(e)) from e

        return path

    def extract_audio_from_video(
        self,
        video_path: Path,
        output_wav_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Path:
        """
        Uses FFmpeg to extract the audio stream from a video file as 16kHz mono 16-bit PCM WAV.
        Detects videos without audio tracks and handles corrupted media files.
        """
        ffmpeg_bin = self.ffmpeg_exe or ensure_ffmpeg_on_path()
        if not ffmpeg_bin:
            raise VideoProcessingError(
                "FFmpeg executable is missing. Please ensure FFmpeg is installed or imageio-ffmpeg is present."
            )

        target_wav = output_wav_path or get_temp_filepath(".wav", prefix="extracted_audio_")

        if progress_callback:
            progress_callback(0.20, f"Extracting audio track from video '{video_path.name}' with FFmpeg...")

        cmd = [
            str(ffmpeg_bin),
            "-y",                     # Overwrite existing destination
            "-i", str(video_path),    # Input video file
            "-vn",                    # Discard video frames (NLP-focused speech extraction)
            "-acodec", "pcm_s16le",   # Uncompressed 16-bit PCM WAV
            "-ar", "16000",           # 16 kHz sample rate (optimal for Whisper)
            "-ac", "1",               # Downmix to mono
            str(target_wav)
        ]

        logger.info(f"Running FFmpeg audio extraction: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
        except Exception as e:
            raise VideoProcessingError(f"FFmpeg execution failed: {e}") from e

        # Validate that extraction succeeded and audio track exists
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if "does not contain any stream" in stderr_lower or "matches no streams" in stderr_lower or "output file is empty" in stderr_lower:
                cleanup_temp_file(target_wav)
                raise VideoProcessingError(
                    f"Video file '{video_path.name}' does not contain an audio track. NEUROLECT requires spoken audio for lecture processing."
                )
            cleanup_temp_file(target_wav)
            raise VideoProcessingError(
                f"FFmpeg failed to extract audio from '{video_path.name}'. Details: {result.stderr.strip()[:300]}"
            )

        if not target_wav.exists() or target_wav.stat().st_size <= 44:
            cleanup_temp_file(target_wav)
            raise VideoProcessingError(
                f"Video file '{video_path.name}' has no audible audio or contains an empty audio track."
            )

        logger.info(f"Audio extracted successfully to {target_wav} ({target_wav.stat().st_size} bytes)")
        return target_wav

    def process(
        self,
        video_path: Union[str, Path],
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        language: Optional[str] = "en"
    ) -> LectureContent:
        """
        Complete video pipeline:
        1. Validates video file.
        2. Extracts audio to temporary WAV file via FFmpeg.
        3. Transcribes via Whisper preserving timestamps.
        4. Cleans up temporary audio file guaranteed via finally block.
        5. Returns unified LectureContent with source_type='video'.
        """
        valid_path = self.validate_video_file(video_path)
        video_title = title or valid_path.stem.replace("_", " ").title()
        temp_audio_path: Optional[Path] = None

        try:
            if progress_callback:
                progress_callback(0.05, f"Validating video file '{valid_path.name}'...")

            # 1. Extract audio track
            temp_audio_path = self.extract_audio_from_video(
                valid_path,
                progress_callback=progress_callback
            )

            # 2. Transcribe extracted audio
            if progress_callback:
                progress_callback(0.40, "Handing audio to Whisper transcription engine...")

            audio_processor = AudioProcessor(whisper_model_name=self.whisper_model_name)

            def sub_callback(pct: float, msg: str):
                if progress_callback:
                    # Map audio processor 0.0 - 1.0 to video processor 0.40 - 0.95
                    mapped = 0.40 + (pct * 0.55)
                    progress_callback(mapped, msg)

            lecture_content = audio_processor.transcribe(
                audio_path=temp_audio_path,
                title=video_title,
                progress_callback=sub_callback,
                language=language
            )

            # Re-tag provenance to video
            lecture_content.source_type = "video"
            lecture_content.source_filename = valid_path.name
            for seg in lecture_content.segments:
                seg.source_type = "video"
            
            lecture_content.metadata["original_video_size_bytes"] = valid_path.stat().st_size
            lecture_content.metadata["video_format"] = valid_path.suffix.lower()

            if progress_callback:
                progress_callback(1.0, "Video lecture processing complete.")

            return lecture_content

        except AudioProcessingError as ape:
            raise VideoProcessingError(f"Transcription error on video audio: {ape}") from ape
        finally:
            # 3. Clean up temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                cleanup_temp_file(temp_audio_path)
                logger.info(f"Cleaned up temporary extracted audio file: {temp_audio_path}")
