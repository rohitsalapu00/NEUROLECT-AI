"""
NEUROLECT AI - Test Video Processor
Validates video file validation, FFmpeg audio extraction, handling of videos without audio,
and guaranteed cleanup of temporary audio files.
"""

import subprocess
import pytest
from pathlib import Path
from modules.video_processor import VideoProcessor, VideoProcessingError
from modules.content_models import LectureContent
from utils.file_manager import ensure_ffmpeg_on_path, cleanup_temp_file
from config import TEMP_DIR


@pytest.fixture
def video_processor():
    return VideoProcessor(whisper_model_name="tiny")


@pytest.fixture
def synthetic_video_with_audio(tmp_path):
    """Generates a synthetic 1-second MP4 video containing an audible tone."""
    ffmpeg_exe = ensure_ffmpeg_on_path()
    video_path = tmp_path / "lecture_clip.mp4"
    cmd = [
        ffmpeg_exe,
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "mpeg4",
        "-c:a", "aac",
        "-y", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True)
    assert res.returncode == 0, f"FFmpeg synthetic video creation failed: {res.stderr}"
    return video_path


@pytest.fixture
def synthetic_silent_video(tmp_path):
    """Generates a synthetic 1-second MP4 video with NO audio track."""
    ffmpeg_exe = ensure_ffmpeg_on_path()
    video_path = tmp_path / "silent_video.mp4"
    cmd = [
        ffmpeg_exe,
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-c:v", "mpeg4",
        "-an",  # Explicitly disable audio stream
        "-y", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True)
    assert res.returncode == 0, f"FFmpeg silent video creation failed: {res.stderr}"
    return video_path


def test_validate_video_valid(video_processor, synthetic_video_with_audio):
    """Verify that a valid MP4 passes validation."""
    valid_path = video_processor.validate_video_file(synthetic_video_with_audio)
    assert valid_path.exists()
    assert valid_path.suffix.lower() == ".mp4"


def test_validate_video_invalid_extension(tmp_path, video_processor):
    """Verify that unsupported extensions raise VideoProcessingError."""
    bad_file = tmp_path / "movie.avi"
    bad_file.write_bytes(b"AVI payload")
    with pytest.raises(VideoProcessingError):
        video_processor.validate_video_file(bad_file)


def test_validate_video_empty_file(tmp_path, video_processor):
    """Verify that empty (0-byte) files raise VideoProcessingError."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    with pytest.raises(VideoProcessingError):
        video_processor.validate_video_file(empty_file)


def test_extract_audio_from_video_success(video_processor, synthetic_video_with_audio):
    """Verify that FFmpeg extracts a valid 16kHz PCM WAV from a video with audio."""
    extracted_wav = video_processor.extract_audio_from_video(synthetic_video_with_audio)
    try:
        assert extracted_wav.exists()
        assert extracted_wav.stat().st_size > 44  # Valid WAV header + data
    finally:
        cleanup_temp_file(extracted_wav)


def test_extract_audio_from_silent_video_raises(video_processor, synthetic_silent_video):
    """Verify that extracting audio from a silent video raises VideoProcessingError."""
    with pytest.raises(VideoProcessingError) as exc_info:
        video_processor.extract_audio_from_video(synthetic_silent_video)
    assert "audio track" in str(exc_info.value).lower() or "no audible audio" in str(exc_info.value).lower()


def test_process_video_cleanup_guarantee(video_processor, synthetic_video_with_audio):
    """Verify that temporary extracted audio files are cleaned up after video processing."""
    # List temp directory files before
    initial_temp_files = set(TEMP_DIR.glob("*.wav"))

    try:
        video_processor.process(synthetic_video_with_audio, title="Lecture Video Clip")
    except Exception:
        # Processing may raise empty speech if test audio is pure tone, but temp files MUST be cleaned up!
        pass

    # Verify no dangling extracted_audio_ files remain in TEMP_DIR
    post_temp_files = set(TEMP_DIR.glob("extracted_audio_*.wav"))
    assert len(post_temp_files) == 0, f"Dangling temp audio files found: {post_temp_files}"
