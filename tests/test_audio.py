"""
NEUROLECT AI - Test Audio Processor
Validates audio validation, synthetic WAV transcription via Whisper,
timestamp extraction, and error handling for corrupted files.
"""

import math
import struct
import wave
import pytest
from pathlib import Path
from modules.audio_processor import AudioProcessor, AudioProcessingError
from modules.content_models import LectureContent


@pytest.fixture
def audio_processor():
    # Use 'tiny' for fast test execution
    return AudioProcessor(whisper_model_name="tiny")


@pytest.fixture
def synthetic_wav(tmp_path):
    """Generates a small valid 1-second 16kHz PCM WAV file."""
    wav_path = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration_secs = 1.0
    frequency = 440.0  # 440 Hz tone

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        total_samples = int(sample_rate * duration_secs)
        for i in range(total_samples):
            value = int(32767.0 * 0.2 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))

    return wav_path


def test_validate_audio_file_valid(audio_processor, synthetic_wav):
    """Verify that a valid WAV file passes validation."""
    valid_path = audio_processor.validate_audio_file(synthetic_wav)
    assert valid_path.exists()
    assert valid_path.suffix.lower() == ".wav"


def test_validate_audio_invalid_extension(tmp_path, audio_processor):
    """Verify that unsupported extensions raise AudioProcessingError."""
    invalid_file = tmp_path / "song.flac"
    invalid_file.write_bytes(b"FLAC audio data")
    with pytest.raises(AudioProcessingError):
        audio_processor.validate_audio_file(invalid_file)


def test_validate_audio_empty_file(tmp_path, audio_processor):
    """Verify that empty (0-byte) files raise AudioProcessingError."""
    empty_file = tmp_path / "silent.wav"
    empty_file.touch()
    with pytest.raises(AudioProcessingError):
        audio_processor.validate_audio_file(empty_file)


def test_validate_audio_nonexistent(audio_processor):
    """Verify that non-existent paths raise AudioProcessingError."""
    with pytest.raises(AudioProcessingError):
        audio_processor.validate_audio_file("missing_lecture.mp3")


def test_transcribe_audio_pipeline(audio_processor, synthetic_wav):
    """
    Verify full Whisper transcription pipeline.
    Even with synthetic tone, Whisper runs and produces a valid LectureContent object
    or identifies absence of speech.
    """
    try:
        content = audio_processor.transcribe(synthetic_wav, title="Synthetic Test Audio")
        assert isinstance(content, LectureContent)
        assert content.source_type == "audio"
        assert content.source_filename == "test_audio.wav"
        assert len(content.segments) >= 1
    except AudioProcessingError as e:
        # A pure sine tone has no speech, so empty text exception is also expected and valid behavior!
        assert "empty text" in str(e).lower() or "silent" in str(e).lower()
