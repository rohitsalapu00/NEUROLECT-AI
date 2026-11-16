"""
NEUROLECT AI - File Management & FFmpeg Path Resolution
Handles safe file persistence, temporary file lifecycle, and FFmpeg path injection.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Union, Any
import logging
from config import TEMP_DIR, UPLOADS_DIR

logger = logging.getLogger("NEUROLECT.FileManager")


def ensure_ffmpeg_on_path() -> Optional[str]:
    """
    Locates FFmpeg executable and prepends its directory to os.environ['PATH'].
    Uses bundled imageio-ffmpeg on Windows or system FFmpeg.
    Returns the absolute path to ffmpeg.exe if found.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            bin_dir = os.path.dirname(ffmpeg_exe)
            standard_exe = os.path.join(bin_dir, "ffmpeg.exe")
            if not os.path.exists(standard_exe):
                try:
                    shutil.copy(ffmpeg_exe, standard_exe)
                except Exception as copy_err:
                    logger.debug(f"Could not duplicate to ffmpeg.exe: {copy_err}")

            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Appended FFmpeg binary directory to PATH: {bin_dir}")
            return standard_exe if os.path.exists(standard_exe) else ffmpeg_exe
    except Exception as e:
        logger.warning(f"Could not load imageio_ffmpeg: {e}")

    # Fallback check for system PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    return None


def save_upload_to_disk(
    uploaded_file: Any,
    filename: Optional[str] = None,
    destination_dir: Path = UPLOADS_DIR
) -> Path:
    """
    Saves an uploaded file (Streamlit UploadedFile, bytes, or file path) to destination_dir.
    Returns the absolute Path of the saved file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        if hasattr(uploaded_file, "name"):
            filename = uploaded_file.name
        else:
            filename = f"upload_{uuid.uuid4().hex[:8]}.bin"

    # Sanitize filename
    safe_filename = Path(filename).name
    target_path = destination_dir / safe_filename

    # If already exists, append unique suffix
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        target_path = destination_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"

    if hasattr(uploaded_file, "getbuffer"):
        # Streamlit UploadedFile
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, "read"):
        with open(target_path, "wb") as f:
            f.write(uploaded_file.read())
    elif isinstance(uploaded_file, bytes):
        with open(target_path, "wb") as f:
            f.write(uploaded_file)
    elif isinstance(uploaded_file, (str, Path)) and os.path.exists(uploaded_file):
        shutil.copy(uploaded_file, target_path)
    else:
        raise ValueError(f"Unsupported file object type: {type(uploaded_file)}")

    logger.info(f"Saved file to {target_path} ({target_path.stat().st_size} bytes)")
    return target_path


def cleanup_temp_file(file_path: Union[str, Path]) -> bool:
    """
    Safely deletes a temporary file. Catches and logs exceptions.
    Returns True if deleted or didn't exist, False if deletion failed.
    """
    try:
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
            logger.info(f"Cleaned up temporary file: {p}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete temp file {file_path}: {e}")
        return False


def get_temp_filepath(suffix: str, prefix: str = "temp_") -> Path:
    """Generates a unique temporary file path inside TEMP_DIR."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"{prefix}{uuid.uuid4().hex[:10]}{suffix}"
    return TEMP_DIR / unique_name
