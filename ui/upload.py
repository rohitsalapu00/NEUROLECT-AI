"""
NEUROLECT AI - Lecture Upload & Ingestion View
Supports multi-modal lecture input: PDF documents, plain text, audio recordings,
and video files with transcription, provenance tracking, and temporary file cleanup.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional

from config import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_DOC_FORMATS,
    MAX_UPLOAD_SIZE_MB
)
from utils.file_manager import save_upload_to_disk, cleanup_temp_file
from utils.validators import validate_file_size, validate_text_content, ValidationError
from modules.pdf_processor import PDFProcessor
from modules.text_processor import TextProcessor, TextProcessingError
from modules.audio_processor import AudioProcessor
from modules.video_processor import VideoProcessor
from modules.content_models import LectureContent
from database.db_manager import DatabaseManager


def render_upload_page(db: DatabaseManager) -> None:
    """Renders the Lecture Upload interface."""
    st.header("📤 Lecture Upload & Content Ingestion")
    st.markdown(
        "Upload audio lectures, video recordings, PDF lecture notes, or paste text. "
        "All modalities are converted into a standardized, RAG-free `LectureContent` representation."
    )

    tab_upload, tab_paste, tab_library = st.tabs([
        "📁 File Upload (Audio / Video / PDF / TXT)",
        "✍️ Paste Lecture Text",
        "📚 Lecture Library (Saved Lectures)"
    ])

    # =========================================================================
    # TAB 1: FILE UPLOAD
    # =========================================================================
    with tab_upload:
        all_exts = [ext.lstrip(".") for ext in (SUPPORTED_DOC_FORMATS + SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS)]
        uploaded_file = st.file_uploader(
            f"Choose a lecture file (Max {MAX_UPLOAD_SIZE_MB}MB)",
            type=all_exts,
            help="Supported: PDF, TXT, MP3, WAV, M4A, MP4, MOV, MKV"
        )

        col_title, col_btn = st.columns([3, 1])
        with col_title:
            custom_title = st.text_input("Lecture Title (Optional)", placeholder="Leave blank to use filename")

        if uploaded_file is not None:
            file_ext = Path(uploaded_file.name).suffix.lower()
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB, format: `{file_ext}`)")

            if st.button("🚀 Process & Transcribe Lecture", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Saving uploaded file to disk...")
                status_placeholder = st.empty()
                saved_temp_path: Optional[Path] = None

                try:
                    # 1. Save upload to disk
                    saved_temp_path = save_upload_to_disk(uploaded_file)
                    progress_bar.progress(20, text="Validating file integrity...")

                    # 2. Validation
                    validate_file_size(saved_temp_path, max_mb=MAX_UPLOAD_SIZE_MB)

                    lecture_title = custom_title.strip() if custom_title.strip() else Path(uploaded_file.name).stem.replace("_", " ").title()

                    # 3. Route to dedicated processor
                    content: Optional[LectureContent] = None

                    def _progress_callback(pct: float, msg: str):
                        normalized = pct if pct <= 1.0 else pct / 100.0
                        clamped_pct = max(0.2, min(0.95, 0.2 + 0.75 * normalized))
                        progress_bar.progress(clamped_pct, text=f"{msg} ({int(normalized * 100)}%)")

                    if file_ext in SUPPORTED_DOC_FORMATS:
                        if file_ext == ".pdf":
                            progress_bar.progress(40, text="Extracting text and page boundaries with PyMuPDF...")
                            content = PDFProcessor().extract_text(str(saved_temp_path), title=lecture_title)
                        else:
                            progress_bar.progress(50, text="Normalizing plain text content...")
                            content = TextProcessor().process_txt_file(
                                str(saved_temp_path),
                                title=lecture_title,
                                progress_callback=_progress_callback
                            )
                        if hasattr(uploaded_file, "name"):
                            content.source_filename = uploaded_file.name

                    elif file_ext in SUPPORTED_AUDIO_FORMATS:
                        progress_bar.progress(30, text="Loading Whisper model and transcribing audio...")
                        content = AudioProcessor().process_audio(
                            str(saved_temp_path),
                            title=lecture_title,
                            progress_callback=_progress_callback
                        )

                    elif file_ext in SUPPORTED_VIDEO_FORMATS:
                        progress_bar.progress(25, text="Extracting audio track via FFmpeg...")
                        content = VideoProcessor().process_video(
                            str(saved_temp_path),
                            title=lecture_title,
                            progress_callback=_progress_callback
                        )

                    else:
                        raise ValidationError(f"Unsupported file extension: {file_ext}")

                    # 4. Finalize & Validate Content
                    progress_bar.progress(95, text="Verifying lexical sanity...")
                    validate_text_content(content.full_text, min_words=5)

                    # 5. Persist to SQLite Database
                    lec_id = db.save_lecture(content)
                    content.metadata["lecture_id"] = lec_id

                    # 6. Set Active Session State
                    st.session_state.active_lecture = content
                    st.session_state.active_lecture_id = lec_id

                    progress_bar.progress(100, text="Processing complete!")
                    st.success(f"Successfully processed lecture: **{content.title}**!")
                    _display_content_summary(content)

                except (ValidationError, TextProcessingError) as ve:
                    st.error(f"Validation Error: {ve}")
                except Exception as e:
                    st.error(f"Processing Error: An unexpected error occurred while processing the file: {e}")
                finally:
                    # Clean up temporary upload file if audio/video
                    if saved_temp_path and saved_temp_path.exists():
                        cleanup_temp_file(saved_temp_path)

    # =========================================================================
    # TAB 2: PASTE TEXT
    # =========================================================================
    with tab_paste:
        paste_title = st.text_input("Lecture Title", placeholder="e.g. Introduction to Neural Networks", key="paste_title")
        paste_text = st.text_area(
            "Lecture Transcript / Study Notes",
            height=280,
            placeholder="Paste your lecture notes, article, or transcribed content here (minimum 5 words)...",
            key="paste_text"
        )

        if st.button("📥 Ingest Pasted Text", type="primary", use_container_width=True):
            if not paste_text.strip():
                st.warning("Please enter lecture text content before submitting.")
            else:
                try:
                    title = paste_title.strip() if paste_title.strip() else "Untitled Lecture Notes"
                    content = TextProcessor().process_raw_text(paste_text, title=title)
                    validate_text_content(content.full_text, min_words=5)

                    lec_id = db.save_lecture(content)
                    content.metadata["lecture_id"] = lec_id

                    st.session_state.active_lecture = content
                    st.session_state.active_lecture_id = lec_id

                    st.success(f"Successfully ingested lecture: **{content.title}**!")
                    _display_content_summary(content)

                except (ValidationError, TextProcessingError) as ve:
                    st.error(f"Validation Error: {ve}")
                except Exception as e:
                    st.error(f"Processing Error: {e}")

    # =========================================================================
    # TAB 3: SAVED LECTURES LIBRARY
    # =========================================================================
    with tab_library:
        saved_lectures = db.list_lectures()

        if not saved_lectures:
            st.info("No saved lectures found in database. Upload or paste a lecture to get started.")
        else:
            st.markdown(f"Found **{len(saved_lectures)}** saved lecture(s) in the database.")

            options = {f"{l['title']} ({l['source_type'].upper()} • {l['num_words']} words • {l['created_at'][:10]})": l["lecture_id"] for l in saved_lectures}
            selected_label = st.selectbox("Select a lecture to load into active session:", list(options.keys()))

            if st.button("📂 Load Selected Lecture", use_container_width=True):
                selected_id = options[selected_label]
                loaded_lec = db.get_lecture(selected_id)
                if loaded_lec:
                    st.session_state.active_lecture = loaded_lec
                    st.session_state.active_lecture_id = selected_id
                    st.success(f"Active lecture set to: **{loaded_lec.title}**!")
                    _display_content_summary(loaded_lec)
                else:
                    st.error("Could not load the selected lecture.")


def _display_content_summary(content: LectureContent) -> None:
    """Helper displaying metrics and content preview for active lecture."""
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Words", content.total_words)
    with m2:
        st.metric("Segments Extracted", content.total_segments)
    with m3:
        st.metric("Input Modality", content.source_type.upper())
    with m4:
        st.metric("Transformer Chunks", content.total_chunks)

    with st.expander("📖 View Extracted Lecture Text Preview", expanded=False):
        st.text_area("Full Extracted Content", content.full_text, height=200, disabled=True)

    c_next1, c_next2 = st.columns(2)
    with c_next1:
        if st.button("➡️ Proceed to Lecture Understanding", use_container_width=True):
            st.session_state.nav_page = "🧠 Lecture Understanding"
            st.rerun()
    with c_next2:
        if st.button("➡️ Proceed to Question Generator", use_container_width=True):
            st.session_state.nav_page = "❓ Question Generator"
            st.rerun()
