"""
NEUROLECT AI - Intelligent Lecture Understanding, Question Generation & Automated Assessment System
Streamlit Main Application Entry Point
"""

import os
import streamlit as st
from pathlib import Path

# Ensure config directories and environment variables are initialized first
from config import APP_NAME, APP_TAGLINE, APP_VERSION, ASSETS_DIR
from database.db_manager import DatabaseManager
from models.model_loader import ModelLoader

# UI Page Views
from ui.home import render_home_page
from ui.upload import render_upload_page
from ui.understanding import render_understanding_page
from ui.questions import render_questions_page
from ui.quiz import render_quiz_page
from ui.results import render_results_page
from ui.analytics import render_analytics_page
from ui.history import render_history_page


def inject_custom_styles() -> None:
    """Injects custom CSS from assets/style.css into the Streamlit app."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception:
            pass


def init_session_state() -> None:
    """Initializes persistent application session state variables."""
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Home Dashboard"
    if "active_lecture" not in st.session_state:
        st.session_state.active_lecture = None
    if "active_lecture_id" not in st.session_state:
        st.session_state.active_lecture_id = None
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []
    if "latest_quiz_result" not in st.session_state:
        st.session_state.latest_quiz_result = None


def main():
    st.set_page_config(
        page_title=f"{APP_NAME} — Lecture Understanding & Assessment",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_styles()
    init_session_state()

    # Initialize Database Connection
    db = DatabaseManager()

    # =========================================================================
    # SIDEBAR NAVIGATION & SYSTEM MONITOR
    # =========================================================================
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 0.5rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.25rem;">🧠</div>
            <h2 style="margin: 0; font-weight: 800; background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {APP_NAME}
            </h2>
            <p style="font-size: 0.75rem; color: #94a3b8; margin: 0.25rem 0 0 0;">
                v{APP_VERSION} • CPU-First NLP
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Active Lecture Status Pill
        active_lec = st.session_state.get("active_lecture")
        if active_lec:
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; color: #818cf8; font-weight: 700; text-transform: uppercase;">Active Lecture</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {active_lec.title}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                    {active_lec.source_type.upper()} • {active_lec.total_words} words
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem; text-align: center;">
                <div style="font-size: 0.8rem; color: #94a3b8;">No Active Lecture Loaded</div>
            </div>
            """, unsafe_allow_html=True)

        # Primary Navigation Radio
        nav_options = [
            "🏠 Home Dashboard",
            "📤 Upload Lecture",
            "🧠 Lecture Understanding",
            "❓ Question Generator",
            "📝 Take Quiz",
            "📊 Assessment Results",
            "📈 Learning Analytics",
            "📜 Quiz History"
        ]

        current_nav = st.session_state.get("nav_page", "🏠 Home Dashboard")
        if current_nav not in nav_options:
            current_nav = "🏠 Home Dashboard"

        chosen_page = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(current_nav),
            label_visibility="collapsed"
        )
        st.session_state.nav_page = chosen_page

        st.markdown("---")

        # System Health Footer
        mem_info = ModelLoader.get_memory_status()
        loaded_count = mem_info.get("loaded_models_count", 0)
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5;">
            <div><strong>Mode:</strong> CPU Offline Inference</div>
            <div><strong>RAG:</strong> False (Zero Vector DB)</div>
            <div><strong>Cached Models:</strong> {loaded_count}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # PAGE ROUTING
    # =========================================================================
    page = st.session_state.nav_page

    if page == "🏠 Home Dashboard":
        render_home_page(db)
    elif page == "📤 Upload Lecture":
        render_upload_page(db)
    elif page == "🧠 Lecture Understanding":
        render_understanding_page(db)
    elif page == "❓ Question Generator":
        render_questions_page(db)
    elif page == "📝 Take Quiz":
        render_quiz_page(db)
    elif page == "📊 Assessment Results":
        render_results_page(db)
    elif page == "📈 Learning Analytics":
        render_analytics_page(db)
    elif page == "📜 Quiz History":
        render_history_page(db)
    else:
        render_home_page(db)


if __name__ == "__main__":
    main()
