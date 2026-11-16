"""
NEUROLECT AI - Home Dashboard View
Displays application overview, key performance indicators, recent activity,
quick-start navigation, and system runtime health.
"""

import streamlit as st
from database.db_manager import DatabaseManager
from models.model_loader import ModelLoader
from config import APP_NAME, APP_TAGLINE, APP_VERSION


def render_home_page(db: DatabaseManager) -> None:
    """Renders the Home Dashboard page."""
    # Hero Section
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🧠 {APP_NAME}</div>
        <p style="font-size: 1.15rem; color: #94a3b8; margin-top: 0.5rem; margin-bottom: 0;">
            {APP_TAGLINE}
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.3);">
                Version {APP_VERSION}
            </span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3); margin-left: 0.5rem;">
                CPU-First & RAG-Free
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch Real Statistics from SQLite
    stats = db.get_analytics_summary()

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📚 Lectures Processed",
            value=stats["total_lectures"],
            help="Total audio, video, PDF, and text lectures stored in database."
        )

    with col2:
        st.metric(
            label="❓ Questions Generated",
            value=stats["total_questions"],
            help="Total transformer-generated questions with ground-truth citations."
        )

    with col3:
        st.metric(
            label="📝 Quizzes Attempted",
            value=stats["total_quizzes_attempted"],
            help="Total interactive assessment sessions completed."
        )

    with col4:
        avg_score = stats["average_score_percentage"]
        st.metric(
            label="🎯 Average Score",
            value=f"{avg_score}%" if avg_score is not None else "N/A",
            help="Mean performance percentage across all completed quiz attempts."
        )

    st.markdown("---")

    # Quick Start Action Cards
    st.subheader("⚡ Quick Actions")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    with qcol1:
        st.markdown("""
        <div class="neuro-card" style="text-align: center;">
            <div style="font-size: 2rem;">📤</div>
            <h4 style="margin: 0.5rem 0;">Upload Lecture</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Process PDF, Audio, Video, or Text into unified lecture content.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload ➡️", key="btn_quick_upload", use_container_width=True):
            st.session_state.nav_page = "📤 Upload Lecture"
            st.rerun()

    with qcol2:
        st.markdown("""
        <div class="neuro-card" style="text-align: center;">
            <div style="font-size: 2rem;">🧠</div>
            <h4 style="margin: 0.5rem 0;">Lecture Insights</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Extract summaries, topics, keywords, entities, and definitions.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Insights ➡️", key="btn_quick_understand", use_container_width=True):
            st.session_state.nav_page = "🧠 Lecture Understanding"
            st.rerun()

    with qcol3:
        st.markdown("""
        <div class="neuro-card" style="text-align: center;">
            <div style="font-size: 2rem;">❓</div>
            <h4 style="margin: 0.5rem 0;">Generate Questions</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Synthesize MCQs and open-ended questions grounded in source.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Questions ➡️", key="btn_quick_questions", use_container_width=True):
            st.session_state.nav_page = "❓ Question Generator"
            st.rerun()

    with qcol4:
        st.markdown("""
        <div class="neuro-card" style="text-align: center;">
            <div style="font-size: 2rem;">📝</div>
            <h4 style="margin: 0.5rem 0;">Take Assessment</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Attempt randomized interactive quizzes with automated grading.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Quiz ➡️", key="btn_quick_quiz", use_container_width=True):
            st.session_state.nav_page = "📝 Take Quiz"
            st.rerun()

    st.markdown("---")

    # Recent Activity & System Status Split
    act_col, sys_col = st.columns([3, 2])

    with act_col:
        st.subheader("🕒 Recent Activity")
        recent_lectures = stats["recent_lectures"]
        recent_attempts = stats["recent_attempts"]

        if not recent_lectures and not recent_attempts:
            st.info("No activity recorded yet. Start by uploading a lecture recording, PDF, or text notes!")
        else:
            if recent_lectures:
                st.markdown("##### 📚 Recent Lectures")
                for lec in recent_lectures:
                    created_date = lec["created_at"][:10] if lec.get("created_at") else ""
                    st.markdown(f"""
                    <div style="padding: 0.75rem; margin-bottom: 0.5rem; background: rgba(255, 255, 255, 0.03); border-left: 3px solid #6366f1; border-radius: 4px;">
                        <strong>{lec['title']}</strong> <span style="color: #94a3b8; font-size: 0.85rem;">({lec['source_type'].upper()} • {lec['num_words']} words • {created_date})</span>
                    </div>
                    """, unsafe_allow_html=True)

            if recent_attempts:
                st.markdown("##### 🎯 Recent Quiz Attempts")
                for att in recent_attempts:
                    p = att["percentage"]
                    badge_color = "#10b981" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444"
                    att_date = att["completed_at"][:16].replace("T", " ") if att.get("completed_at") else ""
                    st.markdown(f"""
                    <div style="padding: 0.75rem; margin-bottom: 0.5rem; background: rgba(255, 255, 255, 0.03); border-left: 3px solid {badge_color}; border-radius: 4px;">
                        <strong>{att['quiz_title']}</strong>: <span style="color: {badge_color}; font-weight: bold;">{p}%</span> 
                        <span style="color: #94a3b8; font-size: 0.85rem;">({att['earned_marks']}/{att['total_marks']} marks • {att_date})</span>
                    </div>
                    """, unsafe_allow_html=True)

    with sys_col:
        st.subheader("🖥️ Architecture & System Health")
        mem_info = ModelLoader.get_memory_status()
        loaded_count = mem_info.get("loaded_models_count", 0)
        models_loaded = mem_info.get("models_loaded", [])

        st.markdown(f"""
        <div class="neuro-card">
            <h4 style="margin-top: 0;">System Profile</h4>
            <ul style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; padding-left: 1.25rem;">
                <li><strong>Execution Device:</strong> CPU-Optimized</li>
                <li><strong>Architecture:</strong> RAG-Free Direct Transformers</li>
                <li><strong>Database:</strong> SQLite Persistent Storage</li>
                <li><strong>Active Models Loaded:</strong> {loaded_count}</li>
            </ul>
            <div style="margin-top: 0.75rem;">
                <span style="font-size: 0.85rem; color: #94a3b8;">Models in Cache:</span><br>
                <code style="font-size: 0.8rem; color: #a5b4fc;">{', '.join(models_loaded) if models_loaded else 'None (Lazy on-demand)'}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if loaded_count > 0:
            if st.button("🧹 Release Unused Models", use_container_width=True):
                ModelLoader.unload_all()
                st.success("All cached model weights successfully unloaded from memory.")
                st.rerun()
