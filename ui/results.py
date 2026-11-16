"""
NEUROLECT AI - Assessment Results & Feedback View
Displays transparent automated grading outcomes, multi-criteria rubric breakdowns,
missing concept identification, source citations, and honest model fallback notices.
"""

import json
import streamlit as st
from typing import Optional
from modules.quiz_models import QuizResult
from database.db_manager import DatabaseManager


def render_results_page(db: DatabaseManager) -> None:
    """Renders the Assessment Results and Feedback interface."""
    st.header("📊 Assessment Results & Detailed Feedback")

    result: Optional[QuizResult] = st.session_state.get("latest_quiz_result")

    # If no active result, offer past attempts from database
    if not result:
        past_attempts = db.list_quiz_attempts(limit=20)
        if not past_attempts:
            st.info("No assessment results available. Please attempt a quiz first.")
            if st.button("📝 Go to Quiz Interface"):
                st.session_state.nav_page = "📝 Take Quiz"
                st.rerun()
            return

        st.markdown("Select a completed quiz attempt to review:")
        opt_map = {f"{a['quiz_title']} — {a['percentage']}% ({a['completed_at'][:16].replace('T', ' ')})": a["attempt_id"] for a in past_attempts}
        selected_label = st.selectbox("Historical Attempts:", list(opt_map.keys()))
        selected_attempt_id = opt_map[selected_label]

        attempt_data = db.get_quiz_attempt(selected_attempt_id)
        if attempt_data:
            _render_attempt_dict(attempt_data)
        return

    _render_quiz_result_object(result)


def _render_quiz_result_object(result: QuizResult) -> None:
    """Renders live QuizResult object."""
    pct = result.percentage
    theme_color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    status_label = "Mastery Demonstrated 🌟" if pct >= 80 else "Proficient Understanding 👍" if pct >= 60 else "Conceptual Review Recommended ⚠️"

    # Hero Result Banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme_color}18 0%, rgba(15, 23, 42, 0.4) 100%); border: 1px solid {theme_color}55; border-radius: 12px; padding: 1.75rem 2rem; margin-bottom: 1.5rem; text-align: center;">
        <span style="background: {theme_color}22; color: {theme_color}; border: 1px solid {theme_color}44; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.9rem;">
            {status_label}
        </span>
        <h1 style="font-size: 3rem; margin: 0.75rem 0 0.25rem 0; color: #f8fafc; font-weight: 800;">
            {pct}%
        </h1>
        <p style="font-size: 1.15rem; color: #cbd5e1; margin: 0;">
            Earned <strong>{result.earned_marks}</strong> out of <strong>{result.total_marks}</strong> maximum marks
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Outcome Counter Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='neuro-card' style='text-align: center;'><h3 style='color: #10b981; margin: 0;'>{result.correct_count}</h3><span style='color: #94a3b8;'>Correct</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='neuro-card' style='text-align: center;'><h3 style='color: #f59e0b; margin: 0;'>{result.partial_count}</h3><span style='color: #94a3b8;'>Partially Correct</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='neuro-card' style='text-align: center;'><h3 style='color: #ef4444; margin: 0;'>{result.incorrect_count}</h3><span style='color: #94a3b8;'>Incorrect</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='neuro-card' style='text-align: center;'><h3 style='color: #64748b; margin: 0;'>{result.unanswered_questions}</h3><span style='color: #94a3b8;'>Unanswered</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Transparency Alert Notice
    manual_review_items = [qr for qr in result.question_results if qr.evaluation_status == "requires_manual_review"]
    if manual_review_items:
        st.warning(
            f"⚠️ **Transparency Notice:** {len(manual_review_items)} response(s) were flagged for **Manual Faculty Review**. "
            f"This occurs when high lexical overlap diverges from semantic intent, or when conservative model fallbacks were applied. "
            f"Scores for these questions are conservatively capped to maintain assessment integrity."
        )

    # Detailed Question Results
    st.subheader("📝 Question-by-Question Diagnostic Review")

    for idx, qr in enumerate(result.question_results, 1):
        status = qr.evaluation_status
        badge_bg = (
            "#10b98122" if status == "correct" else
            "#f59e0b22" if status == "partially_correct" else
            "#f43f5e22" if status == "requires_manual_review" else
            "#ef444422" if status == "incorrect" else "#64748b22"
        )
        badge_border = (
            "#10b981" if status == "correct" else
            "#f59e0b" if status == "partially_correct" else
            "#f43f5e" if status == "requires_manual_review" else
            "#ef4444" if status == "incorrect" else "#64748b"
        )
        status_text = status.replace("_", " ").title()

        with st.expander(f"Question {idx}: {qr.question_text[:75]}... ({qr.marks_awarded}/{qr.max_marks} Marks — {status_text})", expanded=(status != "correct")):
            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <span style="background: {badge_bg}; color: {badge_border}; border: 1px solid {badge_border}; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem;">
                    {status_text}
                </span>
                <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 0.75rem;">
                    Type: {qr.question_type} • Bloom: {qr.bloom_level} • Difficulty: {qr.difficulty}
                </span>
            </div>
            <p style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 1rem;">
                {qr.question_text}
            </p>
            """, unsafe_allow_html=True)

            col_sub, col_ref = st.columns(2)
            with col_sub:
                st.markdown("**Your Submission:**")
                if qr.student_answer and qr.student_answer.strip():
                    st.info(qr.student_answer)
                else:
                    st.caption("*(No answer was provided — Skipped)*")

            with col_ref:
                st.markdown("**Lecture Ground-Truth Answer:**")
                st.success(qr.reference_answer)

            # Constructive Pedagogical Feedback
            st.markdown(f"**Automated Diagnostic Feedback:**")
            st.write(qr.feedback)

            # Missing Concepts
            if qr.missing_concepts:
                st.markdown("**Missing Key Concepts:**")
                badges = " ".join([
                    f"<span style='background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 9999px; padding: 0.15rem 0.6rem; font-size: 0.8rem; margin: 0.15rem;'>{mc}</span>"
                    for mc in qr.missing_concepts[:8]
                ])
                st.markdown(badges, unsafe_allow_html=True)

            # Rubric Breakdown
            if qr.score_breakdown:
                with st.expander("🔍 Rubric Component Metrics", expanded=False):
                    st.json(qr.score_breakdown)

            st.caption(f"Verifiable Source Citation: {qr.source_reference}")

    # Export Report Button
    st.markdown("---")
    res_json = json.dumps(result.to_dict(), indent=2)
    st.download_button(
        label="📥 Export Assessment Diagnostic Report (JSON)",
        data=res_json,
        file_name=f"assessment_report_{result.quiz_id}.json",
        mime="application/json"
    )


def _render_attempt_dict(attempt: dict) -> None:
    """Renders a historical attempt retrieved from database dictionary."""
    pct = attempt["percentage"]
    theme_color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme_color}18 0%, rgba(15, 23, 42, 0.4) 100%); border: 1px solid {theme_color}55; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
        <h2 style="font-size: 2.5rem; margin: 0; color: #f8fafc; font-weight: 800;">{pct}%</h2>
        <p style="color: #cbd5e1; margin-top: 0.25rem;">Earned <strong>{attempt['earned_marks']}</strong> / <strong>{attempt['total_marks']}</strong> marks • {attempt['quiz_title']}</p>
    </div>
    """, unsafe_allow_html=True)

    answers = attempt.get("answers", [])
    st.subheader(f"Historical Question Log ({len(answers)} Questions)")
    for idx, ans in enumerate(answers, 1):
        with st.expander(f"{idx}. {ans['question_text'][:80]}... ({ans['marks_awarded']}/{ans['max_marks']} Marks)", expanded=False):
            st.markdown(f"**Submitted:** {ans['student_answer'] or '*(Unanswered)*'}")
            st.markdown(f"**Reference:** {ans['reference_answer']}")
            st.markdown(f"**Feedback:** {ans['feedback']}")
            st.caption(f"Source: {ans['source_reference']}")
