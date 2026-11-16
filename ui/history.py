"""
NEUROLECT AI - Quiz History View
Displays complete chronological audit logs of all past quiz submissions,
detailed answers, evaluation feedback, and rubric metrics stored in SQLite.
"""

import streamlit as st
from database.db_manager import DatabaseManager


def render_history_page(db: DatabaseManager) -> None:
    """Renders the Quiz History and Audit Log page."""
    st.header("📜 Quiz Submission History & Audit Log")
    st.markdown(
        "Review historical assessment sessions, inspect student submissions, "
        "and audit automated grading transparency."
    )

    attempts = db.list_quiz_attempts(limit=100)

    if not attempts:
        st.info("No assessment sessions recorded in database history yet.")
        if st.button("📝 Start an Assessment Quiz"):
            st.session_state.nav_page = "📝 Take Quiz"
            st.rerun()
        return

    st.markdown(f"Total Completed Sessions: **{len(attempts)}**")

    # Table of past attempts
    table_data = []
    for a in attempts:
        table_data.append({
            "Attempt ID": a["attempt_id"],
            "Quiz Title": a["quiz_title"],
            "Date Completed": a["completed_at"][:16].replace("T", " "),
            "Score": f"{a['earned_marks']} / {a['total_marks']}",
            "Percentage": f"{a['percentage']}%",
            "Correct": a["correct_count"],
            "Partial": a["partial_count"],
            "Incorrect": a["incorrect_count"],
            "Unanswered": a["unanswered_questions"],
            "Status": "Complete" if a["is_complete"] else "Incomplete"
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Detailed Attempt Inspector")

    attempt_options = {f"{a['quiz_title']} ({a['percentage']}%) — {a['completed_at'][:16].replace('T', ' ')}": a["attempt_id"] for a in attempts}
    chosen_label = st.selectbox("Select an attempt to inspect detailed answers and feedback:", list(attempt_options.keys()))

    if chosen_label:
        chosen_id = attempt_options[chosen_label]
        full_attempt = db.get_quiz_attempt(chosen_id)

        if full_attempt:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Marks", f"{full_attempt['earned_marks']} / {full_attempt['total_marks']}")
            with c2:
                st.metric("Percentage", f"{full_attempt['percentage']}%")
            with c3:
                st.metric("Questions Answered", f"{full_attempt['answered_questions']} / {full_attempt['total_questions']}")
            with c4:
                st.metric("Completion Status", "Complete" if full_attempt["is_complete"] else "Incomplete")

            st.markdown("##### 📝 Questions and Graded Answers")
            for idx, ans in enumerate(full_attempt.get("answers", []), 1):
                status = ans["evaluation_status"]
                badge_color = "#10b981" if status == "correct" else "#f59e0b" if status == "partially_correct" else "#ef4444"

                with st.expander(f"{idx}. {ans['question_text'][:80]}... ({ans['marks_awarded']}/{ans['max_marks']} Marks)", expanded=False):
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">
                            {status.replace('_', ' ').title()}
                        </span>
                        <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.5rem;">
                            Type: {ans['question_type']} • Source: {ans['source_reference']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Question:** {ans['question_text']}")
                    st.markdown(f"**Student Answer:** {ans['student_answer'] or '*(No answer provided)*'}")
                    st.markdown(f"**Ground-Truth Reference:** {ans['reference_answer']}")
                    st.markdown(f"**Pedagogical Feedback:** {ans['feedback']}")

                    if ans.get("missing_concepts"):
                        st.caption(f"Missing Concepts: {', '.join(ans['missing_concepts'])}")

    st.markdown("---")
    with st.expander("⚠️ Database Maintenance & Privacy Controls"):
        st.warning("Clearing all data will permanently delete all stored lectures, questions, and historical quiz attempts.")
        if st.button("🗑️ Clear All Historical Database Records"):
            db.clear_all_data()
            st.session_state.active_lecture = None
            st.session_state.active_lecture_id = None
            st.session_state.generated_questions = []
            st.session_state.latest_quiz_result = None
            st.success("All historical database records cleared.")
            st.rerun()
