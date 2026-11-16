"""
NEUROLECT AI - Interactive Quiz Administration View
Provides interactive test-taking interface with stateful question navigation,
randomized options, progress tracking, and strict submission lifecycle controls.
"""

import streamlit as st
from typing import List, Optional
from modules.question_models import GeneratedQuestion
from modules.quiz_models import Quiz, QuizQuestion, QuizResult
from modules.quiz_engine import QuizEngine, QuizEngineError
from database.db_manager import DatabaseManager


def render_quiz_page(db: DatabaseManager) -> None:
    """Renders the interactive Quiz Taking interface."""
    st.header("📝 Interactive Lecture Assessment")

    # Initialize Engine in session state
    if "quiz_engine" not in st.session_state:
        st.session_state.quiz_engine = QuizEngine()

    engine: QuizEngine = st.session_state.quiz_engine

    # =========================================================================
    # 1. QUIZ SETUP SCREEN (If no active quiz in session)
    # =========================================================================
    if not engine.active_quiz:
        st.markdown(
            "Configure your personalized quiz assessment. Questions are sampled from the active lecture "
            "or selected from your questions library."
        )

        candidate_questions = st.session_state.get("quiz_source_questions") or st.session_state.get("generated_questions") or []

        if not candidate_questions:
            # Check database for questions
            candidate_questions = db.list_all_questions(limit=50)

        if not candidate_questions:
            st.warning("⚠️ No assessment questions found. Please generate questions first.")
            if st.button("❓ Go to Question Generator", type="primary"):
                st.session_state.nav_page = "❓ Question Generator"
                st.rerun()
            return

        st.info(f"Available questions in pool: **{len(candidate_questions)}**")

        with st.form("quiz_config_form"):
            quiz_title = st.text_input(
                "Quiz Title",
                value=f"Assessment: {st.session_state.get('active_lecture', {}).title if hasattr(st.session_state.get('active_lecture'), 'title') else 'Lecture Assessment'}"
            )
            c1, c2 = st.columns(2)
            with c1:
                max_q_avail = min(20, len(candidate_questions))
                if max_q_avail > 1:
                    q_count = st.slider("Number of Questions in Quiz", 1, max_q_avail, min(5, max_q_avail))
                else:
                    q_count = 1
                    st.caption("Number of Questions in Quiz: **1**")
                randomize_q = st.checkbox("Randomize Question Order", value=True)
            with c2:
                randomize_opts = st.checkbox("Randomize MCQ Options", value=True)
                custom_seed = st.number_input("Randomization Seed (0 for random)", min_value=0, max_value=9999, value=42)

            start_btn = st.form_submit_button("🚀 Launch Interactive Quiz", type="primary", use_container_width=True)

        if start_btn:
            try:
                seed = int(custom_seed) if custom_seed > 0 else None
                quiz = engine.create_quiz(
                    questions=candidate_questions,
                    num_questions=q_count,
                    title=quiz_title,
                    randomize_questions=randomize_q,
                    randomize_options=randomize_opts,
                    seed=seed
                )
                engine.start_quiz(quiz)
                st.session_state.quiz_current_idx = 0
                st.session_state.quiz_answers = {}
                db.save_quiz(quiz, lecture_id=st.session_state.get("active_lecture_id"))
                st.rerun()
            except QuizEngineError as qe:
                st.error(f"Quiz Setup Error: {qe}")

        return

    # =========================================================================
    # 2. ACTIVE QUIZ ADMINISTRATION SCREEN
    # =========================================================================
    quiz: Quiz = engine.active_quiz
    total_q = len(quiz.questions)
    curr_idx = st.session_state.get("quiz_current_idx", 0)
    curr_idx = max(0, min(total_q - 1, curr_idx))

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    curr_q: QuizQuestion = quiz.questions[curr_idx]

    # Progress and Header
    progress_val = (curr_idx + 1) / total_q
    st.progress(progress_val, text=f"Question {curr_idx + 1} of {total_q}")

    col_meta1, col_meta2 = st.columns([3, 1])
    with col_meta1:
        st.subheader(f"Question {curr_idx + 1}")
    with col_meta2:
        st.markdown(f"<div style='text-align: right; color: #94a3b8;'><strong>{curr_q.max_marks} Mark{'s' if curr_q.max_marks != 1 else ''}</strong> | {curr_q.difficulty}</div>", unsafe_allow_html=True)

    # Question Card
    type_badge = "#3b82f6" if curr_q.question_type == "MCQ" else "#10b981" if "Short" in curr_q.question_type else "#8b5cf6"
    st.markdown(f"""
    <div class="neuro-card" style="margin-bottom: 1.5rem;">
        <div style="margin-bottom: 0.5rem;">
            <span style="background: {type_badge}22; color: {type_badge}; border: 1px solid {type_badge}44; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">
                {curr_q.question_type}
            </span>
            <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.5rem;">
                Bloom: {curr_q.bloom_level} • Topic: {curr_q.topic}
            </span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #f8fafc; line-height: 1.5;">
            {curr_q.question_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input Control by Question Type
    existing_ans = st.session_state.quiz_answers.get(curr_q.question_id, "")

    if curr_q.question_type == "MCQ" and curr_q.options:
        opt_labels = [f"[{chr(ord('A') + i)}] {opt}" for i, opt in enumerate(curr_q.options)]
        default_index = None

        if existing_ans:
            for idx, opt in enumerate(curr_q.options):
                if opt.strip() == existing_ans.strip() or f"[{chr(ord('A') + idx)}]" in existing_ans:
                    default_index = idx
                    break

        selected_opt = st.radio(
            "Select your answer:",
            options=opt_labels,
            index=default_index,
            key=f"mcq_{curr_q.question_id}"
        )
        if selected_opt:
            chosen_clean = selected_opt.split("] ", 1)[-1]
            st.session_state.quiz_answers[curr_q.question_id] = chosen_clean
            engine.submit_answer(curr_q.question_id, chosen_clean)

    elif curr_q.question_type in ["Short Answer", "SHORT ANSWER"]:
        user_text = st.text_input(
            "Your concise answer:",
            value=existing_ans,
            placeholder="Type your response here...",
            key=f"sa_{curr_q.question_id}"
        )
        if user_text != existing_ans:
            st.session_state.quiz_answers[curr_q.question_id] = user_text
            engine.submit_answer(curr_q.question_id, user_text)

    else:
        user_text = st.text_area(
            "Your comprehensive explanation:",
            value=existing_ans,
            height=140,
            placeholder="Structure your answer clearly, including relevant concepts and technical principles...",
            key=f"desc_{curr_q.question_id}"
        )
        if user_text != existing_ans:
            st.session_state.quiz_answers[curr_q.question_id] = user_text
            engine.submit_answer(curr_q.question_id, user_text)

    st.markdown("---")

    # Navigation Controls
    btn_prev, btn_skip, btn_next, btn_finish = st.columns([1, 1, 1, 1.2])

    with btn_prev:
        if st.button("⬅️ Previous", disabled=(curr_idx == 0), use_container_width=True):
            st.session_state.quiz_current_idx = curr_idx - 1
            st.rerun()

    with btn_skip:
        if st.button("⏭️ Skip Question", use_container_width=True):
            engine.skip_question(curr_q.question_id)
            st.session_state.quiz_answers[curr_q.question_id] = ""
            if curr_idx < total_q - 1:
                st.session_state.quiz_current_idx = curr_idx + 1
            st.rerun()

    with btn_next:
        if st.button("Next ➡️", disabled=(curr_idx >= total_q - 1), use_container_width=True):
            st.session_state.quiz_current_idx = curr_idx + 1
            st.rerun()

    with btn_finish:
        finish_confirmed = st.button("🏁 Finish Quiz", type="primary", use_container_width=True)

    # Submission Logic
    if finish_confirmed:
        answered_count = sum(1 for a in st.session_state.quiz_answers.values() if a and a.strip())
        unanswered_count = total_q - answered_count

        if unanswered_count > 0:
            st.warning(f"You have **{unanswered_count}** unanswered question(s). Submitting now will grade them as zero.")

        with st.spinner("Automated assessment engine is evaluating your answers..."):
            try:
                result = engine.finish_quiz()
                st.session_state.latest_quiz_result = result
                # Persist to database
                db.save_quiz_attempt(result, lecture_id=st.session_state.get("active_lecture_id"))

                st.session_state.quiz_answers = {}
                st.session_state.quiz_current_idx = 0

                st.success("Quiz completed! Redirecting to your assessment results...")
                st.session_state.nav_page = "📊 Assessment Results"
                st.rerun()

            except QuizEngineError as qe:
                st.error(f"Quiz Finalization Error: {qe}")
