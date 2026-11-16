"""
NEUROLECT AI - Question Generator View
Generates 6 question types across Bloom's Taxonomy and difficulty tiers
using Flan-T5 with verifiable source citations and MCQ distractor generation.
"""

import streamlit as st
from typing import List, Optional
from config import QUESTION_TYPES, DIFFICULTY_LEVELS, BLOOMS_TAXONOMY
from modules.content_models import LectureContent
from modules.question_models import GeneratedQuestion
from modules.question_generator import QuestionGenerator
from modules.quiz_models import QuizQuestion
from database.db_manager import DatabaseManager


def render_questions_page(db: DatabaseManager) -> None:
    """Renders the Question Generator page."""
    st.header("❓ Intelligent Question Generator")
    st.markdown(
        "Generate multi-type assessment questions directly from lecture content with "
        "Bloom's Taxonomy classification, difficulty ratings, and verifiable source references."
    )

    content: Optional[LectureContent] = st.session_state.get("active_lecture")
    if not content:
        st.warning("⚠️ No active lecture loaded. Please upload or select a lecture first.")
        if st.button("📤 Go to Lecture Upload"):
            st.session_state.nav_page = "📤 Upload Lecture"
            st.rerun()
        return

    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
        <strong>Active Lecture:</strong> {content.title} &nbsp;|&nbsp; 
        <span style="color: #94a3b8;">{content.source_type.upper()} • {content.total_words} words</span>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # GENERATION CONFIGURATION CONTROLS
    # =========================================================================
    with st.expander("⚙️ Generation Parameters & Curriculum Filters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            selected_types = st.multiselect(
                "Question Modalities",
                options=QUESTION_TYPES,
                default=["MCQ", "Short Answer", "Descriptive"],
                help="Select which question types to generate."
            )
            difficulty_filter = st.selectbox(
                "Target Difficulty",
                options=["All Difficulties"] + DIFFICULTY_LEVELS,
                help="Constrain generation to specific cognitive challenge tiers."
            )

        with col2:
            bloom_filter = st.selectbox(
                "Target Bloom's Taxonomy Level",
                options=["All Cognitive Levels"] + BLOOMS_TAXONOMY,
                help="Target specific levels of Bloom's cognitive taxonomy."
            )
            num_questions = st.slider(
                "Number of Questions to Synthesize",
                min_value=1,
                max_value=15,
                value=6,
                step=1
            )

        generate_btn = st.button("✨ Generate Intelligent Questions", type="primary", use_container_width=True)

    # Initialize generator
    generator = QuestionGenerator()

    if generate_btn:
        if not selected_types:
            st.warning("Please select at least one question modality.")
        else:
            with st.spinner("Analyzing lecture semantics and generating curriculum-grounded questions with Flan-T5..."):
                try:
                    diff_arg = [difficulty_filter] if difficulty_filter != "All Difficulties" else None
                    bloom_arg = [bloom_filter] if bloom_filter != "All Cognitive Levels" else None

                    generated = generator.generate_questions(
                        content=content,
                        question_types=selected_types,
                        target_difficulties=diff_arg,
                        target_blooms=bloom_arg,
                        total_questions=num_questions
                    )

                    st.session_state.generated_questions = generated
                    # Persist automatically to database
                    db.save_questions(generated, lecture_id=st.session_state.get("active_lecture_id"))

                    st.success(f"Generated {len(generated)} questions successfully!")

                except Exception as e:
                    st.error(f"Generation error: {e}")

    # =========================================================================
    # DISPLAY GENERATED QUESTIONS
    # =========================================================================
    questions: List[GeneratedQuestion] = st.session_state.get("generated_questions", [])

    if not questions:
        st.info("No questions generated yet. Configure parameters above and click **Generate Intelligent Questions**.")
        # Option to load saved questions from DB
        all_db_q = db.get_questions_by_lecture(st.session_state.get("active_lecture_id", ""))
        if all_db_q:
            if st.button(f"📥 Load {len(all_db_q)} Saved Questions for this Lecture from Database"):
                st.session_state.generated_questions = all_db_q
                st.rerun()
    else:
        st.subheader(f"📋 Generated Assessment Suite ({len(questions)} Questions)")

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("💾 Persist Questions to Library", use_container_width=True):
                count = db.save_questions(questions, lecture_id=st.session_state.get("active_lecture_id"))
                st.success(f"Saved {count} questions to database library.")

        with col_act2:
            if st.button("📝 Create & Start Quiz from Questions ➡️", type="primary", use_container_width=True):
                st.session_state.quiz_source_questions = questions
                st.session_state.nav_page = "📝 Take Quiz"
                st.rerun()

        show_answers = st.checkbox("👁️ Reveal Ground-Truth Answers & Rationale", value=True)

        for idx, q in enumerate(questions, 1):
            type_color = "#3b82f6" if q.question_type == "MCQ" else "#10b981" if "Short" in q.question_type else "#8b5cf6"

            st.markdown(f"""
            <div class="neuro-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div>
                        <span style="background: {type_color}22; color: {type_color}; border: 1px solid {type_color}44; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">
                            {q.question_type}
                        </span>
                        <span style="background: rgba(255, 255, 255, 0.05); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.1); padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; margin-left: 0.4rem;">
                            {q.difficulty}
                        </span>
                        <span style="background: rgba(255, 255, 255, 0.05); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.1); padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; margin-left: 0.4rem;">
                            Bloom: {q.bloom_level}
                        </span>
                    </div>
                    <span style="color: #94a3b8; font-size: 0.8rem;">📍 {q.source_reference}</span>
                </div>
                <h4 style="margin: 0.5rem 0; color: #f8fafc;">{idx}. {q.question_text}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Display options for MCQ
            if q.question_type == "MCQ" and q.options:
                st.markdown("**Options:**")
                for opt_idx, opt in enumerate(q.options):
                    letter = chr(ord("A") + opt_idx)
                    is_correct_opt = (opt.strip() == q.correct_answer.strip())
                    if show_answers and is_correct_opt:
                        st.markdown(f"- **[{letter}] {opt}** &nbsp;*(Correct Answer)*")
                    else:
                        st.markdown(f"- **[{letter}]** {opt}")

            # Ground truth answer section
            if show_answers:
                with st.expander(f"💡 Ground-Truth Answer & Explanation for Question {idx}", expanded=False):
                    st.markdown(f"**Reference Answer:** {q.correct_answer}")
                    if q.explanation:
                        st.markdown(f"**Pedagogical Rationale:** {q.explanation}")
                    st.caption(f"Source Provenance: {q.source_reference} | Topic: {q.topic} › {q.subtopic}")
