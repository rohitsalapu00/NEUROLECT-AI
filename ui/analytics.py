"""
NEUROLECT AI - Learning Analytics View
Interactive analytics dashboard powered by Plotly and SQLite data records.
Visualizes mastery trajectories, topic breakdowns, cognitive distributions, and outcomes.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from database.db_manager import DatabaseManager


def render_analytics_page(db: DatabaseManager) -> None:
    """Renders the Learning Analytics dashboard."""
    st.header("📈 Learning Analytics & Mastery Dashboard")
    st.markdown(
        "Track student cognitive mastery trajectories, curriculum topic performance, "
        "and outcome breakdowns based on verifiable assessment attempts."
    )

    attempts = db.list_quiz_attempts(limit=50)

    # Empty State Guard
    if not attempts:
        st.info(
            "📊 **No Assessment Data Recorded Yet.**\n\n"
            "Complete interactive quizzes to generate longitudinal mastery curves, "
            "curriculum topic breakdowns, and Bloom's Taxonomy cognitive distribution analytics."
        )
        if st.button("📝 Take an Assessment"):
            st.session_state.nav_page = "📝 Take Quiz"
            st.rerun()
        return

    # Top KPI summary
    total_attempts = len(attempts)
    avg_score = round(sum(a["percentage"] for a in attempts) / total_attempts, 1)
    best_score = max(a["percentage"] for a in attempts)
    total_questions = sum(a["total_questions"] for a in attempts)
    total_correct = sum(a["correct_count"] for a in attempts)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Quizzes Completed", total_attempts)
    with k2:
        st.metric("Mean Performance", f"{avg_score}%")
    with k3:
        st.metric("Highest Score", f"{best_score}%")
    with k4:
        st.metric("Questions Attempted", total_questions)

    st.markdown("---")

    # 1. Longitudinal Mastery Trajectory (Chronological Trend)
    st.subheader("📈 Performance Trajectory Over Time")

    # Sort chronologically for timeline
    chronological_attempts = sorted(attempts, key=lambda x: x["completed_at"])
    dates = [a["completed_at"][:16].replace("T", " ") for a in chronological_attempts]
    scores = [a["percentage"] for a in chronological_attempts]
    titles = [a["quiz_title"] for a in chronological_attempts]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode="lines+markers",
        name="Quiz Score %",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=8, color="#818cf8"),
        text=[f"Quiz: {t}<br>Score: {s}%" for t, s in zip(titles, scores)],
        hoverinfo="text"
    ))
    fig_timeline.add_hline(y=75, line_dash="dash", line_color="#10b981", annotation_text="Mastery (75%)")
    fig_timeline.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Proficiency (50%)")
    fig_timeline.update_layout(
        title="Assessment Score Progression",
        yaxis=dict(title="Score (%)", range=[0, 105]),
        xaxis=dict(title="Date & Time"),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=360
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # 2. Aggregated Category Analytics (Topic and Outcome Breakdown)
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("📚 Topic-Wise Mastery")
        topic_totals = {}
        topic_earned = {}

        for att in attempts:
            full_att = db.get_quiz_attempt(att["attempt_id"])
            if full_att and full_att.get("topic_scores"):
                for topic, stats in full_att["topic_scores"].items():
                    topic_totals[topic] = topic_totals.get(topic, 0.0) + stats.get("total_marks", 0.0)
                    topic_earned[topic] = topic_earned.get(topic, 0.0) + stats.get("earned_marks", 0.0)

        if topic_totals:
            topics = list(topic_totals.keys())
            pcts = [round(topic_earned[t] / topic_totals[t] * 100.0, 1) if topic_totals[t] > 0 else 0.0 for t in topics]

            fig_topic = px.bar(
                x=pcts,
                y=topics,
                orientation="h",
                labels={"x": "Mastery Percentage (%)", "y": "Topic"},
                title="Curriculum Topic Mastery Rates",
                color=pcts,
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                range_color=[0, 100],
                template="plotly_dark",
                height=340
            )
            fig_topic.update_layout(xaxis=dict(range=[0, 105]), margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_topic, use_container_width=True)
        else:
            st.info("No topic breakdown data available in recorded attempts.")

    with c_right:
        st.subheader("🎯 Overall Outcome Distribution")
        tot_c = sum(a["correct_count"] for a in attempts)
        tot_p = sum(a["partial_count"] for a in attempts)
        tot_i = sum(a["incorrect_count"] for a in attempts)
        tot_u = sum(a["unanswered_questions"] for a in attempts)

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Correct", "Partially Correct", "Incorrect", "Unanswered"],
            values=[tot_c, tot_p, tot_i, tot_u],
            hole=0.45,
            marker=dict(colors=["#10b981", "#f59e0b", "#ef4444", "#64748b"])
        )])
        fig_pie.update_layout(
            title="Question Evaluation Breakdown",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            height=340
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Difficulty and Bloom's Taxonomy Breakdown
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.subheader("📊 Performance by Difficulty Tier")
        diff_totals = {}
        diff_earned = {}

        for att in attempts:
            full_att = db.get_quiz_attempt(att["attempt_id"])
            if full_att and full_att.get("difficulty_scores"):
                for diff, stats in full_att["difficulty_scores"].items():
                    diff_totals[diff] = diff_totals.get(diff, 0.0) + stats.get("total_marks", 0.0)
                    diff_earned[diff] = diff_earned.get(diff, 0.0) + stats.get("earned_marks", 0.0)

        if diff_totals:
            diff_levels = ["Easy", "Medium", "Hard"]
            diff_present = [d for d in diff_levels if d in diff_totals]
            diff_pcts = [round(diff_earned[d] / diff_totals[d] * 100.0, 1) if diff_totals[d] > 0 else 0.0 for d in diff_present]

            fig_diff = px.bar(
                x=diff_present,
                y=diff_pcts,
                labels={"x": "Difficulty Tier", "y": "Success Rate (%)"},
                title="Cognitive Difficulty Performance",
                color=diff_pcts,
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                range_color=[0, 100],
                template="plotly_dark",
                height=320
            )
            fig_diff.update_layout(yaxis=dict(range=[0, 105]), margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("No difficulty distribution data available.")

    with d_col2:
        st.subheader("🎓 Bloom's Taxonomy Mastery Profile")
        bloom_totals = {}
        bloom_earned = {}

        for att in attempts:
            full_att = db.get_quiz_attempt(att["attempt_id"])
            if full_att and full_att.get("bloom_scores"):
                for bl, stats in full_att["bloom_scores"].items():
                    bloom_totals[bl] = bloom_totals.get(bl, 0.0) + stats.get("total_marks", 0.0)
                    bloom_earned[bl] = bloom_earned.get(bl, 0.0) + stats.get("earned_marks", 0.0)

        if bloom_totals:
            bloom_cats = list(bloom_totals.keys())
            bloom_pcts = [round(bloom_earned[b] / bloom_totals[b] * 100.0, 1) if bloom_totals[b] > 0 else 0.0 for b in bloom_cats]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=bloom_pcts,
                theta=bloom_cats,
                fill='toself',
                name='Bloom Mastery',
                line_color='#8b5cf6'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                title="Cognitive Depth Distribution",
                margin=dict(l=30, r=30, t=50, b=30),
                height=320
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("No Bloom's taxonomy data available.")
