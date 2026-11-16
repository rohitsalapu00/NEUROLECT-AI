"""
NEUROLECT AI - Lecture Understanding View
Integrates genuine Phase 3 transformer modules: Summarizer (Flan-T5),
TopicDetector, KeywordExtractor, NERProcessor (BERT-NER), and ConceptExtractor.
"""

import streamlit as st
from typing import Optional
from modules.content_models import LectureContent
from modules.summarizer import LectureSummarizer
from modules.topic_detector import TopicDetector
from modules.keyword_extractor import KeywordExtractor
from modules.ner_processor import NERProcessor
from modules.concept_extractor import ConceptExtractor
from database.db_manager import DatabaseManager


def render_understanding_page(db: DatabaseManager) -> None:
    """Renders the Lecture Understanding insights page."""
    st.header("🧠 Lecture Understanding & Knowledge Extraction")
    st.markdown(
        "Extract deep educational intelligence from the active lecture using specialized "
        "transformer models and linguistic processors."
    )

    # 1. Check for active lecture
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
        <span style="color: #94a3b8;">{content.source_type.upper()} • {content.total_words} words • {content.total_segments} segments</span>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_topics, tab_keywords, tab_ner = st.tabs([
        "📝 Summaries & Notes",
        "📌 Topics & Outline",
        "🔑 Keywords & Definitions",
        "🏷️ Named Entities"
    ])

    # =========================================================================
    # TAB 1: SUMMARIES
    # =========================================================================
    with tab_summary:
        st.subheader("Transformer-Based Multi-Tier Summarization")
        st.markdown(
            "Uses instruction-tuned `google/flan-t5-small` with sentence boundary chunking. "
            "No document content is discarded."
        )

        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            gen_concise = st.button("📄 Generate Concise Summary", use_container_width=True)
        with s_col2:
            gen_detailed = st.button("📑 Generate Detailed Summary", use_container_width=True)
        with s_col3:
            gen_notes = st.button("📋 Generate Structured Notes", use_container_width=True)

        # Cache in session state
        if "summary_cache" not in st.session_state or st.session_state.get("summary_lec_id") != content.title:
            st.session_state.summary_cache = {}
            st.session_state.summary_lec_id = content.title

        summarizer = LectureSummarizer()

        if gen_concise:
            with st.spinner("Generating concise summary with Flan-T5..."):
                try:
                    res = summarizer.generate_concise_summary(content)
                    st.session_state.summary_cache["concise"] = res
                except Exception as e:
                    st.error(f"Summarization error: {e}")

        if gen_detailed:
            with st.spinner("Processing all text chunks for detailed summary..."):
                try:
                    res = summarizer.generate_detailed_summary(content)
                    st.session_state.summary_cache["detailed"] = res
                except Exception as e:
                    st.error(f"Summarization error: {e}")

        if gen_notes:
            with st.spinner("Synthesizing structured lecture notes..."):
                try:
                    res = summarizer.generate_structured_notes(content)
                    st.session_state.summary_cache["notes"] = res
                except Exception as e:
                    st.error(f"Structured notes error: {e}")

        # Display cached summaries
        if "concise" in st.session_state.summary_cache:
            st.markdown("#### 📄 Concise Summary")
            st.info(st.session_state.summary_cache["concise"])

        if "detailed" in st.session_state.summary_cache:
            st.markdown("#### 📑 Detailed Summary")
            st.write(st.session_state.summary_cache["detailed"])

        if "notes" in st.session_state.summary_cache:
            st.markdown("#### 📋 Structured Lecture Notes")
            st.markdown(st.session_state.summary_cache["notes"])

    # =========================================================================
    # TAB 2: TOPICS & OUTLINE
    # =========================================================================
    with tab_topics:
        st.subheader("Topic Detection & Hierarchical Outline")
        st.markdown(
            "Discovers thematic clusters using semantic similarity embeddings without vector databases. "
            "Maps segments to specific curriculum topics."
        )

        if st.button("🔍 Detect Topics & Outline", key="btn_detect_topics", use_container_width=True):
            with st.spinner("Clustering lecture segments into topics..."):
                try:
                    detector = TopicDetector()
                    topics = detector.detect_topics(content)
                    outline = detector.generate_lecture_outline(content)
                    st.session_state.topic_cache = {
                        "topics": topics,
                        "outline": outline,
                        "lec_id": content.title
                    }
                except Exception as e:
                    st.error(f"Topic detection error: {e}")

        topic_data = st.session_state.get("topic_cache")
        if topic_data and topic_data.get("lec_id") == content.title:
            topics = topic_data["topics"]
            outline = topic_data["outline"]

            st.markdown(f"Found **{len(topics)}** core topic(s) across lecture segments:")

            t_cols = st.columns(min(3, max(1, len(topics))))
            for idx, top in enumerate(topics):
                with t_cols[idx % len(t_cols)]:
                    st.markdown(f"""
                    <div class="neuro-card">
                        <h4 style="margin: 0; color: #818cf8;">📌 {top['topic_name']}</h4>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">Segments: {len(top.get('segments', []))}</p>
                        <p style="font-size: 0.9rem; color: #cbd5e1;">{top.get('summary', '')[:140]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📑 View Complete Hierarchical Outline", expanded=True):
                st.markdown(outline)

    # =========================================================================
    # TAB 3: KEYWORDS & DEFINITIONS
    # =========================================================================
    with tab_keywords:
        st.subheader("Key Educational Concepts & Technical Vocabulary")

        if st.button("🔑 Extract Keywords & Definitions", key="btn_extract_keywords", use_container_width=True):
            with st.spinner("Extracting salient keywords and core definitions..."):
                try:
                    kw_extractor = KeywordExtractor()
                    keywords = kw_extractor.extract_keywords(content, top_n=15)

                    c_extractor = ConceptExtractor()
                    definitions = c_extractor.extract_definitions(content)
                    principles = c_extractor.extract_principles(content)

                    st.session_state.keyword_cache = {
                        "keywords": keywords,
                        "definitions": definitions,
                        "principles": principles,
                        "lec_id": content.title
                    }
                except Exception as e:
                    st.error(f"Concept extraction error: {e}")

        kw_data = st.session_state.get("keyword_cache")
        if kw_data and kw_data.get("lec_id") == content.title:
            k_list = kw_data["keywords"]
            defs = kw_data["definitions"]
            prins = kw_data["principles"]

            st.markdown("##### 🏷️ Salient Keywords")
            badge_html = " ".join([
                f'<span style="display: inline-block; background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 9999px; padding: 0.25rem 0.75rem; margin: 0.25rem; font-size: 0.85rem; font-weight: 500;">{k[0] if isinstance(k, tuple) else k}</span>'
                for k in k_list
            ])
            st.markdown(badge_html, unsafe_allow_html=True)

            col_def, col_prin = st.columns(2)
            with col_def:
                st.markdown("##### 📖 Formal Definitions")
                if defs:
                    for d in defs[:8]:
                        st.markdown(f"- **{d.get('concept', 'Concept')}**: {d.get('definition', '')} *(Source: {d.get('source_reference', 'Text')})*")
                else:
                    st.info("No explicit formal definition patterns ('is defined as', 'refers to') identified.")

            with col_prin:
                st.markdown("##### ⚖️ Core Principles & Rules")
                if prins:
                    for p in prins[:8]:
                        st.markdown(f"- {p.get('text', '')} *(Source: {p.get('source_reference', 'Text')})*")
                else:
                    st.info("No explicit principle markers ('therefore', 'must satisfy') identified.")

    # =========================================================================
    # TAB 4: NAMED ENTITY RECOGNITION
    # =========================================================================
    with tab_ner:
        st.subheader("Named Entity Recognition (NER)")
        st.markdown("Powered by pretrained Hugging Face checkpoint: `dslim/bert-base-NER`.")

        if st.button("🏷️ Run Named Entity Extraction", key="btn_run_ner", use_container_width=True):
            with st.spinner("Extracting entities via BERT-NER..."):
                try:
                    ner = NERProcessor()
                    entities = ner.extract_entities(content)
                    st.session_state.ner_cache = {
                        "entities": entities,
                        "lec_id": content.title
                    }
                except Exception as e:
                    st.error(f"NER extraction error: {e}")

        ner_data = st.session_state.get("ner_cache")
        if ner_data and ner_data.get("lec_id") == content.title:
            entity_result = ner_data["entities"]
            categories = entity_result.get("categories", {}) if isinstance(entity_result, dict) else {}
            total_detected = entity_result.get("total_entities_detected", 0) if isinstance(entity_result, dict) else 0

            if total_detected == 0:
                st.info("No high-confidence named entities identified in this lecture.")
            else:
                st.markdown(f"Detected **{total_detected}** unique named entity mention(s):")
                active_cats = {k: v for k, v in categories.items() if v.get("count", 0) > 0}
                if active_cats:
                    ner_cols = st.columns(len(active_cats))
                    for idx, (cat_key, cat_data) in enumerate(active_cats.items()):
                        with ner_cols[idx]:
                            ent_list = cat_data.get("entities", [])
                            items_html = "".join([
                                f"<li><strong>{e.get('name', '')}</strong> <span style='color: #94a3b8; font-size: 0.8rem;'>({e.get('source_references', [''])[0]})</span></li>"
                                for e in ent_list[:8]
                            ])
                            st.markdown(f"""
                            <div class="neuro-card">
                                <h4 style="margin: 0; color: #38bdf8; font-size: 0.95rem;">{cat_data.get('label', cat_key)}</h4>
                                <ul style="padding-left: 1.1rem; margin-top: 0.5rem; color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">
                                    {items_html}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No named entities found in content segments.")
