"""
NEUROLECT AI - Topic and Subtopic Detection Engine
Discovers major educational themes and hierarchical subtopics from lecture content.
Uses SentenceTransformer semantic embeddings and Agglomerative Clustering with cosine distance.
Extracts meaningful topic titles and subtopics using semantic keyphrases and centroid thesis statements.
Transparently associates source segments (pages/timestamps) with detected topics.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from models.model_loader import ModelLoader, ModelLoadingError
from modules.content_models import LectureContent, ContentSegment
from utils.helpers import clean_text

logger = logging.getLogger("NEUROLECT.TopicDetector")


class TopicDetectionError(Exception):
    """Raised when topic modeling or outline generation fails."""
    pass


class TopicDetector:
    """
    Identifies major lecture topics, subtopics, and hierarchical outlines.
    Technique: Pretrained SentenceTransformer embeddings (all-MiniLM-L6-v2)
    with Agglomerative Clustering on cosine distance, paired with KeyBERT keyphrase title extraction
    and centroid-based thesis extraction.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name

    def detect_topics(
        self,
        content: LectureContent,
        num_topics: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Discovers major topics and subtopics from LectureContent segments.
        Handles short and long documents gracefully with semantic clustering.
        Returns a list of structured topic objects with source associations.
        """
        if not content.segments:
            raise TopicDetectionError("LectureContent contains no segments for topic detection.")

        # Substantive segments with sufficient educational content
        substantive_segments = [s for s in content.segments if len(s.text.split()) >= 6]
        if not substantive_segments:
            substantive_segments = content.segments

        total_segs = len(substantive_segments)

        # 1. Graceful handling of very short documents
        if total_segs == 1:
            seg = substantive_segments[0]
            title = self._derive_topic_title([seg], 1)
            subtopics = self._extract_subtopics([seg])
            return [{
                "topic_id": 1,
                "name": title,
                "thesis_statement": seg.text[:200] + ("..." if len(seg.text) > 200 else ""),
                "subtopics": subtopics,
                "segment_ids": [seg.segment_id],
                "source_labels": [seg.source_label],
                "source_label": seg.source_label,
                "associated_text": seg.text,
                "word_count": len(seg.text.split()),
            }]

        # 2. Determine target topic count
        if num_topics is None:
            if total_segs <= 3:
                target_k = min(2, total_segs)
            elif total_segs <= 8:
                target_k = max(2, total_segs // 3)
            else:
                target_k = min(5, max(3, total_segs // 3))
        else:
            target_k = max(1, min(num_topics, total_segs))

        try:
            st_model = ModelLoader.get_sentence_transformer(self.model_name)
        except ModelLoadingError as e:
            raise TopicDetectionError(f"Embedding model unavailable for topic detection: {e}") from e

        # 3. Compute semantic embeddings for each segment
        segment_texts = [s.text for s in substantive_segments]
        embeddings = st_model.encode(segment_texts, convert_to_numpy=True, show_progress_bar=False)

        # 4. Genuine Semantic Clustering via AgglomerativeClustering on Cosine Distance
        clusters = self._cluster_semantic_segments(embeddings, target_k)

        # 5. Extract topic titles, subtopics, and centroid thesis statements
        detected_topics = []
        for cluster_idx, seg_indices in enumerate(clusters):
            cluster_segs = [substantive_segments[i] for i in seg_indices]
            cluster_embs = embeddings[seg_indices]
            centroid = np.mean(cluster_embs, axis=0, keepdims=True)

            # Find segment closest to cluster centroid (thesis statement)
            sims = cosine_similarity(centroid, cluster_embs)[0]
            best_seg_idx = np.argmax(sims)
            thesis_seg = cluster_segs[best_seg_idx]

            # Derive meaningful title via keyphrase extraction
            topic_title = self._derive_topic_title(cluster_segs, cluster_idx + 1)
            subtopics = self._extract_subtopics(cluster_segs)

            associated_text = " ".join([s.text for s in cluster_segs])
            source_labels = list(dict.fromkeys([s.source_label for s in cluster_segs]))

            detected_topics.append({
                "topic_id": cluster_idx + 1,
                "name": topic_title,
                "thesis_statement": thesis_seg.text[:200] + ("..." if len(thesis_seg.text) > 200 else ""),
                "subtopics": subtopics,
                "segment_ids": [s.segment_id for s in cluster_segs],
                "source_labels": source_labels,
                "source_label": ", ".join(source_labels[:3]),
                "associated_text": associated_text,
                "word_count": len(associated_text.split()),
            })

        logger.info(f"Semantically clustered {total_segs} segments into {len(detected_topics)} topics.")
        return detected_topics

    def _cluster_semantic_segments(
        self,
        embeddings: np.ndarray,
        num_clusters: int
    ) -> List[List[int]]:
        """
        Uses Agglomerative Clustering with cosine distance to group semantically related segments.
        Sorts the resulting clusters chronologically by their earliest segment occurrence.
        """
        n = len(embeddings)
        if n <= num_clusters:
            return [[i] for i in range(n)]

        if num_clusters == 1:
            return [list(range(n))]

        try:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric="cosine",
                linkage="average"
            )
            labels = clustering.fit_predict(embeddings)

            cluster_map: Dict[int, List[int]] = {}
            for idx, label in enumerate(labels):
                cluster_map.setdefault(int(label), []).append(idx)

            # Sort clusters by their first appearance in the lecture to maintain natural narrative
            sorted_clusters = sorted(cluster_map.values(), key=lambda indices: min(indices))
            return sorted_clusters

        except Exception as e:
            logger.warning(f"Clustering fallback triggered: {e}")
            step = n / num_clusters
            clusters = []
            for i in range(num_clusters):
                start = int(round(i * step))
                end = int(round((i + 1) * step))
                if start < end:
                    clusters.append(list(range(start, end)))
            return clusters

    def _derive_topic_title(self, segments: List[ContentSegment], fallback_idx: int) -> str:
        """
        Derives an informative topic title from semantic keyphrases or opening sentence.
        """
        combined = " ".join([s.text for s in segments])

        # Attempt KeyBERT keyphrase extraction for genuine semantic title
        try:
            kw_model = ModelLoader.get_keybert_model()
            keyphrases = kw_model.extract_keywords(
                combined,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=3
            )
            if keyphrases:
                top_kp = keyphrases[0][0].title()
                if len(top_kp.split()) >= 2:
                    return top_kp
                elif len(keyphrases) > 1:
                    return f"{top_kp} & {keyphrases[1][0].title()}"
                return top_kp
        except Exception as e:
            logger.debug(f"Keyphrase title fallback: {e}")

        # Fallback to salient sentence title
        first_sentence = combined.split(".")[0].strip()
        if ":" in first_sentence:
            candidate = first_sentence.split(":")[1].strip().title()
            if 2 <= len(candidate.split()) <= 6:
                return candidate

        words = first_sentence.split()
        if 2 <= len(words) <= 6:
            return first_sentence.title()
        elif len(words) > 6:
            return " ".join(words[:5]).title()

        return f"Topic {fallback_idx}: Core Principles"

    def _extract_subtopics(self, segments: List[ContentSegment]) -> List[str]:
        """
        Extracts salient sub-themes or keyphrases within a cluster.
        """
        subtopics = []
        combined = " ".join([s.text for s in segments])

        # Use KeyBERT to get distinct 2-word subtopic concepts
        try:
            kw_model = ModelLoader.get_keybert_model()
            kws = kw_model.extract_keywords(
                combined,
                keyphrase_ngram_range=(2, 3),
                stop_words="english",
                use_mmr=True,
                diversity=0.5,
                top_n=4
            )
            for kw, _ in kws:
                subtopics.append(kw.title())
        except Exception:
            pass

        # If not enough, find sentences with illustrative markers
        if len(subtopics) < 2:
            for seg in segments:
                sentences = [s.strip() for s in seg.text.split(".") if len(s.strip().split()) >= 4]
                for s in sentences:
                    if any(kw in s.lower() for kw in ["includes", "such as", "for example", "consists of", "known as", "defined as"]):
                        clean_s = s[:90].strip()
                        if clean_s not in subtopics:
                            subtopics.append(clean_s)
                    if len(subtopics) >= 4:
                        break

        return subtopics[:4]

    def generate_lecture_outline(self, content: LectureContent) -> Dict[str, Any]:
        """
        Generates a full structured hierarchical lecture syllabus outline.
        """
        topics = self.detect_topics(content)
        outline = {
            "title": content.title,
            "source_type": content.source_type,
            "total_topics": len(topics),
            "topics": [
                {
                    "topic_number": t["topic_id"],
                    "title": t["name"],
                    "source": t["source_label"],
                    "summary": t["thesis_statement"],
                    "subtopics": t["subtopics"],
                }
                for t in topics
            ]
        }
        return outline
