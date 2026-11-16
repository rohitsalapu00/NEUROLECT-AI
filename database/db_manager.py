"""
NEUROLECT AI - Database & Persistence Manager
Provides lightweight, thread-safe SQLite persistence for lectures, summaries,
generated questions, quiz sessions, student submissions, and assessment results.
Preserves full source provenance without introducing RAG or external vector stores.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import DATABASE_PATH
from modules.content_models import LectureContent, ContentChunk
from modules.question_models import GeneratedQuestion
from modules.quiz_models import Quiz, QuizQuestion, QuestionResult, QuizResult

logger = logging.getLogger("NEUROLECT.Database")


class DatabaseManager:
    """
    Thread-safe SQLite database manager for educational assessment records.
    """

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a SQLite connection with timeout and dictionary row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initializes database schema if tables do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Lectures Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS lectures (
                lecture_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_filename TEXT,
                raw_text TEXT NOT NULL,
                num_words INTEGER NOT NULL,
                num_segments INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Lecture Summaries Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS lecture_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id TEXT NOT NULL,
                summary_type TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (lecture_id) REFERENCES lectures(lecture_id) ON DELETE CASCADE
            );
            """)

            # 3. Generated Questions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_questions (
                question_id TEXT PRIMARY KEY,
                lecture_id TEXT,
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                options TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                topic TEXT,
                subtopic TEXT,
                bloom_level TEXT,
                difficulty TEXT,
                source_reference TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (lecture_id) REFERENCES lectures(lecture_id) ON DELETE SET NULL
            );
            """)

            # 4. Quizzes Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                quiz_id TEXT PRIMARY KEY,
                lecture_id TEXT,
                title TEXT NOT NULL,
                total_marks REAL NOT NULL,
                total_questions INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 5. Quiz Attempts Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                attempt_id TEXT PRIMARY KEY,
                quiz_id TEXT NOT NULL,
                quiz_title TEXT NOT NULL,
                lecture_id TEXT,
                total_questions INTEGER NOT NULL,
                answered_questions INTEGER NOT NULL,
                unanswered_questions INTEGER NOT NULL,
                total_marks REAL NOT NULL,
                earned_marks REAL NOT NULL,
                percentage REAL NOT NULL,
                correct_count INTEGER NOT NULL,
                partial_count INTEGER NOT NULL,
                incorrect_count INTEGER NOT NULL,
                topic_scores TEXT,
                difficulty_scores TEXT,
                bloom_scores TEXT,
                is_complete INTEGER NOT NULL,
                completed_at TEXT NOT NULL
            );
            """)

            # 6. Attempt Answers Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempt_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                student_answer TEXT,
                reference_answer TEXT,
                marks_awarded REAL NOT NULL,
                max_marks REAL NOT NULL,
                evaluation_status TEXT NOT NULL,
                feedback TEXT,
                missing_concepts TEXT,
                score_breakdown TEXT,
                source_reference TEXT,
                segment_id INTEGER,
                FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(attempt_id) ON DELETE CASCADE
            );
            """)

            conn.commit()
            logger.info("Database schema initialized successfully.")

    # =========================================================================
    # LECTURES API
    # =========================================================================

    def save_lecture(self, lecture: LectureContent, lecture_id: Optional[str] = None) -> str:
        """Saves a LectureContent instance to the database."""
        import uuid
        lec_id = lecture_id or lecture.metadata.get("lecture_id") or f"lec_{uuid.uuid4().hex[:8]}"
        lecture.metadata["lecture_id"] = lec_id

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO lectures (
                lecture_id, title, source_type, source_filename,
                raw_text, num_words, num_segments, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                lec_id,
                lecture.title,
                lecture.source_type,
                lecture.source_filename,
                lecture.full_text,
                lecture.total_words,
                lecture.total_segments,
                json.dumps(lecture.to_dict()),
                lecture.created_at
            ))
            conn.commit()
            logger.info(f"Saved lecture '{lecture.title}' (ID: {lec_id}).")
            return lec_id

    def get_lecture(self, lecture_id: str) -> Optional[LectureContent]:
        """Retrieves a LectureContent instance by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lectures WHERE lecture_id = ?;", (lecture_id,))
            row = cursor.fetchone()
            if not row:
                return None

            if row["metadata"]:
                try:
                    data = json.loads(row["metadata"])
                    return LectureContent.from_dict(data)
                except Exception:
                    pass

            return LectureContent(
                title=row["title"],
                source_type=row["source_type"],
                source_filename=row["source_filename"] or "",
                full_text=row["raw_text"],
                created_at=row["created_at"]
            )

    def list_lectures(self) -> List[Dict[str, Any]]:
        """Lists all processed lectures ordered by most recent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT lecture_id, title, source_type, source_filename, num_words, num_segments, created_at
            FROM lectures ORDER BY created_at DESC;
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # =========================================================================
    # QUESTIONS API
    # =========================================================================

    def save_questions(self, questions: List[GeneratedQuestion], lecture_id: Optional[str] = None) -> int:
        """Persists a list of GeneratedQuestion objects, avoiding duplicates."""
        if not questions:
            return 0

        saved = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if lecture exists to satisfy FK constraint
            target_lec_id = lecture_id
            if target_lec_id:
                cursor.execute("SELECT 1 FROM lectures WHERE lecture_id = ?;", (target_lec_id,))
                if not cursor.fetchone():
                    target_lec_id = None

            for q in questions:
                lec_ref = target_lec_id or q.generation_metadata.get("lecture_id")
                if lec_ref:
                    cursor.execute("SELECT 1 FROM lectures WHERE lecture_id = ?;", (lec_ref,))
                    if not cursor.fetchone():
                        lec_ref = None

                cursor.execute("""
                INSERT OR REPLACE INTO generated_questions (
                    question_id, lecture_id, question_text, question_type, options,
                    correct_answer, explanation, topic, subtopic, bloom_level,
                    difficulty, source_reference, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    q.question_id,
                    lec_ref,
                    q.question_text,
                    q.question_type,
                    json.dumps(q.options),
                    q.correct_answer,
                    q.explanation,
                    q.topic,
                    q.subtopic,
                    q.bloom_level,
                    q.difficulty,
                    q.source_reference,
                    json.dumps(q.generation_metadata),
                    q.created_at
                ))
                saved += 1
            conn.commit()
            logger.info(f"Saved {saved} generated questions to database.")
            return saved

    def get_questions_by_lecture(self, lecture_id: str) -> List[GeneratedQuestion]:
        """Retrieves questions generated for a specific lecture."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM generated_questions WHERE lecture_id = ? ORDER BY created_at ASC;", (lecture_id,))
            rows = cursor.fetchall()
            return [self._row_to_generated_question(r) for r in rows]

    def list_all_questions(self, limit: int = 200) -> List[GeneratedQuestion]:
        """Lists all stored questions across all lectures."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM generated_questions ORDER BY created_at DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [self._row_to_generated_question(r) for r in rows]

    def _row_to_generated_question(self, row: sqlite3.Row) -> GeneratedQuestion:
        """Helper to convert database row to GeneratedQuestion."""
        return GeneratedQuestion(
            question_id=row["question_id"],
            question_text=row["question_text"],
            question_type=row["question_type"],
            options=json.loads(row["options"]) if row["options"] else [],
            correct_answer=row["correct_answer"],
            explanation=row["explanation"] or "",
            topic=row["topic"] or "General",
            subtopic=row["subtopic"] or "General",
            bloom_level=row["bloom_level"] or "Understand",
            difficulty=row["difficulty"] or "Medium",
            source_reference=row["source_reference"] or "Lecture Source",
            generation_metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"]
        )

    # =========================================================================
    # QUIZ & ATTEMPT API
    # =========================================================================

    def save_quiz(self, quiz: Quiz, lecture_id: Optional[str] = None) -> str:
        """Saves a Quiz structure."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO quizzes (
                quiz_id, lecture_id, title, total_marks, total_questions, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                quiz.quiz_id,
                lecture_id,
                quiz.title,
                quiz.total_marks,
                len(quiz.questions),
                json.dumps(quiz.metadata),
                quiz.created_at
            ))
            conn.commit()
            return quiz.quiz_id

    def save_quiz_attempt(self, result: QuizResult, lecture_id: Optional[str] = None) -> str:
        """
        Persists a completed QuizResult and its individual question evaluations.
        """
        attempt_id = f"attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{result.quiz_id[:6]}"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert Attempt Record
            cursor.execute("""
            INSERT OR REPLACE INTO quiz_attempts (
                attempt_id, quiz_id, quiz_title, lecture_id,
                total_questions, answered_questions, unanswered_questions,
                total_marks, earned_marks, percentage,
                correct_count, partial_count, incorrect_count,
                topic_scores, difficulty_scores, bloom_scores,
                is_complete, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                attempt_id,
                result.quiz_id,
                result.quiz_title,
                lecture_id,
                result.total_questions,
                result.answered_questions,
                result.unanswered_questions,
                result.total_marks,
                result.earned_marks,
                result.percentage,
                result.correct_count,
                result.partial_count,
                result.incorrect_count,
                json.dumps(result.topic_scores),
                json.dumps(result.difficulty_scores),
                json.dumps(result.bloom_scores),
                1 if result.is_complete else 0,
                result.completed_at
            ))

            # Insert Question Answer Breakdown
            for qr in result.question_results:
                cursor.execute("""
                INSERT INTO attempt_answers (
                    attempt_id, question_id, question_text, question_type,
                    student_answer, reference_answer, marks_awarded, max_marks,
                    evaluation_status, feedback, missing_concepts, score_breakdown,
                    source_reference, segment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    attempt_id,
                    qr.question_id,
                    qr.question_text,
                    qr.question_type,
                    qr.student_answer,
                    qr.reference_answer,
                    qr.marks_awarded,
                    qr.max_marks,
                    qr.evaluation_status,
                    qr.feedback,
                    json.dumps(qr.missing_concepts),
                    json.dumps(qr.score_breakdown),
                    qr.source_reference,
                    qr.segment_id
                ))

            conn.commit()
            logger.info(f"Saved quiz attempt '{attempt_id}' for quiz '{result.quiz_id}'.")
            return attempt_id

    def list_quiz_attempts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists historical quiz attempts ordered from newest to oldest."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT attempt_id, quiz_id, quiz_title, lecture_id,
                   total_questions, answered_questions, unanswered_questions,
                   total_marks, earned_marks, percentage,
                   correct_count, partial_count, incorrect_count,
                   is_complete, completed_at
            FROM quiz_attempts
            ORDER BY completed_at DESC
            LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_quiz_attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full details of a past attempt including question answers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quiz_attempts WHERE attempt_id = ?;", (attempt_id,))
            row = cursor.fetchone()
            if not row:
                return None

            attempt = dict(row)
            attempt["topic_scores"] = json.loads(attempt["topic_scores"]) if attempt["topic_scores"] else {}
            attempt["difficulty_scores"] = json.loads(attempt["difficulty_scores"]) if attempt["difficulty_scores"] else {}
            attempt["bloom_scores"] = json.loads(attempt["bloom_scores"]) if attempt["bloom_scores"] else {}

            cursor.execute("SELECT * FROM attempt_answers WHERE attempt_id = ? ORDER BY id ASC;", (attempt_id,))
            ans_rows = cursor.fetchall()
            question_results = []
            for ar in ans_rows:
                q_dict = dict(ar)
                q_dict["missing_concepts"] = json.loads(q_dict["missing_concepts"]) if q_dict["missing_concepts"] else []
                q_dict["score_breakdown"] = json.loads(q_dict["score_breakdown"]) if q_dict["score_breakdown"] else {}
                question_results.append(q_dict)

            attempt["answers"] = question_results
            return attempt

    # =========================================================================
    # ANALYTICS & DASHBOARD SUMMARY API
    # =========================================================================

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Calculates real-time dashboard analytics.
        Does NOT fabricate statistics if data is empty.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Total Lectures
            cursor.execute("SELECT COUNT(*) AS count FROM lectures;")
            total_lectures = cursor.fetchone()["count"]

            # 2. Total Questions Generated
            cursor.execute("SELECT COUNT(*) AS count FROM generated_questions;")
            total_questions = cursor.fetchone()["count"]

            # 3. Total Quizzes Attempted
            cursor.execute("SELECT COUNT(*) AS count FROM quiz_attempts;")
            total_attempts = cursor.fetchone()["count"]

            # 4. Average Score Percentage
            cursor.execute("SELECT AVG(percentage) AS avg_score FROM quiz_attempts;")
            avg_row = cursor.fetchone()
            avg_score = round(avg_row["avg_score"], 1) if avg_row and avg_row["avg_score"] is not None else None

            # 5. Recent Lectures
            cursor.execute("""
            SELECT lecture_id, title, source_type, num_words, created_at
            FROM lectures ORDER BY created_at DESC LIMIT 5;
            """)
            recent_lectures = [dict(r) for r in cursor.fetchall()]

            # 6. Recent Quiz Attempts
            cursor.execute("""
            SELECT attempt_id, quiz_title, percentage, earned_marks, total_marks, completed_at
            FROM quiz_attempts ORDER BY completed_at DESC LIMIT 5;
            """)
            recent_attempts = [dict(r) for r in cursor.fetchall()]

            return {
                "total_lectures": total_lectures,
                "total_questions": total_questions,
                "total_quizzes_attempted": total_attempts,
                "average_score_percentage": avg_score,
                "recent_lectures": recent_lectures,
                "recent_attempts": recent_attempts
            }

    def clear_all_data(self) -> None:
        """Wipes all data tables. Used for test suites and testing isolation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attempt_answers;")
            cursor.execute("DELETE FROM quiz_attempts;")
            cursor.execute("DELETE FROM quizzes;")
            cursor.execute("DELETE FROM generated_questions;")
            cursor.execute("DELETE FROM lecture_summaries;")
            cursor.execute("DELETE FROM lectures;")
            conn.commit()
            logger.info("Cleared all data from database.")
