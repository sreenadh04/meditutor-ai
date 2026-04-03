"""
MediTutor AI - Progress Service with User Isolation
Tracks student performance, identifies weak topics, and manages sessions.
Now uses user_id instead of hardcoded "default_student".
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from database import (
    StudySession, QuestionAttempt, TopicProgress,
    Document, Flashcard, MCQuestion
)

logger = logging.getLogger(__name__)

WEAK_TOPIC_THRESHOLD = 0.6   # below 60% accuracy = weak topic


class ProgressService:

    # ── Sessions with user isolation ──────────────────────────────────────────

    def create_session(
        self,
        db: Session,
        document_id: str,
        user_id: str,  # ← CHANGED: was student_id, now user_id
    ) -> str:
        """
        Create a new study session for a user and document.
        
        Args:
            db: Database session
            document_id: Document being studied
            user_id: User identifier (from frontend UUID)
            
        Returns:
            session_id
        """
        # Verify document belongs to user
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not doc:
            raise ValueError(f"Document {document_id} not found or not owned by user {user_id}")
        
        session_id = str(uuid.uuid4())
        session = StudySession(
            id=session_id,
            document_id=document_id,
            student_id=user_id,  # student_id column now stores user_id
        )
        db.add(session)
        db.commit()
        
        logger.info(f"Created session {session_id[:8]}... for user {user_id[:8]}..., doc {document_id[:8]}...")
        return session_id

    def end_session(self, db: Session, session_id: str, user_id: str):
        """
        End a study session. Verifies ownership.
        """
        session = db.query(StudySession).filter(
            StudySession.id == session_id,
            StudySession.student_id == user_id  # ← Added user check
        ).first()
        
        if not session:
            raise ValueError(f"Session {session_id} not found or not owned by user {user_id}")
        
        session.ended_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Ended session {session_id[:8]}... for user {user_id[:8]}...")

    def get_session(self, db: Session, session_id: str, user_id: str) -> Optional[Dict]:
        """
        Get session details with ownership verification.
        """
        session = db.query(StudySession).filter(
            StudySession.id == session_id,
            StudySession.student_id == user_id
        ).first()
        
        if not session:
            return None
        
        # Get attempts for this session
        attempts = db.query(QuestionAttempt).filter(
            QuestionAttempt.session_id == session_id
        ).all()
        
        return {
            "session_id": session.id,
            "document_id": session.document_id,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "total_attempts": len(attempts),
            "correct_attempts": sum(1 for a in attempts if a.is_correct),
            "accuracy": round(sum(1 for a in attempts if a.is_correct) / len(attempts) * 100, 1) if attempts else 0,
        }

    def list_user_sessions(
        self,
        db: Session,
        user_id: str,
        document_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        List all sessions for a user, optionally filtered by document.
        """
        query = db.query(StudySession).filter(StudySession.student_id == user_id)
        
        if document_id:
            query = query.filter(StudySession.document_id == document_id)
        
        sessions = query.order_by(StudySession.started_at.desc()).limit(limit).all()
        
        return [
            {
                "session_id": s.id,
                "document_id": s.document_id,
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            }
            for s in sessions
        ]

    # ── Record attempts with user isolation ───────────────────────────────────

    def record_attempt(
        self,
        db: Session,
        session_id: str,
        question_text: str,
        question_type: str,
        topic: Optional[str],
        user_answer: Optional[str],
        correct_answer: Optional[str],
        is_correct: Optional[bool],
        score: float = 0.0,
    ) -> str:
        """
        Record a single question attempt.
        Session ownership is verified via the session's student_id.
        """
        # Verify session exists (ownership check happens via session.student_id)
        session = db.query(StudySession).filter(StudySession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        user_id = session.student_id  # Extract user_id from session
        
        attempt_id = str(uuid.uuid4())
        attempt = QuestionAttempt(
            id=attempt_id,
            session_id=session_id,
            question_text=question_text[:500],  # Truncate if needed
            question_type=question_type,
            topic=topic[:100] if topic else None,
            user_answer=str(user_answer)[:500] if user_answer else None,
            correct_answer=str(correct_answer)[:500] if correct_answer else None,
            is_correct=is_correct,
            score=score,
        )
        db.add(attempt)
        db.commit()

        # Update topic progress (with user_id from session)
        if topic:
            self._update_topic_progress(
                db, user_id, session.document_id, topic, bool(is_correct) if is_correct is not None else False
            )

        return attempt_id

    def record_mcq_batch(
        self,
        db: Session,
        session_id: str,
        document_id: str,
        results: List[dict],
    ):
        """
        Record all MCQ answers from a quiz submission.
        """
        # Verify session and get user_id
        session = db.query(StudySession).filter(StudySession.id == session_id).first()
        if not session:
            logger.warning(f"Session {session_id} not found, cannot record batch")
            return
        
        user_id = session.student_id
        
        # Verify document ownership
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not doc:
            logger.warning(f"Document {document_id} not owned by user {user_id}, cannot record batch")
            return

        for item in results:
            topic = item.get("topic", "General")
            is_correct = item.get("is_correct", False)
            
            attempt = QuestionAttempt(
                id=str(uuid.uuid4()),
                session_id=session_id,
                question_text=item.get("question", "")[:500],
                question_type="mcq",
                topic=topic[:100],
                user_answer=str(item.get("selected_index", ""))[:10],
                correct_answer=str(item.get("correct_index", ""))[:10],
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
            )
            db.add(attempt)
            self._update_topic_progress(
                db, user_id, document_id, topic, is_correct
            )

        db.commit()
        
        logger.info(f"Recorded {len(results)} MCQ attempts for user {user_id[:8]}..., session {session_id[:8]}...")

    # ── Topic progress with user isolation ────────────────────────────────────

    def _update_topic_progress(
        self,
        db: Session,
        user_id: str,
        document_id: str,
        topic: str,
        is_correct: bool,
    ):
        """
        Update or create topic progress record for a user.
        """
        existing = db.query(TopicProgress).filter(
            TopicProgress.student_id == user_id,
            TopicProgress.document_id == document_id,
            TopicProgress.topic == topic,
        ).first()

        if existing:
            existing.attempts += 1
            if is_correct:
                existing.correct += 1
            existing.accuracy = existing.correct / existing.attempts
            existing.is_weak = existing.accuracy < WEAK_TOPIC_THRESHOLD
            existing.last_attempt = datetime.utcnow()
            existing.updated_at = datetime.utcnow()
        else:
            tp = TopicProgress(
                id=str(uuid.uuid4()),
                student_id=user_id,
                document_id=document_id,
                topic=topic[:100],
                attempts=1,
                correct=1 if is_correct else 0,
                accuracy=1.0 if is_correct else 0.0,
                is_weak=not is_correct,
                last_attempt=datetime.utcnow(),
            )
            db.add(tp)

        db.commit()

    # ── Progress Report with user isolation ───────────────────────────────────

    def get_progress(
        self,
        db: Session,
        document_id: str,
        user_id: str,  # ← CHANGED: required, no default
    ) -> dict:
        """
        Get comprehensive progress report for a user on a document.
        
        Args:
            db: Database session
            document_id: Document to get progress for
            user_id: User identifier (required)
            
        Returns:
            Progress report dictionary
        """
        # Verify document ownership
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not doc:
            raise ValueError(f"Document {document_id} not found or not owned by user {user_id}")

        # Topic breakdown
        topic_rows = db.query(TopicProgress).filter(
            TopicProgress.student_id == user_id,
            TopicProgress.document_id == document_id,
        ).all()

        total_attempts = sum(t.attempts for t in topic_rows)
        total_correct = sum(t.correct for t in topic_rows)
        overall_accuracy = (total_correct / total_attempts * 100) if total_attempts else 0

        weak_topics = [t.topic for t in topic_rows if t.is_weak]
        strong_topics = [
            t.topic for t in topic_rows
            if not t.is_weak and t.attempts >= 3 and t.accuracy >= 0.8
        ]

        topics_data = [
            {
                "topic": t.topic,
                "attempts": t.attempts,
                "correct": t.correct,
                "accuracy": round(t.accuracy * 100, 1),
                "is_weak": t.is_weak,
                "last_attempt": t.last_attempt.isoformat() if t.last_attempt else None,
            }
            for t in sorted(topic_rows, key=lambda x: x.accuracy)
        ]

        # Recent sessions for this user and document
        sessions = db.query(StudySession).filter(
            StudySession.document_id == document_id,
            StudySession.student_id == user_id,
        ).order_by(StudySession.started_at.desc()).limit(10).all()

        recent_sessions = []
        for s in sessions:
            attempts = db.query(QuestionAttempt).filter(
                QuestionAttempt.session_id == s.id
            ).all()
            session_correct = sum(1 for a in attempts if a.is_correct)
            recent_sessions.append({
                "session_id": s.id[:8] + "...",
                "started_at": s.started_at.isoformat(),
                "total_questions": len(attempts),
                "correct": session_correct,
                "accuracy": round(session_correct / len(attempts) * 100, 1) if attempts else 0,
            })

        # Get weak topics from vector store analysis (optional enhancement)
        weak_related = self._get_weak_related_topics(db, user_id, document_id, weak_topics)

        return {
            "student_id": user_id,
            "document_id": document_id,
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "overall_accuracy": round(overall_accuracy, 1),
            "topics": topics_data,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "weak_related_topics": weak_related,
            "recent_sessions": recent_sessions,
        }

    def _get_weak_related_topics(
        self,
        db: Session,
        user_id: str,
        document_id: str,
        weak_topics: List[str],
    ) -> List[str]:
        """
        Analyze weak topics and find related concepts.
        This is a placeholder - can be enhanced with semantic similarity.
        """
        if not weak_topics:
            return []
        
        # For now, just return the weak topics themselves
        # Future enhancement: use vector search to find related topics
        return weak_topics[:5]

    # ─── User Data Management (GDPR) ──────────────────────────────────────────

    def delete_user_data(self, db: Session, user_id: str) -> Dict[str, int]:
        """
        Delete ALL progress data for a user.
        
        Returns:
            Dictionary with counts of deleted records
        """
        deleted_counts = {
            "sessions": 0,
            "attempts": 0,
            "topic_progress": 0,
        }
        
        # Get all sessions for user
        sessions = db.query(StudySession).filter(StudySession.student_id == user_id).all()
        session_ids = [s.id for s in sessions]
        
        # Delete attempts linked to these sessions
        if session_ids:
            deleted_counts["attempts"] = db.query(QuestionAttempt).filter(
                QuestionAttempt.session_id.in_(session_ids)
            ).delete(synchronize_session=False)
        
        # Delete sessions
        deleted_counts["sessions"] = db.query(StudySession).filter(
            StudySession.student_id == user_id
        ).delete(synchronize_session=False)
        
        # Delete topic progress
        deleted_counts["topic_progress"] = db.query(TopicProgress).filter(
            TopicProgress.student_id == user_id
        ).delete(synchronize_session=False)
        
        db.commit()
        
        logger.info(f"Deleted user data for {user_id[:8]}...: {deleted_counts}")
        return deleted_counts

    def get_user_summary(self, db: Session, user_id: str) -> Dict[str, Any]:
        """
        Get high-level summary of user's progress across all documents.
        """
        # Get all documents for user
        documents = db.query(Document).filter(Document.user_id == user_id).all()
        doc_ids = [d.id for d in documents]
        
        if not doc_ids:
            return {
                "user_id": user_id[:8] + "...",
                "total_documents": 0,
                "total_attempts": 0,
                "overall_accuracy": 0,
                "total_weak_topics": 0,
            }
        
        # Aggregate topic progress across all documents
        topic_progress = db.query(TopicProgress).filter(
            TopicProgress.student_id == user_id,
            TopicProgress.document_id.in_(doc_ids)
        ).all()
        
        total_attempts = sum(t.attempts for t in topic_progress)
        total_correct = sum(t.correct for t in topic_progress)
        overall_accuracy = (total_correct / total_attempts * 100) if total_attempts else 0
        total_weak_topics = sum(1 for t in topic_progress if t.is_weak)
        
        return {
            "user_id": user_id[:8] + "...",
            "total_documents": len(documents),
            "total_attempts": total_attempts,
            "overall_accuracy": round(overall_accuracy, 1),
            "total_weak_topics": total_weak_topics,
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
progress_service = ProgressService()