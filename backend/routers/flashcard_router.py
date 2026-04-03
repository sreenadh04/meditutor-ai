"""
MediTutor AI - Flashcard Router with User Isolation
Generate, manage, and export flashcards with full user isolation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from database import get_db, Document, Flashcard
from models import FlashcardRequest, FlashcardResponse, FlashcardItem
from services.flashcard_service import flashcard_service
from services.vector_service import vector_service
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/flashcards", tags=["Flashcards"])


async def verify_document_ownership(
    db: Session,
    user_id: str,
    document_id: str,
) -> Document:
    """
    Verify that a document exists and belongs to the user.
    Returns the document if valid, raises HTTPException otherwise.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you don't have access to it."
        )
    
    # Also verify vector index exists (for generation)
    index_exists = await vector_service.index_exists(user_id, document_id)
    if not index_exists:
        raise HTTPException(
            status_code=404,
            detail="Document index not found. Please re-upload the PDF."
        )
    
    return doc


@router.post("/generate", response_model=FlashcardResponse)
async def generate_flashcards(
    request: Request,
    flashcard_request: FlashcardRequest,
    db: Session = Depends(get_db),
):
    """
    Generate flashcards for a document or specific topic.
    Verifies ownership before generation.
    """
    # Extract user_id from middleware
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header required."
        )
    
    # Verify document ownership
    doc = await verify_document_ownership(db, user_id, flashcard_request.document_id)
    
    try:
        cards, model_used = await flashcard_service.generate(
            user_id=user_id,  # ← NEW: pass user_id for cache isolation
            doc_id=flashcard_request.document_id,
            count=flashcard_request.count,
            topic=flashcard_request.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Flashcard generation failed for user {user_id[:8]}...: {e}")
        raise HTTPException(status_code=503, detail=f"Generation failed: {str(e)}")

    # Save to DB with user context (flashcards table doesn't have user_id,
    # but we link through document which has user_id)
    for card in cards:
        fc = Flashcard(
            id=card["id"],
            document_id=flashcard_request.document_id,
            question=card["question"],
            answer=card["answer"],
            topic=card.get("topic"),
            difficulty=card.get("difficulty", "medium"),
        )
        db.merge(fc)
    db.commit()

    cached = "(cached)" in model_used
    logger.info(
        f"Generated {len(cards)} flashcards for user {user_id[:8]}..., "
        f"doc {flashcard_request.document_id[:8]}..., "
        f"model: {model_used}"
    )

    return FlashcardResponse(
        flashcards=[FlashcardItem(**c) for c in cards],
        document_id=flashcard_request.document_id,
        total_generated=len(cards),
        model_used=model_used,
        cached=cached,
    )


@router.get("/list/{doc_id}")
async def list_flashcards(
    request: Request,
    doc_id: str,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
    db: Session = Depends(get_db),
):
    """
    List saved flashcards for a document (paginated).
    Verifies ownership before returning.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, doc_id)
    
    # Get flashcards (ownership implied via document)
    flashcards = db.query(Flashcard).filter(
        Flashcard.document_id == doc_id
    ).offset(offset).limit(limit).all()
    
    total = db.query(Flashcard).filter(Flashcard.document_id == doc_id).count()
    
    return {
        "flashcards": [
            {
                "id": f.id,
                "question": f.question,
                "answer": f.answer,
                "topic": f.topic,
                "difficulty": f.difficulty,
                "review_count": f.review_count,
                "last_reviewed": f.last_reviewed.isoformat() if f.last_reviewed else None,
            }
            for f in flashcards
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{flashcard_id}")
async def get_flashcard(
    request: Request,
    flashcard_id: str,
    db: Session = Depends(get_db),
):
    """
    Get a single flashcard by ID.
    Verifies ownership through document relationship.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    flashcard = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found.")
    
    # Verify ownership through document
    doc = db.query(Document).filter(
        Document.id == flashcard.document_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    return {
        "id": flashcard.id,
        "question": flashcard.question,
        "answer": flashcard.answer,
        "topic": flashcard.topic,
        "difficulty": flashcard.difficulty,
        "review_count": flashcard.review_count,
        "last_reviewed": flashcard.last_reviewed,
        "created_at": flashcard.created_at,
    }


@router.post("/{flashcard_id}/review")
async def review_flashcard(
    request: Request,
    flashcard_id: str,
    difficulty: str,  # "easy", "medium", "hard"
    db: Session = Depends(get_db),
):
    """
    Record a flashcard review (update review count and difficulty).
    Implements spaced repetition tracking.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    if difficulty not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Difficulty must be easy, medium, or hard")
    
    flashcard = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found.")
    
    # Verify ownership
    doc = db.query(Document).filter(
        Document.id == flashcard.document_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    # Update flashcard
    from datetime import datetime
    flashcard.review_count += 1
    flashcard.last_reviewed = datetime.utcnow()
    
    # Update difficulty based on performance (optional)
    if difficulty == "easy" and flashcard.difficulty == "hard":
        flashcard.difficulty = "medium"
    elif difficulty == "easy" and flashcard.difficulty == "medium":
        flashcard.difficulty = "easy"
    elif difficulty == "hard" and flashcard.difficulty == "easy":
        flashcard.difficulty = "medium"
    elif difficulty == "hard" and flashcard.difficulty == "medium":
        flashcard.difficulty = "hard"
    
    db.commit()
    
    logger.info(f"Flashcard {flashcard_id[:8]}... reviewed as {difficulty} by user {user_id[:8]}...")
    
    return {
        "message": "Review recorded",
        "flashcard_id": flashcard_id,
        "difficulty": flashcard.difficulty,
        "review_count": flashcard.review_count,
        "last_reviewed": flashcard.last_reviewed.isoformat(),
    }


@router.get("/export/{doc_id}")
async def export_flashcards_csv(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Export saved flashcards as Anki-compatible CSV.
    Verifies ownership before export.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, doc_id)
    
    # Get flashcards for this document
    cards = db.query(Flashcard).filter(Flashcard.document_id == doc_id).all()
    
    if not cards:
        raise HTTPException(
            status_code=404, 
            detail="No flashcards found for this document. Generate some first."
        )
    
    # Prepare for export
    card_dicts = [
        {
            "question": c.question,
            "answer": c.answer,
            "topic": c.topic or "General",
            "difficulty": c.difficulty,
        }
        for c in cards
    ]
    
    csv_content = flashcard_service.to_anki_csv(card_dicts)
    
    logger.info(f"Exported {len(cards)} flashcards for user {user_id[:8]}..., doc {doc_id[:8]}...")
    
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=flashcards_{doc_id[:8]}_{user_id[:8]}.csv"
        }
    )


@router.delete("/{flashcard_id}")
async def delete_flashcard(
    request: Request,
    flashcard_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete a single flashcard.
    Verifies ownership through document.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    flashcard = db.query(Flashcard).filter(Flashcard.id == flashcard_id).first()
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found.")
    
    # Verify ownership
    doc = db.query(Document).filter(
        Document.id == flashcard.document_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    db.delete(flashcard)
    db.commit()
    
    logger.info(f"Deleted flashcard {flashcard_id[:8]}... for user {user_id[:8]}...")
    
    return {"message": "Flashcard deleted successfully", "flashcard_id": flashcard_id}


@router.delete("/document/{doc_id}/all")
async def delete_all_flashcards(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete ALL flashcards for a document.
    Verifies ownership before deletion.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, doc_id)
    
    # Delete all flashcards for this document
    deleted = db.query(Flashcard).filter(Flashcard.document_id == doc_id).delete()
    db.commit()
    
    logger.info(f"Deleted {deleted} flashcards for user {user_id[:8]}..., doc {doc_id[:8]}...")
    
    return {
        "message": f"Deleted {deleted} flashcards",
        "document_id": doc_id,
        "deleted_count": deleted,
    }


@router.get("/stats/{doc_id}")
async def get_flashcard_stats(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Get statistics about flashcards for a document.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, doc_id)
    
    cards = db.query(Flashcard).filter(Flashcard.document_id == doc_id).all()
    
    if not cards:
        return {
            "document_id": doc_id,
            "total_flashcards": 0,
            "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
            "by_topic": {},
            "total_reviews": 0,
        }
    
    # Calculate statistics
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    topic_counts = {}
    total_reviews = 0
    
    for card in cards:
        difficulty_counts[card.difficulty] = difficulty_counts.get(card.difficulty, 0) + 1
        topic = card.topic or "General"
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        total_reviews += card.review_count
    
    return {
        "document_id": doc_id,
        "total_flashcards": len(cards),
        "by_difficulty": difficulty_counts,
        "by_topic": topic_counts,
        "total_reviews": total_reviews,
        "average_reviews_per_card": round(total_reviews / len(cards), 1) if cards else 0,
    }