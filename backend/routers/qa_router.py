"""
MediTutor AI - Q&A Router with User Isolation
RAG-based question answering with source citations and ownership verification.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from database import get_db, Document
from models import QARequest, QAResponse, SourceChunk
from services.vector_service import vector_service
from services.llm_service import llm_service
from services.progress_service import progress_service
from config import MAX_CONTEXT_LENGTH, TOP_K_CHUNKS
from utils.cache import get_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["Q&A"])
cache = get_cache()


QA_SYSTEM_PROMPT = """You are MediTutor AI, an expert educational assistant.
Answer questions accurately and clearly based on the provided study material.
Always base your answer on the given context. If the context doesn't contain enough
information, say so clearly. Be concise but thorough."""


def _build_qa_prompt(question: str, chunks: list) -> str:
    """Build the prompt for LLM with context chunks."""
    context_parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        page = chunk.get("page_number", "?")
        if total + len(text) > MAX_CONTEXT_LENGTH:
            break
        context_parts.append(f"[Source {i} - Page {page}]:\n{text}")
        total += len(text)

    context = "\n\n".join(context_parts)
    return f"""Answer the following question using ONLY the context provided below.
If the context doesn't contain the answer, say "The provided material doesn't cover this topic."

QUESTION: {question}

CONTEXT:
{context}

Provide a clear, well-structured answer. Reference source numbers where relevant (e.g., "According to Source 1...").
"""


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
    
    # Also verify vector index exists
    index_exists = await vector_service.index_exists(user_id, document_id)
    if not index_exists:
        raise HTTPException(
            status_code=404,
            detail="Document index not found. Please re-upload the PDF."
        )
    
    return doc


@router.post("/ask", response_model=QAResponse)
async def ask_question(
    request: Request,
    qa_request: QARequest,
    db: Session = Depends(get_db),
):
    """
    RAG-based Q&A with source citations and user isolation.
    Retrieves relevant chunks and generates a grounded answer.
    """
    # Extract user_id from middleware
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header required."
        )
    
    # Verify document ownership
    doc = await verify_document_ownership(db, user_id, qa_request.document_id)
    
    # Check cache with user isolation
    cache_key = f"qa:{qa_request.document_id}:{hash(qa_request.question)}"
    cached = cache.get(user_id, cache_key)
    if cached:
        logger.info(f"Cache hit for user {user_id[:8]}..., doc {qa_request.document_id[:8]}...")
        return QAResponse(**cached, cached=True)

    # Retrieve relevant chunks (with user isolation)
    try:
        chunks = await vector_service.search(
            user_id=user_id,
            doc_id=qa_request.document_id,
            query=qa_request.question,
            top_k=TOP_K_CHUNKS,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No relevant content found. Try rephrasing your question or uploading more content."
        )

    # Build prompt and generate answer
    prompt = _build_qa_prompt(qa_request.question, chunks)
    try:
        answer, model_used = await llm_service.generate(
            prompt=prompt,
            system=QA_SYSTEM_PROMPT,
            use_cache=False,  # We cache at router level with user isolation
        )
    except Exception as e:
        logger.error(f"LLM generation failed for user {user_id[:8]}...: {e}")
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)}")

    # Build source list
    sources = [
        SourceChunk(
            text=c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            page_number=c.get("page_number"),
            chunk_index=c.get("chunk_index", i),
            relevance_score=c.get("relevance_score", 0.0),
        )
        for i, c in enumerate(chunks[:5])  # Limit to 5 sources in response
    ]

    result = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "model_used": model_used,
        "cached": False,
    }
    
    # Cache with user isolation
    cache.set(user_id, cache_key, result)

    # Record attempt if session provided
    if qa_request.session_id:
        try:
            progress_service.record_attempt(
                db=db,
                session_id=qa_request.session_id,
                question_text=qa_request.question,
                question_type="qa",
                topic=None,
                user_answer=None,
                correct_answer=None,
                is_correct=None,
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to record attempt for user {user_id[:8]}...: {e}")
            # Don't fail the request if tracking fails

    logger.info(
        f"Q&A completed for user {user_id[:8]}..., "
        f"doc {qa_request.document_id[:8]}..., "
        f"model: {model_used}, "
        f"sources: {len(sources)}"
    )

    return QAResponse(**result)


@router.post("/ask/batch")
async def ask_batch(
    request: Request,
    questions: list[str],
    document_id: str,
    db: Session = Depends(get_db),
):
    """
    Batch ask multiple questions (for efficiency).
    Returns list of answers in the same order.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership once
    await verify_document_ownership(db, user_id, document_id)
    
    results = []
    for question in questions:
        try:
            # Create a temporary request object
            qa_request = QARequest(document_id=document_id, question=question)
            
            # Reuse the single ask logic
            chunks = await vector_service.search(
                user_id=user_id,
                doc_id=document_id,
                query=question,
                top_k=TOP_K_CHUNKS,
            )
            
            if not chunks:
                results.append({"question": question, "answer": "No relevant content found.", "sources": []})
                continue
            
            prompt = _build_qa_prompt(question, chunks)
            answer, model_used = await llm_service.generate(
                prompt=prompt,
                system=QA_SYSTEM_PROMPT,
                use_cache=True,
            )
            
            results.append({
                "question": question,
                "answer": answer,
                "model_used": model_used,
                "sources_count": len(chunks),
            })
        except Exception as e:
            logger.error(f"Batch Q&A failed for question '{question[:50]}...': {e}")
            results.append({"question": question, "answer": f"Error: {str(e)}", "sources": []})
    
    return {"results": results, "total": len(results)}


@router.get("/suggestions/{document_id}")
async def get_suggested_questions(
    request: Request,
    document_id: str,
    limit: int = 5,
    db: Session = Depends(get_db),
):
    """
    Generate suggested questions based on document content.
    Useful for helping users get started.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, document_id)
    
    # Get a sample of chunks to generate questions
    try:
        # Use a generic query to get top chunks
        chunks = await vector_service.search(
            user_id=user_id,
            doc_id=document_id,
            query="key concepts main topics important information",
            top_k=10,
        )
    except Exception as e:
        logger.error(f"Failed to get chunks for suggestions: {e}")
        raise HTTPException(status_code=500, detail="Could not generate suggestions")
    
    if not chunks:
        return {"suggestions": ["What is the main topic of this document?"]}
    
    # Build context from chunks
    context = "\n\n".join([c["text"][:500] for c in chunks[:5]])
    
    suggestion_prompt = f"""Based on this document content, generate {limit} good questions that a student might ask.
Return ONLY a JSON array of strings, no other text.

Content:
{context}

Questions:"""
    
    try:
        raw, model = await llm_service.generate(
            prompt=suggestion_prompt,
            system="You are a helpful assistant that generates study questions.",
            max_tokens=300,
        )
        
        # Parse JSON array
        import json
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            suggestions = suggestions[:limit]
        else:
            suggestions = ["What are the key concepts in this document?"]
            
    except Exception as e:
        logger.warning(f"Failed to generate suggestions: {e}")
        suggestions = [
            "What are the main topics covered?",
            "Can you summarize the key points?",
            "What are the important definitions?",
            "Explain the core concepts in simple terms.",
        ][:limit]
    
    return {"suggestions": suggestions, "model_used": model if 'model' in locals() else "fallback"}


@router.get("/context/{document_id}")
async def get_document_context(
    request: Request,
    document_id: str,
    query: Optional[str] = None,
    top_k: int = 3,
    db: Session = Depends(get_db),
):
    """
    Debug endpoint: Get raw chunks for a query.
    Returns the actual context that would be sent to LLM.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required.")
    
    # Verify ownership
    await verify_document_ownership(db, user_id, document_id)
    
    search_query = query or "sample content"
    chunks = await vector_service.search(
        user_id=user_id,
        doc_id=document_id,
        query=search_query,
        top_k=top_k,
    )
    
    return {
        "document_id": document_id,
        "query": search_query,
        "chunks_found": len(chunks),
        "chunks": [
            {
                "text": c["text"][:500],
                "page": c.get("page_number"),
                "score": c.get("relevance_score"),
            }
            for c in chunks
        ],
    }