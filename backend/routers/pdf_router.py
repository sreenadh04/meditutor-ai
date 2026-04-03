"""
MediTutor AI - PDF Router with User Isolation
Upload, process, and manage PDF documents with full user isolation.
"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query, Request
from sqlalchemy.orm import Session

from database import get_db, Document
from models import DocumentResponse, DocumentListResponse
from services.pdf_service import pdf_service
from services.vector_service import vector_service
from config import MAX_PDF_SIZE_MB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF"])

MAX_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


def get_user_id_from_request(request: Request) -> str:
    """
    Extract user_id from request state (set by AuthMiddleware).
    Raises HTTPException if not found.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header required. Please provide a valid user identifier."
        )
    
    # Basic validation
    if len(user_id) < 8:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format. Must be at least 8 characters."
        )
    
    return user_id


@router.post("/upload", response_model=DocumentResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload and process a PDF:
    1. Extract text page-by-page
    2. Split into overlapping chunks
    3. Embed and store in FAISS (user-isolated)
    4. Save document record to SQLite with user_id
    """
    user_id = get_user_id_from_request(request)
    
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    # Validate size
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_PDF_SIZE_MB} MB."
        )

    if len(content) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty.")

    try:
        # Process PDF (returns doc_id and chunks)
        doc_id, chunks, stats = pdf_service.process_pdf(content, file.filename)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be image-only (scanned)."
            )

        # Build vector index with user isolation
        await vector_service.build_index(user_id, doc_id, chunks)

        # Save to database with user_id
        doc = Document(
            id=doc_id,
            filename=file.filename,
            total_pages=stats.get("total_pages", 0),
            total_chunks=len(chunks),
            vector_store_path=f"{user_id}/{doc_id}",
            user_id=user_id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        logger.info(
            f"Uploaded and indexed: {file.filename} | "
            f"User: {user_id[:8]}... | Doc: {doc_id[:8]}... | "
            f"Chunks: {len(chunks)}"
        )
        
        return DocumentResponse.model_validate(doc)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"ValueError during upload for user {user_id[:8]}...: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error for user {user_id[:8]}...: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    db: Session = Depends(get_db),
):
    """List all documents uploaded by the current user."""
    user_id = get_user_id_from_request(request)
    
    docs = db.query(Document).filter(
        Document.user_id == user_id
    ).order_by(Document.created_at.desc()).all()
    
    logger.info(f"Listed {len(docs)} documents for user {user_id[:8]}...")
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Get a single document's details.
    Verifies that the document belongs to the requesting user.
    """
    user_id = get_user_id_from_request(request)
    
    # Query with user_id filter (security)
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=404, 
            detail="Document not found or you don't have access to it."
        )
    
    # Verify vector index exists
    index_exists = await vector_service.index_exists(user_id, doc_id)
    if not index_exists:
        logger.warning(f"Document {doc_id[:8]}... exists in DB but vector index missing for user {user_id[:8]}...")
    
    return DocumentResponse.model_validate(doc)


@router.delete("/{doc_id}")
async def delete_document(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete a document and its vector index.
    Verifies ownership before deletion.
    """
    user_id = get_user_id_from_request(request)
    
    # Query with user_id filter (security)
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you don't have permission to delete it."
        )

    # Delete vector index (user-isolated)
    deleted = await vector_service.delete_index(user_id, doc_id)
    
    # Delete from database
    db.delete(doc)
    db.commit()
    
    logger.info(
        f"Deleted document: {doc.filename} | "
        f"User: {user_id[:8]}... | Doc: {doc_id[:8]}... | "
        f"Vector deleted: {deleted}"
    )
    
    return {
        "message": "Document deleted successfully.",
        "id": doc_id,
        "filename": doc.filename,
        "vector_deleted": deleted,
    }


@router.head("/{doc_id}/exists")
async def check_document_exists(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Quick check if a document exists and belongs to the user.
    Returns 200 if exists, 404 otherwise.
    """
    user_id = get_user_id_from_request(request)
    
    exists = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first() is not None
    
    if not exists:
        raise HTTPException(status_code=404)
    
    return {"exists": True}


@router.post("/{doc_id}/reprocess")
async def reprocess_document(
    request: Request,
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    Reprocess a document (rebuild vector index).
    Useful if chunks are corrupted or need re-indexing.
    """
    user_id = get_user_id_from_request(request)
    
    # Get document with ownership check
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    # Need to reload original file
    # This requires storing the original file path
    from pathlib import Path
    from config import UPLOAD_DIR
    
    # Find original file (format: {doc_id}_{filename})
    upload_pattern = f"{doc_id}_*"
    matching_files = list(UPLOAD_DIR.glob(upload_pattern))
    
    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Original PDF file not found. Please re-upload."
        )
    
    original_file = matching_files[0]
    
    # Reprocess
    try:
        with open(original_file, "rb") as f:
            content = f.read()
        
        # Process PDF
        new_doc_id, chunks, stats = pdf_service.process_pdf(content, doc.filename)
        
        # Rebuild vector index
        await vector_service.delete_index(user_id, doc_id)
        await vector_service.build_index(user_id, doc_id, chunks)
        
        # Update document stats
        doc.total_pages = stats.get("total_pages", 0)
        doc.total_chunks = len(chunks)
        doc.vector_store_path = f"{user_id}/{doc_id}"
        db.commit()
        
        logger.info(f"Reprocessed document {doc_id[:8]}... for user {user_id[:8]}...")
        
        return {
            "message": "Document reprocessed successfully",
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "total_pages": stats.get("total_pages", 0),
        }
        
    except Exception as e:
        logger.error(f"Reprocess error for {doc_id[:8]}...: {e}")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


@router.get("/stats/summary")
async def get_user_stats(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Get storage statistics for the current user.
    """
    user_id = get_user_id_from_request(request)
    
    # Database stats
    doc_count = db.query(Document).filter(Document.user_id == user_id).count()
    total_chunks = db.query(Document).filter(
        Document.user_id == user_id
    ).with_entities(Document.total_chunks).all()
    total_chunks_sum = sum(c[0] for c in total_chunks if c[0])
    
    # Vector store stats
    vector_docs = await vector_service.list_user_indexes(user_id)
    
    return {
        "user_id": user_id[:8] + "...",
        "documents": {
            "total": doc_count,
            "ids": vector_docs[:10],
        },
        "storage": {
            "total_chunks": total_chunks_sum,
            "approx_size_kb": total_chunks_sum * 1.5,  # Rough estimate
        },
    }