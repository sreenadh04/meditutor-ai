"""
MediTutor AI — FastAPI Backend (Production Ready)
Educational AI assistant backend with user isolation, rate limiting, and monitoring.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import APP_NAME, APP_VERSION, ALLOWED_ORIGINS, DEBUG
from database import create_tables, SessionLocal
from routers import pdf_router, qa_router, flashcard_router, mcq_router, progress_router, prereq_router
from services.llm_service import llm_service
from services.vector_service import vector_service

# ─── Logging with graceful request_id handling ──────────────────────────────────

class RequestIdFilter(logging.Filter):
    """Adds request_id to log records, defaults to 'startup' if not present."""
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'startup'
        return True

# Configure basic logging without request_id first
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Add filter to root logger
logging.getLogger().addFilter(RequestIdFilter())

# Reconfigure handlers with request_id format
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | [%(request_id)s] | %(message)s"
        ))

logger = logging.getLogger(__name__)


# ─── Custom Middleware ────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Adds a unique request ID to every request for tracing.
    Reads from X-Request-ID header or generates new one.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Add to logger context using filter approach (safer)
        # Store in contextvars for access in logs
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Extracts user_id from request and injects into request.state.
    Priority: Header > Query param > Session (future)
    """
    async def dispatch(self, request: Request, call_next):
        user_id = None
        
        # Try header first (preferred)
        user_id = request.headers.get("X-User-ID")
        
        # Fallback to query parameter (for GET requests, testing)
        if not user_id:
            user_id = request.query_params.get("user_id")
        
        # For POST/PUT, also check body (careful - consumes body)
        if not user_id and request.method in ["POST", "PUT", "DELETE"]:
            try:
                body = await request.json()
                user_id = body.get("user_id") or body.get("student_id")
            except Exception:
                pass  # Not JSON or no body
        
        # Validate UUID format
        if user_id:
            try:
                # Basic UUID validation (doesn't require actual UUID, just format)
                if len(user_id) >= 32:  # UUID has 32 hex chars + dashes
                    request.state.user_id = user_id
                    request.state.is_authenticated = True
                else:
                    logger.warning(f"Invalid user_id format: {user_id}")
                    request.state.user_id = None
                    request.state.is_authenticated = False
            except Exception:
                request.state.user_id = None
                request.state.is_authenticated = False
        else:
            # For development only - generate temporary user_id
            if DEBUG:
                temp_id = f"dev_{uuid.uuid4().hex[:16]}"
                logger.warning(f"No user_id provided, using temp: {temp_id}")
                request.state.user_id = temp_id
                request.state.is_authenticated = True
            else:
                request.state.user_id = None
                request.state.is_authenticated = False
        
        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-user rate limiting to prevent abuse.
    Simple in-memory implementation (use Redis for production).
    """
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._requests: dict = {}  # {user_id: [(timestamp, count)]}
    
    async def dispatch(self, request: Request, call_next):
        user_id = getattr(request.state, "user_id", None)
        
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        if user_id:
            now = time.time()
            window_start = now - 60
            
            # Clean old entries
            if user_id in self._requests:
                self._requests[user_id] = [
                    (ts, c) for ts, c in self._requests[user_id] if ts > window_start
                ]
            else:
                self._requests[user_id] = []
            
            # Count requests in last minute
            request_count = sum(c for ts, c in self._requests[user_id])
            
            if request_count >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for user {user_id[:8]}...")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded. Max {self.requests_per_minute} requests per minute.",
                        "retry_after": 60,
                    }
                )
            
            # Add current request
            self._requests[user_id].append((now, 1))
        
        response = await call_next(request)
        return response


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logger.info(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    
    # Initialize database
    create_tables()
    logger.info("✅ Database tables ready")
    
    # Check LLM availability
    avail = await llm_service.check_availability()
    if avail["groq"]["configured"]:
        logger.info("✅ Groq API configured")
    else:
        logger.warning("⚠️  Groq API key not set — will use HuggingFace only")
    
    if avail["huggingface"]["configured"]:
        logger.info("✅ HuggingFace API configured")
    else:
        logger.warning("⚠️  HuggingFace API key not set")
    
    # Check vector store
    stats = await vector_service.get_stats()
    logger.info(f"✅ Vector store ready — {stats.get('total_users', 0)} users")
    
    yield  # App running
    
    # Shutdown
    logger.info("🛑 Shutting down MediTutor AI")
    
    # Close database connections
    try:
        SessionLocal().close_all()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    # Clean up thread pools
    try:
        from services.vector_service import _executor
        _executor.shutdown(wait=True)
        logger.info("✅ Thread pools cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up threads: {e}")


# ─── App Instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI-powered study assistant with RAG, flashcards, MCQs, and progress tracking. Features full user isolation for multi-tenant support.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Middleware (order matters!) ──────────────────────────────────────────────
# 1. Request ID (earliest for tracing)
app.add_middleware(RequestIDMiddleware)

# 2. Auth (needs to run before rate limiting)
app.add_middleware(AuthMiddleware)

# 3. Rate limiting (depends on auth)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# 4. CORS (handle cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # NO MORE ["*"] in production!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-User-ID", "X-Request-ID", "Content-Type", "Authorization"],
)

# 5. Compression (after auth, before response)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─── Custom Middleware: Timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start):.3f}s"
    return response


# ─── Global error handler ─────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    user_id = getattr(request.state, "user_id", "unknown")
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled error: {exc} | User: {user_id[:8] if user_id != 'unknown' else 'unknown'}... | URL: {request.url}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again.",
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    user_id = getattr(request.state, "user_id", "unknown")
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} | User: {user_id[:8] if user_id != 'unknown' else 'unknown'}...")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(pdf_router.router, prefix="/api/v1")
app.include_router(qa_router.router, prefix="/api/v1")
app.include_router(flashcard_router.router, prefix="/api/v1")
app.include_router(mcq_router.router, prefix="/api/v1")
app.include_router(progress_router.router, prefix="/api/v1")
app.include_router(prereq_router.router, prefix="/api/v1")


# ─── Enhanced Health Check ────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health(request: Request):
    """Comprehensive health check including all dependencies."""
    health_status = {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "timestamp": time.time(),
    }
    
    # Check LLM
    try:
        llm_avail = await llm_service.check_availability()
        health_status["llm"] = llm_avail
    except Exception as e:
        health_status["llm"] = {"error": str(e)}
        health_status["status"] = "degraded"
    
    # Check vector store
    try:
        vector_stats = await vector_service.get_stats()
        health_status["vector_store"] = {
            "available": True,
            "total_users": vector_stats.get("total_users", 0),
        }
    except Exception as e:
        health_status["vector_store"] = {"available": False, "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check database
    try:
        from database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["database"] = {"available": True}
    except Exception as e:
        health_status["database"] = {"available": False, "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check auth middleware status
    user_id = getattr(request.state, "user_id", None)
    health_status["auth"] = {
        "user_id_provided": user_id is not None,
        "mode": "authenticated" if user_id else "anonymous",
    }
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


# ─── User Management Endpoints (GDPR) ─────────────────────────────────────────
@app.delete("/api/v1/user/data")
async def delete_user_data(request: Request):
    """
    Delete ALL data for the current user (GDPR compliance).
    Requires X-User-ID header.
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")
    
    if user_id.startswith("dev_"):
        raise HTTPException(status_code=403, detail="Cannot delete development user data")
    
    deleted = {
        "vector_files": 0,
        "cache_files": 0,
        "database_records": 0,
    }
    
    # Delete vector data
    deleted["vector_files"] = await vector_service.delete_user_data(user_id)
    
    # Delete cache data
    from utils.cache import get_cache
    cache = get_cache()
    deleted["cache_files"] = cache.clear_user_cache(user_id)
    
    # Delete database records
    from database import SessionLocal, Document
    db = SessionLocal()
    try:
        # Delete documents (cascade should handle related records)
        docs = db.query(Document).filter(Document.user_id == user_id).all()
        deleted["database_records"] = len(docs)
        for doc in docs:
            db.delete(doc)
        db.commit()
    except Exception as e:
        logger.error(f"Error deleting user data from DB: {e}")
        db.rollback()
    finally:
        db.close()
    
    logger.info(f"Deleted user data for {user_id[:8]}...: {deleted}")
    
    return {
        "message": "User data deleted successfully",
        "deleted": deleted,
        "user_id": user_id[:8] + "...",
    }


@app.get("/api/v1/user/stats")
async def get_user_stats(request: Request):
    """Get storage statistics for the current user."""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header required")
    
    # Get vector stats
    vector_docs = await vector_service.list_user_indexes(user_id)
    
    # Get cache stats
    from utils.cache import get_cache
    cache = get_cache()
    cache_stats = cache.stats(user_id)
    
    return {
        "user_id": user_id[:8] + "...",
        "documents": {
            "count": len(vector_docs),
            "ids": vector_docs[:10],  # Limit for response size
        },
        "cache": cache_stats,
        "storage_path": f"/data/vectors/{user_id}",
    }


# ─── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )