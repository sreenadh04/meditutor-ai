"""
MediTutor AI - Vector Store Service with User Isolation
FAISS-backed semantic search with local sentence-transformers embeddings.
100% FREE — no API calls needed for embeddings.
Supports per-user isolation and async operations.
"""

import json
import pickle
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import faiss
from sentence_transformers import SentenceTransformer

from config import VECTOR_DIR, EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K_CHUNKS
from services.pdf_service import TextChunk

logger = logging.getLogger(__name__)

# Thread pool for blocking operations (embeddings)
_executor = ThreadPoolExecutor(max_workers=2)


class VectorStoreService:
    """
    Manages FAISS indexes per user + per document.
    
    Directory structure:
        /vectors/{user_id}/{doc_id}.index
        /vectors/{user_id}/{doc_id}.meta.json
    
    Each user's documents are completely isolated from others.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self._indexes: Dict[str, Dict[str, faiss.Index]] = {}  # {user_id: {doc_id: index}}
        self._metadata: Dict[str, Dict[str, List[dict]]] = {}  # {user_id: {doc_id: metadata}}
        self._executor = ThreadPoolExecutor(max_workers=2)
        logger.info("Embedding model ready")

    # ── File paths with user isolation ────────────────────────────────────────
    def _get_user_vector_dir(self, user_id: str) -> Path:
        """Get user-specific vector directory."""
        if not user_id:
            raise ValueError("user_id is required for vector operations")
        
        user_dir = VECTOR_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _index_path(self, user_id: str, doc_id: str) -> Path:
        """Get path to FAISS index file for a user's document."""
        return self._get_user_vector_dir(user_id) / f"{doc_id}.index"

    def _meta_path(self, user_id: str, doc_id: str) -> Path:
        """Get path to metadata file for a user's document."""
        return self._get_user_vector_dir(user_id) / f"{doc_id}.meta.json"

    def _ensure_cached(self, user_id: str, doc_id: str):
        """Ensure index and metadata are cached in memory."""
        if user_id not in self._indexes:
            self._indexes[user_id] = {}
            self._metadata[user_id] = {}
        
        if doc_id not in self._indexes[user_id]:
            self._indexes[user_id][doc_id] = None
            self._metadata[user_id][doc_id] = None

    # ── Async Embedding ───────────────────────────────────────────────────────
    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        """
        Async wrapper for embedding generation.
        Runs the blocking encoder in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_texts_sync,
            texts
        )

    def _embed_texts_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous embedding (runs in thread pool)."""
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        return embeddings.astype("float32")

    # ── Embed (sync version for backward compatibility) ───────────────────────
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Sync version — prefer embed_texts_async in async contexts."""
        return self._embed_texts_sync(texts)

    # ── Build index with user isolation ───────────────────────────────────────
    async def build_index(
        self,
        user_id: str,
        doc_id: str,
        chunks: List[TextChunk],
        use_ivf: bool = False,
    ) -> int:
        """
        Build a FAISS index for a user's document and persist to disk.
        
        Args:
            user_id: Owner of the document
            doc_id: Document identifier
            chunks: List of text chunks to index
            use_ivf: Use IVF index for large documents (>10k chunks)
        
        Returns:
            Number of vectors indexed
        """
        if not user_id:
            raise ValueError("user_id is required")
        
        if not chunks:
            raise ValueError("No chunks to index")

        texts = [c.text for c in chunks]
        metadata = [c.to_dict() for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks for user {user_id[:8]}..., doc {doc_id[:8]}...")
        
        # Async embedding
        embeddings = await self.embed_texts_async(texts)

        # Choose index type based on size
        index = self._create_index(embeddings.shape[1], len(chunks), use_ivf)
        index.add(embeddings)

        # Persist to user-isolated directory
        idx_path = self._index_path(user_id, doc_id)
        meta_path = self._meta_path(user_id, doc_id)
        
        faiss.write_index(index, str(idx_path))
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        # Cache in memory
        self._ensure_cached(user_id, doc_id)
        self._indexes[user_id][doc_id] = index
        self._metadata[user_id][doc_id] = metadata

        logger.info(f"Indexed {index.ntotal} vectors for user {user_id[:8]}..., doc {doc_id[:8]}...")
        return index.ntotal

    def _create_index(self, dimension: int, num_vectors: int, force_ivf: bool = False) -> faiss.Index:
        """
        Create appropriate FAISS index based on number of vectors.
        
        - Flat IP: Small documents (< 10k chunks) — exact search
        - IVF: Large documents (> 10k chunks) — approximate search (faster)
        """
        USE_IVF_THRESHOLD = 10000
        
        if force_ivf or num_vectors > USE_IVF_THRESHOLD:
            # IVF index for large documents
            nlist = min(max(100, num_vectors // 100), 1000)  # 100-1000 centroids
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(self._get_training_data(dimension, nlist * 10))
            logger.info(f"Using IVF index with {nlist} centroids for {num_vectors} vectors")
            return index
        else:
            # Flat index for small documents
            logger.info(f"Using FlatIP index for {num_vectors} vectors")
            return faiss.IndexFlatIP(dimension)

    def _get_training_data(self, dimension: int, num_samples: int) -> np.ndarray:
        """Generate dummy training data for IVF index."""
        return np.random.random((num_samples, dimension)).astype("float32")

    # ── Load index with user isolation ────────────────────────────────────────
    async def load_index(self, user_id: str, doc_id: str) -> bool:
        """
        Load index from disk into memory cache.
        
        Returns:
            True if loaded successfully, False if not found
        """
        if not user_id or not doc_id:
            return False

        # Check memory cache first
        if user_id in self._indexes and doc_id in self._indexes[user_id]:
            if self._indexes[user_id][doc_id] is not None:
                return True

        idx_path = self._index_path(user_id, doc_id)
        meta_path = self._meta_path(user_id, doc_id)

        if not idx_path.exists() or not meta_path.exists():
            return False

        # Load in thread pool (FAISS I/O is blocking)
        loop = asyncio.get_event_loop()
        
        def _load():
            index = faiss.read_index(str(idx_path))
            with open(meta_path) as f:
                metadata = json.load(f)
            return index, metadata

        index, metadata = await loop.run_in_executor(self._executor, _load)

        self._ensure_cached(user_id, doc_id)
        self._indexes[user_id][doc_id] = index
        self._metadata[user_id][doc_id] = metadata

        logger.info(f"Loaded index for user {user_id[:8]}..., doc {doc_id[:8]}...: {index.ntotal} vectors")
        return True

    # ── Verify ownership ──────────────────────────────────────────────────────
    async def verify_ownership(self, user_id: str, doc_id: str) -> bool:
        """
        Verify that a document belongs to a user.
        
        Args:
            user_id: User to check
            doc_id: Document to verify
            
        Returns:
            True if document exists and belongs to user, False otherwise
        """
        if not user_id or not doc_id:
            return False
        
        idx_path = self._index_path(user_id, doc_id)
        return idx_path.exists()

    # ── Search with user isolation ────────────────────────────────────────────
    async def search(
        self,
        user_id: str,
        doc_id: str,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> List[Dict]:
        """
        Semantic search within a user's document.
        
        Args:
            user_id: Owner of the document
            doc_id: Document to search
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunk dicts with relevance score
            
        Raises:
            ValueError: If document not found or not owned by user
        """
        if not user_id or not doc_id:
            raise ValueError("user_id and doc_id are required")

        # Load index if not cached
        if not await self.load_index(user_id, doc_id):
            raise ValueError(f"No index found for document {doc_id}. User {user_id[:8]}... does not own this document or it doesn't exist.")

        index = self._indexes[user_id][doc_id]
        metadata = self._metadata[user_id][doc_id]

        if index is None or metadata is None:
            raise ValueError(f"Index not loaded for doc {doc_id}")

        # Embed query (async)
        q_emb = await self.embed_texts_async([query])

        # Search
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            chunk = metadata[idx].copy()
            chunk["relevance_score"] = float(score)
            results.append(chunk)

        logger.info(f"Search for user {user_id[:8]}..., doc {doc_id[:8]}... returned {len(results)} results")
        return results

    # ── Index existence check with user isolation ─────────────────────────────
    async def index_exists(self, user_id: str, doc_id: str) -> bool:
        """Check if a user has a vector index for a document."""
        if not user_id or not doc_id:
            return False
        return self._index_path(user_id, doc_id).exists()

    # ── Delete index with user isolation ──────────────────────────────────────
    async def delete_index(self, user_id: str, doc_id: str) -> bool:
        """
        Delete a user's vector index for a document.
        
        Returns:
            True if deleted, False if not found
        """
        if not user_id or not doc_id:
            return False

        idx_path = self._index_path(user_id, doc_id)
        meta_path = self._meta_path(user_id, doc_id)
        
        deleted = False
        for path in [idx_path, meta_path]:
            if path.exists():
                path.unlink()
                deleted = True

        # Remove from cache
        if user_id in self._indexes and doc_id in self._indexes[user_id]:
            del self._indexes[user_id][doc_id]
        if user_id in self._metadata and doc_id in self._metadata[user_id]:
            del self._metadata[user_id][doc_id]

        if deleted:
            logger.info(f"Deleted vector index for user {user_id[:8]}..., doc {doc_id[:8]}...")
        
        return deleted

    # ── List user's indexes ───────────────────────────────────────────────────
    async def list_user_indexes(self, user_id: str) -> List[str]:
        """List all document IDs that a user has indexed."""
        if not user_id:
            return []
        
        user_dir = self._get_user_vector_dir(user_id)
        return [p.stem for p in user_dir.glob("*.index")]

    # ─── Delete all user data (GDPR compliance) ────────────────────────────────
    async def delete_user_data(self, user_id: str) -> int:
        """
        Delete ALL vector data for a user.
        
        Args:
            user_id: User to delete
            
        Returns:
            Number of files deleted
        """
        if not user_id:
            return 0
        
        user_dir = self._get_user_vector_dir(user_id)
        if not user_dir.exists():
            return 0
        
        deleted = 0
        for file in user_dir.glob("*"):
            try:
                file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
        
        # Remove from cache
        self._indexes.pop(user_id, None)
        self._metadata.pop(user_id, None)
        
        logger.info(f"Deleted {deleted} vector files for user {user_id[:8]}...")
        return deleted

    # ─── Stats ─────────────────────────────────────────────────────────────────
    async def get_stats(self, user_id: Optional[str] = None) -> dict:
        """Get statistics about vector storage."""
        if user_id:
            user_dir = self._get_user_vector_dir(user_id)
            indexes = await self.list_user_indexes(user_id)
            return {
                "user_id": user_id[:8] + "...",
                "total_documents": len(indexes),
                "documents": indexes,
                "storage_path": str(user_dir),
            }
        
        # Aggregate all users
        users = []
        for user_dir in VECTOR_DIR.iterdir():
            if user_dir.is_dir() and not user_dir.name.startswith("."):
                users.append(await self.get_stats(user_dir.name))
        
        return {
            "total_users": len(users),
            "users": users,
            "base_vector_dir": str(VECTOR_DIR),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
vector_service = VectorStoreService()