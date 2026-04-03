"""
MediTutor AI - Cache Manager with User Isolation
Disk-based cache with TTL support, per-user directories, and LRU eviction.
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from config import CACHE_DIR, CACHE_TTL, MAX_CACHE_SIZE

logger = logging.getLogger(__name__)


class CacheManager:
    """
    User-isolated JSON-based disk cache with TTL and LRU eviction.
    
    Directory structure:
        /cache/{user_id}/{cache_key_hash}.json
    
    Each user has their own cache directory, preventing cross-user data leakage.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        self.base_cache_dir = Path(cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._indexes: Dict[str, Dict] = {}  # {user_id: {key: metadata}}

    def _get_user_cache_dir(self, user_id: str) -> Path:
        """Get or create user-specific cache directory."""
        if not user_id:
            raise ValueError("user_id is required for cache operations")
        
        user_dir = self.base_cache_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_index_file(self, user_id: str) -> Path:
        """Get path to user's index file."""
        return self._get_user_cache_dir(user_id) / "_index.json"

    def _load_index(self, user_id: str) -> Dict:
        """Load user's cache index."""
        if user_id not in self._indexes:
            index_file = self._get_index_file(user_id)
            if index_file.exists():
                try:
                    with open(index_file) as f:
                        self._indexes[user_id] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache index for {user_id}: {e}")
                    self._indexes[user_id] = {}
            else:
                self._indexes[user_id] = {}
        return self._indexes[user_id]

    def _save_index(self, user_id: str):
        """Save user's cache index."""
        if user_id in self._indexes:
            index_file = self._get_index_file(user_id)
            try:
                with open(index_file, "w") as f:
                    json.dump(self._indexes[user_id], f)
            except Exception as e:
                logger.warning(f"Failed to save cache index for {user_id}: {e}")

    def _key_to_path(self, user_id: str, key: str) -> Path:
        """Convert cache key to file path (with user isolation)."""
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._get_user_cache_dir(user_id) / f"{safe}.json"

    def get(self, user_id: str, key: str) -> Optional[Any]:
        """
        Get cached value for a specific user.
        
        Args:
            user_id: User identifier (UUID)
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not user_id:
            logger.warning("get() called without user_id — skipping cache")
            return None

        index = self._load_index(user_id)
        
        if key not in index:
            return None
        
        entry = index[key]
        if time.time() > entry.get("expires", 0):
            self.delete(user_id, key)
            return None
        
        cache_file = self._key_to_path(user_id, key)
        if not cache_file.exists():
            self.delete(user_id, key)
            return None
        
        try:
            with open(cache_file) as f:
                data = json.load(f)
            
            # Update access time for LRU
            index[key]["last_access"] = time.time()
            self._save_index(user_id)
            
            return data.get("value")
        except Exception as e:
            logger.warning(f"Cache read error for user {user_id}, key {key}: {e}")
            return None

    def set(self, user_id: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached value for a specific user.
        
        Args:
            user_id: User identifier (UUID)
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (optional, uses default)
            
        Returns:
            True if successful, False otherwise
        """
        if not user_id:
            logger.warning("set() called without user_id — skipping cache")
            return False

        try:
            index = self._load_index(user_id)
            
            # Evict if at capacity
            if len(index) >= MAX_CACHE_SIZE:
                self._evict_oldest(user_id)

            expire_time = time.time() + (ttl or self.ttl)
            cache_file = self._key_to_path(user_id, key)
            
            with open(cache_file, "w") as f:
                json.dump({"value": value, "key": key, "user_id": user_id}, f)
            
            index[key] = {
                "expires": expire_time,
                "last_access": time.time(),
                "file": str(cache_file),
            }
            self._save_index(user_id)
            return True
        except Exception as e:
            logger.warning(f"Cache write error for user {user_id}, key {key}: {e}")
            return False

    def delete(self, user_id: str, key: str):
        """Delete a specific cache entry for a user."""
        if not user_id:
            return

        index = self._load_index(user_id)
        if key in index:
            cache_file = self._key_to_path(user_id, key)
            if cache_file.exists():
                cache_file.unlink()
            del index[key]
            self._save_index(user_id)

    def clear_user_cache(self, user_id: str) -> int:
        """
        Clear ALL cache entries for a specific user.
        
        Args:
            user_id: User identifier (UUID)
            
        Returns:
            Number of files deleted
        """
        if not user_id:
            return 0

        user_dir = self._get_user_cache_dir(user_id)
        if not user_dir.exists():
            return 0
        
        deleted = 0
        for file in user_dir.glob("*.json"):
            try:
                file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
        
        # Clear in-memory index
        if user_id in self._indexes:
            del self._indexes[user_id]
        
        logger.info(f"Cleared {deleted} cache files for user {user_id[:8]}...")
        return deleted

    def _evict_oldest(self, user_id: str):
        """Evict least recently used cache entry for a user."""
        index = self._load_index(user_id)
        if not index:
            return
        
        oldest_key = min(
            index.keys(),
            key=lambda k: index[k].get("last_access", 0)
        )
        self.delete(user_id, oldest_key)

    def clear_expired(self, user_id: Optional[str] = None) -> int:
        """
        Clear expired cache entries.
        
        Args:
            user_id: If provided, clear only for this user. Otherwise clear all.
            
        Returns:
            Number of entries cleared
        """
        if user_id:
            return self._clear_expired_for_user(user_id)
        
        # Clear for all users
        total = 0
        for user_dir in self.base_cache_dir.iterdir():
            if user_dir.is_dir() and not user_dir.name.startswith("."):
                total += self._clear_expired_for_user(user_dir.name)
        return total

    def _clear_expired_for_user(self, user_id: str) -> int:
        """Clear expired entries for a specific user."""
        index = self._load_index(user_id)
        now = time.time()
        expired = [k for k, v in index.items() if v.get("expires", 0) < now]
        for k in expired:
            self.delete(user_id, k)
        return len(expired)

    def stats(self, user_id: Optional[str] = None) -> dict:
        """
        Get cache statistics.
        
        Args:
            user_id: If provided, get stats for specific user.
            
        Returns:
            Dictionary with cache statistics
        """
        if user_id:
            index = self._load_index(user_id)
            user_dir = self._get_user_cache_dir(user_id)
            return {
                "user_id": user_id[:8] + "...",
                "total_items": len(index),
                "cache_dir": str(user_dir),
                "ttl_seconds": self.ttl,
                "max_size": MAX_CACHE_SIZE,
            }
        
        # Aggregate stats for all users
        users = []
        for user_dir in self.base_cache_dir.iterdir():
            if user_dir.is_dir() and not user_dir.name.startswith("."):
                users.append(self.stats(user_dir.name))
        
        return {
            "total_users": len(users),
            "users": users,
            "base_cache_dir": str(self.base_cache_dir),
        }

    def user_exists(self, user_id: str) -> bool:
        """Check if a user has any cached data."""
        if not user_id:
            return False
        user_dir = self._get_user_cache_dir(user_id)
        return user_dir.exists() and any(user_dir.glob("*.json"))


# ─── Singleton instance (maintained for backward compatibility) ────────────────
_cache_instance = None


def get_cache() -> CacheManager:
    """Get singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


# DEPRECATED: Use get_cache() instead and pass user_id to all methods
cache = get_cache()