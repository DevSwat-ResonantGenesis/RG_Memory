"""
Semantic Query Cache Service
Cache query results for instant responses on repeated queries.

Uses in-memory cache with optional Redis backend for distributed caching.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import List, Dict, Optional, Any
from collections import OrderedDict
from threading import Lock
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """Cached query result with metadata."""
    memories: List[Dict]
    query: str
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    In-memory semantic query cache with TTL.
    
    Caches query -> results mapping for instant repeated queries.
    Automatically invalidates on new memory ingestion.
    """
    
    def __init__(
        self,
        maxsize: int = 500,
        ttl_seconds: int = 3600,  # 1 hour default
        redis_url: Optional[str] = None
    ):
        """
        Initialize the semantic cache.
        
        Args:
            maxsize: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries
            redis_url: Optional Redis URL for distributed caching
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.redis_url = redis_url
        
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()
        self._lock = Lock()
        self._redis = None
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        
        # Initialize Redis if URL provided
        if redis_url:
            self._init_redis(redis_url)
    
    def _init_redis(self, redis_url: str):
        """Initialize Redis connection."""
        try:
            import redis
            self._redis = redis.from_url(redis_url)
            self._redis.ping()
            logger.info(f"✅ Connected to Redis for semantic cache")
        except ImportError:
            logger.warning("⚠️ redis package not installed, using in-memory cache only")
            self._redis = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}, using in-memory cache")
            self._redis = None
    
    def _cache_key(self, user_id: str, query: str) -> str:
        """Generate cache key from user_id and query."""
        # Normalize query
        normalized = query.lower().strip()
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"sem_cache:{user_id}:{query_hash}"
    
    def get(self, user_id: str, query: str) -> Optional[List[Dict]]:
        """
        Get cached results for a query.
        
        Args:
            user_id: User ID
            query: Search query
            
        Returns:
            Cached memories list or None if not found/expired
        """
        key = self._cache_key(user_id, query)
        
        # Try Redis first
        if self._redis:
            try:
                cached = self._redis.get(key)
                if cached:
                    data = json.loads(cached)
                    self._hits += 1
                    logger.debug(f"[SemanticCache] Redis HIT for {key[:20]}...")
                    return data.get("memories", [])
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fall back to in-memory cache
        with self._lock:
            if key in self._cache:
                cached = self._cache[key]
                
                # Check TTL
                if time.time() - cached.timestamp > self.ttl_seconds:
                    del self._cache[key]
                    self._misses += 1
                    logger.debug(f"[SemanticCache] EXPIRED for {key[:20]}...")
                    return None
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                cached.hit_count += 1
                self._hits += 1
                logger.debug(f"[SemanticCache] HIT for {key[:20]}...")
                return cached.memories
            
            self._misses += 1
            logger.debug(f"[SemanticCache] MISS for {key[:20]}...")
            return None
    
    def set(self, user_id: str, query: str, memories: List[Dict]) -> None:
        """
        Cache query results.
        
        Args:
            user_id: User ID
            query: Search query
            memories: List of memory dicts to cache
        """
        key = self._cache_key(user_id, query)
        
        # Store in Redis
        if self._redis:
            try:
                data = {
                    "memories": memories,
                    "query": query,
                    "timestamp": time.time(),
                }
                self._redis.setex(key, self.ttl_seconds, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Store in memory cache
        with self._lock:
            cached = CachedResult(
                memories=memories,
                query=query,
                timestamp=time.time(),
            )
            
            # Update or add
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = cached
            
            # Evict oldest if over capacity
            while len(self._cache) > self.maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
    
    def invalidate_user(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a user.
        
        Call this when a user ingests new memories.
        
        Args:
            user_id: User ID to invalidate
            
        Returns:
            Number of entries invalidated
        """
        prefix = f"sem_cache:{user_id}:"
        count = 0
        
        # Invalidate Redis
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis invalidation failed: {e}")
        
        # Invalidate in-memory
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
        
        self._invalidations += count
        logger.info(f"[SemanticCache] Invalidated {count} entries for user {user_id[:8]}...")
        return count
    
    def invalidate_all(self) -> int:
        """
        Clear entire cache.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        
        # Clear Redis
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match="sem_cache:*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear in-memory
        with self._lock:
            count += len(self._cache)
            self._cache.clear()
        
        self._invalidations += count
        logger.info(f"[SemanticCache] Cleared all {count} entries")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_percent": f"{hit_rate * 100:.1f}%",
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds,
                "invalidations": self._invalidations,
                "redis_enabled": self._redis is not None,
            }
    
    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"SemanticCache(size={stats['size']}/{stats['maxsize']}, hit_rate={stats['hit_rate_percent']})"


# Global singleton (can be configured with Redis URL from settings)
semantic_cache = SemanticCache(maxsize=500, ttl_seconds=3600)
