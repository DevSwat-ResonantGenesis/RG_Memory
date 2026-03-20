"""
Performance Logger Service
Track and log performance metrics for memory operations.
"""
from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable, Dict, List, Any
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track and log performance metrics for memory operations.
    
    Provides timing decorators, manual logging, and statistics.
    """
    
    def __init__(self, sample_size: int = 100):
        """
        Initialize the performance tracker.
        
        Args:
            sample_size: Number of samples to keep for each metric
        """
        self.sample_size = sample_size
        self._lock = Lock()
        
        # Timing metrics (store last N samples)
        self._timings: Dict[str, deque] = {
            "embedding_generation": deque(maxlen=sample_size),
            "retrieval": deque(maxlen=sample_size),
            "memory_merge": deque(maxlen=sample_size),
            "ingest": deque(maxlen=sample_size),
            "hash_sphere": deque(maxlen=sample_size),
        }
        
        # Counter metrics
        self._counters: Dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_retrievals": 0,
            "total_ingests": 0,
            "errors": 0,
        }
    
    def log_timing(self, operation: str, duration_ms: float) -> None:
        """
        Log a timing measurement.
        
        Args:
            operation: Name of the operation (e.g., "embedding_generation")
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            if operation not in self._timings:
                self._timings[operation] = deque(maxlen=self.sample_size)
            
            self._timings[operation].append(duration_ms)
        
        # Log slow operations
        if duration_ms > 500:
            logger.warning(f"[PERF] SLOW {operation}: {duration_ms:.2f}ms")
        else:
            logger.debug(f"[PERF] {operation}: {duration_ms:.2f}ms")
    
    def increment(self, counter: str, amount: int = 1) -> None:
        """
        Increment a counter.
        
        Args:
            counter: Name of the counter
            amount: Amount to increment by
        """
        with self._lock:
            if counter not in self._counters:
                self._counters[counter] = 0
            self._counters[counter] += amount
    
    def log_cache_hit(self) -> None:
        """Log a cache hit."""
        self.increment("cache_hits")
    
    def log_cache_miss(self) -> None:
        """Log a cache miss."""
        self.increment("cache_misses")
    
    def log_error(self) -> None:
        """Log an error occurrence."""
        self.increment("errors")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all performance statistics.
        
        Returns:
            Dictionary with timing stats and counters
        """
        with self._lock:
            stats = {}
            
            # Calculate timing statistics
            for operation, samples in self._timings.items():
                if samples:
                    sorted_samples = sorted(samples)
                    n = len(sorted_samples)
                    
                    stats[f"{operation}_avg_ms"] = sum(samples) / n
                    stats[f"{operation}_min_ms"] = sorted_samples[0]
                    stats[f"{operation}_max_ms"] = sorted_samples[-1]
                    stats[f"{operation}_p50_ms"] = sorted_samples[n // 2]
                    stats[f"{operation}_p95_ms"] = sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1]
                    stats[f"{operation}_samples"] = n
            
            # Add counters
            stats.update(self._counters)
            
            # Calculate cache hit rate
            total_cache = self._counters.get("cache_hits", 0) + self._counters.get("cache_misses", 0)
            if total_cache > 0:
                stats["cache_hit_rate"] = self._counters.get("cache_hits", 0) / total_cache
                stats["cache_hit_rate_percent"] = f"{stats['cache_hit_rate'] * 100:.1f}%"
            else:
                stats["cache_hit_rate"] = 0.0
                stats["cache_hit_rate_percent"] = "0.0%"
            
            return stats
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with avg, min, max, p50, p95
        """
        with self._lock:
            samples = self._timings.get(operation, deque())
            
            if not samples:
                return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "p50_ms": 0, "p95_ms": 0, "samples": 0}
            
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            
            return {
                "avg_ms": sum(samples) / n,
                "min_ms": sorted_samples[0],
                "max_ms": sorted_samples[-1],
                "p50_ms": sorted_samples[n // 2],
                "p95_ms": sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
                "samples": n,
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for key in self._timings:
                self._timings[key].clear()
            for key in self._counters:
                self._counters[key] = 0
        
        logger.info("[PERF] Performance metrics reset")
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"PerformanceTracker(retrievals={stats.get('total_retrievals', 0)}, cache_hit_rate={stats.get('cache_hit_rate_percent', '0%')})"


# Global singleton instance
perf_tracker = PerformanceTracker(sample_size=100)


def track_time(operation: str):
    """
    Decorator to track function execution time.
    
    Works with both sync and async functions.
    
    Args:
        operation: Name of the operation to track
        
    Example:
        @track_time("embedding_generation")
        async def generate_embedding(text: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                perf_tracker.log_timing(operation, duration_ms)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                perf_tracker.log_timing(operation, duration_ms)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks.
    
    Example:
        with TimingContext("embedding_generation"):
            embedding = await generate_embedding(text)
    """
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        perf_tracker.log_timing(self.operation, duration_ms)
        return False
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        perf_tracker.log_timing(self.operation, duration_ms)
        return False
