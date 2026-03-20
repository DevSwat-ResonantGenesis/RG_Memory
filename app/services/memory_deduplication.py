"""
Memory Deduplication Service
=============================

Prevents duplicate memories from being stored by:
1. Content hash comparison (exact duplicates)
2. Semantic similarity detection (near-duplicates)
3. Hash Sphere proximity detection (spatially similar memories)

Author: Resonant Genesis Team
Date: December 29, 2025
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCheckResult:
    """Result of duplicate check."""
    is_duplicate: bool
    duplicate_type: Optional[str] = None  # "exact", "semantic", "spatial"
    existing_memory_id: Optional[str] = None
    similarity_score: float = 0.0
    message: str = ""


class MemoryDeduplicationService:
    """
    Detects and prevents duplicate memories.
    
    Three-level deduplication:
    1. Exact: Content hash match (fastest)
    2. Semantic: Embedding similarity > threshold
    3. Spatial: Hash Sphere proximity in 3D space
    """
    
    # Thresholds for duplicate detection
    SEMANTIC_SIMILARITY_THRESHOLD = 0.95  # Very high similarity = duplicate
    SPATIAL_PROXIMITY_THRESHOLD = 0.05    # Distance in 3D space
    
    def __init__(self):
        self.content_hashes: Dict[str, str] = {}  # hash -> memory_id
        self._stats = {
            "checks": 0,
            "exact_duplicates": 0,
            "semantic_duplicates": 0,
            "spatial_duplicates": 0,
            "unique": 0,
        }
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of normalized content."""
        # Normalize content (lowercase, strip whitespace, remove extra spaces)
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def check_exact_duplicate(
        self,
        content: str,
        existing_hashes: Dict[str, str] = None,
    ) -> DuplicateCheckResult:
        """
        Check for exact content duplicate using hash.
        
        Args:
            content: Memory content to check
            existing_hashes: Dict of {content_hash: memory_id}
            
        Returns:
            DuplicateCheckResult
        """
        content_hash = self.compute_content_hash(content)
        
        # Check in-memory cache first
        if content_hash in self.content_hashes:
            return DuplicateCheckResult(
                is_duplicate=True,
                duplicate_type="exact",
                existing_memory_id=self.content_hashes[content_hash],
                similarity_score=1.0,
                message="Exact duplicate found (content hash match)",
            )
        
        # Check provided hashes
        if existing_hashes and content_hash in existing_hashes:
            return DuplicateCheckResult(
                is_duplicate=True,
                duplicate_type="exact",
                existing_memory_id=existing_hashes[content_hash],
                similarity_score=1.0,
                message="Exact duplicate found in database",
            )
        
        return DuplicateCheckResult(
            is_duplicate=False,
            message="No exact duplicate found",
        )
    
    def check_semantic_duplicate(
        self,
        embedding: List[float],
        existing_embeddings: List[Tuple[str, List[float]]],
        threshold: float = None,
    ) -> DuplicateCheckResult:
        """
        Check for semantic duplicate using embedding similarity.
        
        Args:
            embedding: Embedding vector of new content
            existing_embeddings: List of (memory_id, embedding) tuples
            threshold: Similarity threshold (default: 0.95)
            
        Returns:
            DuplicateCheckResult
        """
        if threshold is None:
            threshold = self.SEMANTIC_SIMILARITY_THRESHOLD
        
        if not embedding or not existing_embeddings:
            return DuplicateCheckResult(
                is_duplicate=False,
                message="No embeddings to compare",
            )
        
        try:
            import numpy as np
            
            new_vec = np.array(embedding)
            new_norm = np.linalg.norm(new_vec)
            
            if new_norm == 0:
                return DuplicateCheckResult(
                    is_duplicate=False,
                    message="Invalid embedding (zero norm)",
                )
            
            max_similarity = 0.0
            most_similar_id = None
            
            for memory_id, existing_emb in existing_embeddings:
                existing_vec = np.array(existing_emb)
                existing_norm = np.linalg.norm(existing_vec)
                
                if existing_norm == 0:
                    continue
                
                # Cosine similarity
                similarity = np.dot(new_vec, existing_vec) / (new_norm * existing_norm)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = memory_id
            
            if max_similarity >= threshold:
                return DuplicateCheckResult(
                    is_duplicate=True,
                    duplicate_type="semantic",
                    existing_memory_id=most_similar_id,
                    similarity_score=float(max_similarity),
                    message=f"Semantic duplicate found (similarity: {max_similarity:.3f})",
                )
            
            return DuplicateCheckResult(
                is_duplicate=False,
                similarity_score=float(max_similarity),
                message=f"No semantic duplicate (max similarity: {max_similarity:.3f})",
            )
            
        except ImportError:
            logger.warning("NumPy not available for semantic deduplication")
            return DuplicateCheckResult(
                is_duplicate=False,
                message="Semantic check skipped (NumPy not available)",
            )
        except Exception as e:
            logger.error(f"Semantic deduplication error: {e}")
            return DuplicateCheckResult(
                is_duplicate=False,
                message=f"Semantic check error: {e}",
            )
    
    def check_spatial_duplicate(
        self,
        xyz: Tuple[float, float, float],
        existing_coords: List[Tuple[str, Tuple[float, float, float]]],
        threshold: float = None,
    ) -> DuplicateCheckResult:
        """
        Check for spatial duplicate using Hash Sphere 3D proximity.
        
        Args:
            xyz: 3D coordinates of new memory
            existing_coords: List of (memory_id, (x, y, z)) tuples
            threshold: Distance threshold (default: 0.05)
            
        Returns:
            DuplicateCheckResult
        """
        if threshold is None:
            threshold = self.SPATIAL_PROXIMITY_THRESHOLD
        
        if not xyz or not existing_coords:
            return DuplicateCheckResult(
                is_duplicate=False,
                message="No coordinates to compare",
            )
        
        try:
            import math
            
            min_distance = float('inf')
            closest_id = None
            
            for memory_id, (ex, ey, ez) in existing_coords:
                # Euclidean distance in 3D space
                distance = math.sqrt(
                    (xyz[0] - ex) ** 2 +
                    (xyz[1] - ey) ** 2 +
                    (xyz[2] - ez) ** 2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_id = memory_id
            
            if min_distance <= threshold:
                return DuplicateCheckResult(
                    is_duplicate=True,
                    duplicate_type="spatial",
                    existing_memory_id=closest_id,
                    similarity_score=1.0 - min_distance,  # Convert distance to similarity
                    message=f"Spatial duplicate found (distance: {min_distance:.4f})",
                )
            
            return DuplicateCheckResult(
                is_duplicate=False,
                similarity_score=1.0 - min_distance if min_distance < 1 else 0,
                message=f"No spatial duplicate (min distance: {min_distance:.4f})",
            )
            
        except Exception as e:
            logger.error(f"Spatial deduplication error: {e}")
            return DuplicateCheckResult(
                is_duplicate=False,
                message=f"Spatial check error: {e}",
            )
    
    def check_duplicate(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        xyz: Optional[Tuple[float, float, float]] = None,
        existing_hashes: Dict[str, str] = None,
        existing_embeddings: List[Tuple[str, List[float]]] = None,
        existing_coords: List[Tuple[str, Tuple[float, float, float]]] = None,
        check_semantic: bool = True,
        check_spatial: bool = True,
    ) -> DuplicateCheckResult:
        """
        Full duplicate check using all three methods.
        
        Checks in order (fastest to slowest):
        1. Exact hash match
        2. Semantic similarity
        3. Spatial proximity
        
        Returns on first duplicate found.
        """
        self._stats["checks"] += 1
        
        # 1. Check exact duplicate (fastest)
        result = self.check_exact_duplicate(content, existing_hashes)
        if result.is_duplicate:
            self._stats["exact_duplicates"] += 1
            logger.info(f"[Dedup] Exact duplicate: {result.existing_memory_id}")
            return result
        
        # 2. Check semantic duplicate
        if check_semantic and embedding and existing_embeddings:
            result = self.check_semantic_duplicate(embedding, existing_embeddings)
            if result.is_duplicate:
                self._stats["semantic_duplicates"] += 1
                logger.info(f"[Dedup] Semantic duplicate: {result.existing_memory_id} (sim={result.similarity_score:.3f})")
                return result
        
        # 3. Check spatial duplicate
        if check_spatial and xyz and existing_coords:
            result = self.check_spatial_duplicate(xyz, existing_coords)
            if result.is_duplicate:
                self._stats["spatial_duplicates"] += 1
                logger.info(f"[Dedup] Spatial duplicate: {result.existing_memory_id}")
                return result
        
        # No duplicate found
        self._stats["unique"] += 1
        return DuplicateCheckResult(
            is_duplicate=False,
            message="Memory is unique",
        )
    
    def register_memory(self, content: str, memory_id: str):
        """Register a memory's content hash for future duplicate checks."""
        content_hash = self.compute_content_hash(content)
        self.content_hashes[content_hash] = memory_id
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        total = self._stats["checks"]
        return {
            "total_checks": total,
            "exact_duplicates": self._stats["exact_duplicates"],
            "semantic_duplicates": self._stats["semantic_duplicates"],
            "spatial_duplicates": self._stats["spatial_duplicates"],
            "unique_memories": self._stats["unique"],
            "duplicate_rate": f"{((self._stats['exact_duplicates'] + self._stats['semantic_duplicates'] + self._stats['spatial_duplicates']) / total * 100):.1f}%" if total > 0 else "0%",
            "cache_size": len(self.content_hashes),
        }


# Global instance
memory_deduplication = MemoryDeduplicationService()


def check_for_duplicate(
    content: str,
    embedding: Optional[List[float]] = None,
    xyz: Optional[Tuple[float, float, float]] = None,
) -> DuplicateCheckResult:
    """Convenience function to check for duplicates."""
    return memory_deduplication.check_duplicate(
        content=content,
        embedding=embedding,
        xyz=xyz,
    )
