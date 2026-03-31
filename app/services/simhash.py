"""
SimHash — Locality-Sensitive Hashing for Semantic Memory
=========================================================

Replaces SHA-256 cryptographic hashing with SimHash for the Hash Sphere.

SHA-256 problem: "I love dogs" and "I love puppies" produce completely
unrelated hashes (Hamming distance ≈ 128 bits). This makes hash-based
resonance comparison pure noise.

SimHash solution: Similar text → similar hashes. The Hamming distance
between SimHash values correlates with semantic dissimilarity.

Two modes:
1. Token-based SimHash: Uses word n-grams (fast, no model needed)
2. Embedding-based SimHash: Uses embedding vector to produce hash (best quality)

Both produce 64-bit integer hashes where:
- Identical text → identical hash
- Similar text → small Hamming distance
- Dissimilar text → large Hamming distance
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Number of bits in the SimHash
SIMHASH_BITS = 64


def _token_hash(token: str) -> int:
    """Hash a single token to a 64-bit integer using MD5 (fast, uniform)."""
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def simhash_text(text: str, ngram_size: int = 3) -> int:
    """Compute SimHash from text using character n-grams.
    
    Algorithm:
    1. Extract n-grams from text
    2. Hash each n-gram to 64-bit integer
    3. For each bit position, sum +1 if bit=1, -1 if bit=0
    4. Final hash: bit=1 if sum > 0, else bit=0
    
    Args:
        text: Input text
        ngram_size: Size of character n-grams (default 3)
    
    Returns:
        64-bit SimHash integer
    """
    normalized = text.lower().strip()
    if not normalized:
        return 0
    
    # Generate character n-grams
    tokens = []
    for i in range(len(normalized) - ngram_size + 1):
        tokens.append(normalized[i:i + ngram_size])
    
    # Also add word-level tokens for broader semantic capture
    words = normalized.split()
    tokens.extend(words)
    # Add word bigrams
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
    
    if not tokens:
        return _token_hash(normalized)
    
    # Compute weighted bit vector
    bit_sums = [0] * SIMHASH_BITS
    
    for token in tokens:
        token_h = _token_hash(token)
        for i in range(SIMHASH_BITS):
            if token_h & (1 << i):
                bit_sums[i] += 1
            else:
                bit_sums[i] -= 1
    
    # Convert to final hash
    result = 0
    for i in range(SIMHASH_BITS):
        if bit_sums[i] > 0:
            result |= (1 << i)
    
    return result


def simhash_embedding(embedding: List[float]) -> int:
    """Compute SimHash from an embedding vector.
    
    Uses random hyperplane hashing (a form of LSH):
    - Generate 64 random hyperplanes (seeded for determinism)
    - For each hyperplane, hash bit = 1 if dot(embedding, hyperplane) > 0
    
    This preserves cosine similarity: similar embeddings → similar hashes.
    
    Args:
        embedding: Embedding vector (any dimension)
    
    Returns:
        64-bit SimHash integer
    """
    if not embedding:
        return 0
    
    emb = np.array(embedding, dtype=np.float64)
    dim = len(emb)
    
    # Deterministic random hyperplanes (seeded)
    rng = np.random.RandomState(seed=42)
    hyperplanes = rng.randn(SIMHASH_BITS, dim)
    
    # Compute hash: bit_i = 1 if dot(embedding, hyperplane_i) > 0
    projections = hyperplanes @ emb
    
    result = 0
    for i in range(SIMHASH_BITS):
        if projections[i] > 0:
            result |= (1 << i)
    
    return result


def simhash_to_hex(simhash: int) -> str:
    """Convert SimHash integer to hex string (16 chars for 64-bit)."""
    return f"{simhash:016x}"


def hex_to_simhash(hex_str: str) -> int:
    """Convert hex string back to SimHash integer."""
    return int(hex_str, 16)


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two SimHash values.
    
    Returns:
        Number of differing bits (0 = identical, 64 = maximally different)
    """
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def simhash_similarity(hash1: int, hash2: int) -> float:
    """Compute similarity from SimHash Hamming distance.
    
    Returns:
        Similarity in [0, 1] range (1 = identical, 0 = maximally different)
    """
    dist = hamming_distance(hash1, hash2)
    return 1.0 - (dist / SIMHASH_BITS)


class SimHasher:
    """SimHash service for the Hash Sphere memory system.
    
    Replaces SHA-256 hashing with locality-sensitive hashing.
    Provides both text-based and embedding-based SimHash generation.
    """
    
    def hash_text(self, text: str, context: Optional[str] = None) -> str:
        """Generate SimHash hex string from text.
        
        If context is provided, it's appended to text for richer hashing.
        
        Args:
            text: Input text
            context: Optional context string
        
        Returns:
            SimHash as 16-character hex string
        """
        combined = text
        if context:
            combined = f"{text} {context}"
        
        sh = simhash_text(combined)
        return simhash_to_hex(sh)
    
    def hash_embedding(self, embedding: List[float]) -> str:
        """Generate SimHash hex string from embedding vector.
        
        This is preferred over hash_text when an embedding is available,
        as it preserves cosine similarity in the hash space.
        
        Args:
            embedding: Embedding vector
        
        Returns:
            SimHash as 16-character hex string
        """
        sh = simhash_embedding(embedding)
        return simhash_to_hex(sh)
    
    def calculate_resonance(self, hash1: str, hash2: str) -> float:
        """Calculate resonance between two SimHash hex strings.
        
        Unlike SHA-256 Hamming (random noise ~50%), SimHash Hamming
        is semantically meaningful: similar content → high resonance.
        
        Args:
            hash1: First SimHash hex string
            hash2: Second SimHash hex string
        
        Returns:
            Resonance score in [0, 1] range
        """
        try:
            sh1 = hex_to_simhash(hash1)
            sh2 = hex_to_simhash(hash2)
            return simhash_similarity(sh1, sh2)
        except (ValueError, TypeError):
            return 0.0


# Global singleton
_simhasher: Optional[SimHasher] = None


def get_simhasher() -> SimHasher:
    """Get or create the global SimHasher instance."""
    global _simhasher
    if _simhasher is None:
        _simhasher = SimHasher()
    return _simhasher
