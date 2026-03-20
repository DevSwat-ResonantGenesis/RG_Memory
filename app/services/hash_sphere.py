"""
Hash Sphere Service for Memory Service
========================================

Provides Hash Sphere functionality for memory operations.
Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/resonance_hashing.py
"""
from __future__ import annotations

import hashlib
import math
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ResonanceHasher:
    """
    Hash Sphere Resonance Hasher
    
    Generates semantic hashes that encode meaning, energy, and spin.
    """
    
    def __init__(self):
        self.anchor_patterns = [
            r'\b(important|critical|key|essential|vital)\b',
            r'\b(always|never|must|should|need)\b',
            r'\b(remember|note|recall|memorize)\b',
            r'\b(goal|objective|target|aim)\b',
            r'\b(problem|issue|challenge|difficulty)\b',
            r'\b(solution|answer|fix|resolve)\b',
        ]
    
    def hash_text(self, text: str, context: Optional[str] = None) -> str:
        """Generate a resonance hash for text."""
        normalized = text.lower().strip()
        
        if context:
            normalized = f"{context}:{normalized}"
        
        # Generate semantic hash
        semantic_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        # Calculate energy and spin
        energy = self._calculate_energy(text)
        spin = self._calculate_spin(text)
        
        # Encode energy and spin into hash
        energy_hex = format(int(energy * 255), '02x')
        spin_hex = format(int(spin * 255), '02x')
        
        return f"{semantic_hash}{energy_hex}{spin_hex}"
    
    def _calculate_energy(self, text: str) -> float:
        """Calculate energy score (0-1) based on text intensity."""
        energy = 0.5
        
        # Exclamation marks increase energy
        energy += min(text.count('!') * 0.1, 0.3)
        
        # Question marks slightly increase energy
        energy += min(text.count('?') * 0.05, 0.15)
        
        # Caps words increase energy
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        energy += min(caps_words * 0.05, 0.2)
        
        # Emotional words
        emotional_words = ['love', 'hate', 'amazing', 'terrible', 'urgent', 'critical']
        for word in emotional_words:
            if word in text.lower():
                energy += 0.05
        
        return min(max(energy, 0.0), 1.0)
    
    def _calculate_spin(self, text: str) -> float:
        """Calculate spin score (0-1) based on text sentiment direction."""
        positive_words = ['good', 'great', 'excellent', 'love', 'happy', 'success', 'win', 'best']
        negative_words = ['bad', 'terrible', 'hate', 'sad', 'fail', 'lose', 'worst', 'problem']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
        
        return (positive_count / total) * 0.5 + 0.25
    
    def extract_anchors(self, text: str) -> List[str]:
        """Extract semantic anchors from text."""
        anchors = []
        text_lower = text.lower()
        
        for pattern in self.anchor_patterns:
            matches = re.findall(pattern, text_lower)
            anchors.extend(matches)
        
        # Also extract capitalized phrases as potential anchors
        caps_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        anchors.extend([p.lower() for p in caps_phrases[:5]])
        
        return list(set(anchors))[:10]
    
    def calculate_resonance(self, hash1: str, hash2: str) -> float:
        """Calculate resonance score between two hashes."""
        if not hash1 or not hash2:
            return 0.0
        
        # Compare semantic portions
        semantic1 = hash1[:16] if len(hash1) >= 16 else hash1
        semantic2 = hash2[:16] if len(hash2) >= 16 else hash2
        
        # Character-level similarity
        matches = sum(1 for a, b in zip(semantic1, semantic2) if a == b)
        char_similarity = matches / max(len(semantic1), len(semantic2))
        
        # Hamming distance based similarity
        try:
            int1 = int(semantic1, 16)
            int2 = int(semantic2, 16)
            xor = int1 ^ int2
            hamming = bin(xor).count('1')
            hamming_similarity = 1 - (hamming / 64)
        except ValueError:
            hamming_similarity = char_similarity
        
        return (char_similarity + hamming_similarity) / 2
    
    def to_xyz(self, hash_value: str) -> Tuple[float, float, float]:
        """Convert hash to 3D coordinates."""
        hash_bytes = hashlib.sha256(hash_value.encode()).digest()
        
        x = int.from_bytes(hash_bytes[:8], 'big') / 0xFFFFFFFFFFFFFFFF
        y = int.from_bytes(hash_bytes[8:16], 'big') / 0xFFFFFFFFFFFFFFFF
        z = int.from_bytes(hash_bytes[16:24], 'big') / 0xFFFFFFFFFFFFFFFF
        
        return (x, y, z)
    
    def to_hyperspherical(self, coords: np.ndarray) -> Dict[str, float]:
        """Convert Cartesian to hyperspherical coordinates."""
        x, y, z = coords[0], coords[1], coords[2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            return {"r": 0.0, "phi": 0.0, "theta": 0.0}
        
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        return {
            "r": float(r),
            "phi": float(phi),
            "theta": float(theta)
        }
    
    def magnetic_pull(self, resonance_score: float) -> float:
        """
        Hash Sphere Magnetic Pull System
        
        Non-linear boost to strong memories.
        """
        magnetic = (resonance_score ** 2) * 1.5
        return min(magnetic, 1.0)


class MemoryAnchorService:
    """Service for managing memory anchors."""
    
    def __init__(self):
        self.hasher = ResonanceHasher()
    
    def create_anchor_hash(self, text: str, context: Optional[str] = None) -> str:
        """Create a hash for a memory anchor."""
        return self.hasher.hash_text(text, context)
    
    def calculate_anchor_resonance(self, anchor1: str, anchor2: str) -> float:
        """Calculate resonance between two anchors."""
        hash1 = self.hasher.hash_text(anchor1)
        hash2 = self.hasher.hash_text(anchor2)
        return self.hasher.calculate_resonance(hash1, hash2)
    
    def rank_by_resonance(
        self,
        query: str,
        anchors: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank anchors by resonance to query."""
        query_hash = self.hasher.hash_text(query)
        
        scored_anchors = []
        for anchor in anchors:
            anchor_hash = anchor.get('hash') or self.hasher.hash_text(anchor.get('text', ''))
            resonance = self.hasher.calculate_resonance(query_hash, anchor_hash)
            
            # Apply magnetic pull
            boosted_resonance = self.hasher.magnetic_pull(resonance)
            
            scored_anchors.append({
                **anchor,
                'resonance_score': resonance,
                'boosted_score': boosted_resonance
            })
        
        scored_anchors.sort(key=lambda x: x['boosted_score'], reverse=True)
        return scored_anchors[:limit]


# Global instances
resonance_hasher = ResonanceHasher()
memory_anchor_service = MemoryAnchorService()
