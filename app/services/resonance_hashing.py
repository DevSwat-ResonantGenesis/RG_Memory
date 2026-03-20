"""
Resonance Hashing Service
Hashes text by meaning, energy, and spin to create resonance signatures
Includes full Hash Sphere coordinate system:
- XYZ: 3D semantic space coordinates
- Hyperspherical: r, phi (latitude), theta (longitude)
- Resonance: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
- Anchor Energy: E_j(s) = exp(-β·||s - A_j||²)
- Spin & Drift: Semantic rotation and decay
- Evidence Aggregation: Layer 7 weighted sum
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class HashSphereCoordinates:
    """Full Hash Sphere coordinate system.
    
    Contains all coordinate representations for a point in the Hash Sphere:
    - Cartesian (xyz): 3D semantic space
    - Hyperspherical (r, phi, theta): Radius, latitude, longitude
    - Resonance: R(h) value from resonance function
    - Energy: Anchor attraction energy
    - Spin: Semantic rotation vector
    """
    # Cartesian coordinates (3D semantic space)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Hyperspherical coordinates
    r: float = 1.0  # Radius (should be 1 for unit sphere)
    phi: float = 0.0  # Latitude in radians (-π/2 to π/2)
    theta: float = 0.0  # Longitude in radians (-π to π)
    
    # Resonance values
    resonance_score: float = 0.0  # R(h) = sin(a·x) + cos(b·y) + tan(c·z)
    normalized_resonance: float = 0.5  # Normalized to 0-1
    
    # Energy values
    energy: float = 0.0  # Anchor attraction energy
    spin_magnitude: float = 0.0  # Magnitude of spin vector
    
    # Spin vector (semantic rotation)
    spin_x: float = 0.0
    spin_y: float = 0.0
    spin_z: float = 0.0
    
    # Metadata
    universe_id: str = ""
    hash: str = ""
    meaning_hash: str = ""
    energy_hash: str = ""
    spin_hash: str = ""
    
    # Semantic components
    meaning_score: float = 0.0  # Semantic meaning strength
    intensity_score: float = 0.0  # Emotional intensity
    sentiment_score: float = 0.5  # Positive/negative sentiment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "xyz": [self.x, self.y, self.z],
            "hyperspherical": {
                "r": self.r,
                "phi": self.phi,
                "theta": self.theta
            },
            "resonance": {
                "score": self.resonance_score,
                "normalized": self.normalized_resonance
            },
            "energy": self.energy,
            "spin": {
                "vector": [self.spin_x, self.spin_y, self.spin_z],
                "magnitude": self.spin_magnitude
            },
            "hashes": {
                "universe_id": self.universe_id,
                "hash": self.hash,
                "meaning": self.meaning_hash,
                "energy": self.energy_hash,
                "spin": self.spin_hash
            },
            "semantic": {
                "meaning": self.meaning_score,
                "intensity": self.intensity_score,
                "sentiment": self.sentiment_score
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HashSphereCoordinates":
        """Create from dictionary."""
        coords = cls()
        if "xyz" in data:
            coords.x, coords.y, coords.z = data["xyz"]
        if "hyperspherical" in data:
            h = data["hyperspherical"]
            coords.r = h.get("r", 1.0)
            coords.phi = h.get("phi", 0.0)
            coords.theta = h.get("theta", 0.0)
        if "resonance" in data:
            r = data["resonance"]
            coords.resonance_score = r.get("score", 0.0)
            coords.normalized_resonance = r.get("normalized", 0.5)
        if "energy" in data:
            coords.energy = data["energy"]
        if "spin" in data:
            s = data["spin"]
            if "vector" in s:
                coords.spin_x, coords.spin_y, coords.spin_z = s["vector"]
            coords.spin_magnitude = s.get("magnitude", 0.0)
        if "hashes" in data:
            h = data["hashes"]
            coords.universe_id = h.get("universe_id", "")
            coords.hash = h.get("hash", "")
            coords.meaning_hash = h.get("meaning", "")
            coords.energy_hash = h.get("energy", "")
            coords.spin_hash = h.get("spin", "")
        if "semantic" in data:
            s = data["semantic"]
            coords.meaning_score = s.get("meaning", 0.0)
            coords.intensity_score = s.get("intensity", 0.0)
            coords.sentiment_score = s.get("sentiment", 0.5)
        return coords


class ResonanceHasher:
    """Hash text by meaning, energy, and spin.
    
    Full Hash Sphere implementation with:
    - 9-Layer Architecture
    - Hyperspherical coordinates
    - Anchor energy calculation
    - Spin and drift functions
    - Evidence aggregation (Layer 7)
    """
    
    # Cache for PCA model (reused across calls)
    _pca_model: Optional[PCA] = None
    
    # Resonance function constants (from foundational architecture)
    RESONANCE_A = np.pi / 4  # sin coefficient
    RESONANCE_B = np.e / 3    # cos coefficient
    RESONANCE_C = 1.618 / 2   # tan coefficient (golden ratio / 2)
    
    # Golden ratio for various calculations
    PHI = 1.618033988749895
    
    # Fusion weight (α) for embedding fusion
    FUSION_ALPHA = 0.9  # 90% meaning, 10% identity
    
    # Drift coefficient (γ) for anchor attraction
    DRIFT_GAMMA = 0.1
    
    # Sharpness parameter (β) for anchor energy
    ANCHOR_BETA = 1.0
    
    # Output correction weight (λ)
    OUTPUT_LAMBDA = 0.8  # 80% LLM, 20% evidence
    
    @staticmethod
    def hash_to_universe_id(text: str) -> str:
        """
        Convert text to Universe ID (256-bit hash).
        
        Mathematical definition: u = H(x)
        Where H is SHA-256 cryptographic hash.
        
        Args:
            text: Input text
        
        Returns:
            Universe ID as hex string (64 characters)
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    @staticmethod
    def universe_id_to_vector(universe_id: str) -> np.ndarray:
        """
        Convert Universe ID to normalized vector.
        
        Mathematical definition: v_h = int(u) / ||int(u)||
        
        Args:
            universe_id: 256-bit hash as hex string
        
        Returns:
            Normalized vector (numpy array)
        """
        # Convert hex to integer
        int_val = int(universe_id, 16)
        
        # Convert to bytes (32 bytes for 256 bits)
        bytes_val = int_val.to_bytes(32, 'big')
        
        # Convert to numpy array (uint8)
        vec = np.frombuffer(bytes_val, dtype=np.uint8).astype(np.float64)
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    @staticmethod
    def hash_text(text: str, context: Optional[str] = None) -> str:
        """
        Create a resonance hash from text.
        
        FIXED: Now uses full content hash to prevent collisions.
        The hash encodes:
        - Meaning: Full semantic content (20 chars)
        - Timestamp: Nanosecond precision for uniqueness
        - Context: Optional context hash
        """
        # Normalize text
        normalized = text.lower().strip()
        
        # Extract meaning (semantic hash) - use more characters to prevent collisions
        meaning_hash = hashlib.sha256(normalized.encode()).hexdigest()[:20]
        
        # Add timestamp for uniqueness (nanosecond precision)
        timestamp_hash = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
        
        # Combine with context if provided
        if context:
            context_hash = hashlib.sha256(context.lower().encode()).hexdigest()[:8]
            combined = f"{meaning_hash}-{timestamp_hash}-{context_hash}"
        else:
            combined = f"{meaning_hash}-{timestamp_hash}"
        
        # Final hash
        final_hash = hashlib.sha256(combined.encode()).hexdigest()
        return final_hash
    
    @staticmethod
    def hash_text_deterministic(text: str, context: Optional[str] = None) -> str:
        """
        Create a DETERMINISTIC resonance hash from text.
        
        Same content always produces same hash (no timestamp).
        Use for deduplication and anchor matching.
        """
        # Normalize text
        normalized = text.lower().strip()
        
        # Full content hash for uniqueness
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        # Combine with context if provided
        if context:
            context_hash = hashlib.sha256(context.lower().encode()).hexdigest()[:16]
            combined = f"{content_hash}-{context_hash}"
            final_hash = hashlib.sha256(combined.encode()).hexdigest()
        else:
            final_hash = content_hash
        
        return final_hash
    
    @staticmethod
    def compute_full_coordinates(
        text: str,
        embedding: Optional[List[float]] = None,
        context: Optional[str] = None,
        anchors: Optional[List[Tuple[float, float, float]]] = None
    ) -> HashSphereCoordinates:
        """
        Compute FULL Hash Sphere coordinates for text.
        
        This is the main entry point for the 9-Layer Architecture:
        - Layer 1: Input Processing (normalization)
        - Layer 2: Hash Generation (meaning + energy + spin)
        - Layer 3: Universe ID (SHA-256)
        - Layer 4: Embedding (if provided)
        - Layer 5: Coordinate Calculation (PCA or hash-based)
        - Layer 6: Resonance Scoring
        - Layer 7: Evidence Aggregation (if anchors provided)
        - Layer 8: Multi-LLM Routing (handled externally)
        - Layer 9: Output Correction (handled externally)
        
        Args:
            text: Input text
            embedding: Optional embedding vector (1536-dim)
            context: Optional context string
            anchors: Optional list of anchor coordinates for energy calculation
        
        Returns:
            HashSphereCoordinates with all coordinate representations
        """
        coords = HashSphereCoordinates()
        
        # Layer 1: Input Processing
        normalized = text.lower().strip()
        
        # Layer 2: Hash Generation
        coords.meaning_hash = hashlib.sha256(normalized.encode()).hexdigest()[:20]
        coords.intensity_score = ResonanceHasher._calculate_energy(normalized)
        coords.sentiment_score = ResonanceHasher._calculate_spin(normalized)
        coords.energy_hash = hashlib.sha256(str(coords.intensity_score).encode()).hexdigest()[:8]
        coords.spin_hash = hashlib.sha256(str(coords.sentiment_score).encode()).hexdigest()[:8]
        
        # Generate unique hash (with timestamp for uniqueness)
        coords.hash = ResonanceHasher.hash_text(text, context)
        
        # Layer 3: Universe ID
        coords.universe_id = ResonanceHasher.hash_to_universe_id(text)
        
        # Layer 4 & 5: Coordinate Calculation
        # HYBRID: Use semantic_encoder for XYZ (better clustering)
        try:
            from .semantic_encoder import get_semantic_encoder
            semantic_enc = get_semantic_encoder()
            x, y, z = semantic_enc.calculate_xyz(text)
        except ImportError:
            # Fallback to original hash-based coordinates
            if embedding and len(embedding) >= 3:
                x, y, z = ResonanceHasher.calculate_xyz_coordinates(embedding)
            else:
                x, y, z = ResonanceHasher.hash_to_coords(coords.universe_id)
        
        coords.x, coords.y, coords.z = x, y, z
        
        # Calculate hyperspherical coordinates
        hyperspherical = ResonanceHasher.to_hyperspherical(np.array([x, y, z]))
        coords.r = hyperspherical['r']
        coords.phi = hyperspherical['phi']
        coords.theta = hyperspherical['theta']
        
        # Layer 6: Resonance Scoring
        coords.resonance_score = ResonanceHasher.calculate_resonance_function((x, y, z))
        # Normalize resonance to 0-1 (resonance can be negative due to tan)
        coords.normalized_resonance = ResonanceHasher._normalize_resonance(coords.resonance_score)
        
        # Calculate spin vector from semantic analysis
        spin_vec = ResonanceHasher._calculate_spin_vector(normalized)
        coords.spin_x, coords.spin_y, coords.spin_z = spin_vec
        coords.spin_magnitude = np.linalg.norm(spin_vec)
        
        # Layer 7: Evidence Aggregation (if anchors provided)
        if anchors:
            # Calculate anchor energy (attraction to nearest anchor)
            best_idx, best_energy = ResonanceHasher.find_best_anchor(
                np.array([x, y, z]),
                [np.array(a) for a in anchors]
            )
            coords.energy = best_energy
        else:
            # Default energy based on resonance
            coords.energy = coords.normalized_resonance
        
        # Meaning score based on content length and complexity
        coords.meaning_score = ResonanceHasher._calculate_meaning_score(normalized)
        
        return coords
    
    @staticmethod
    def _normalize_resonance(resonance: float) -> float:
        """
        Normalize resonance score to 0-1 range.
        
        Resonance function R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        can produce values from -∞ to +∞ due to tan.
        
        We use sigmoid to normalize: 1 / (1 + e^(-r))
        """
        # Clip extreme values
        clipped = np.clip(resonance, -10, 10)
        # Sigmoid normalization
        normalized = 1 / (1 + np.exp(-clipped))
        return float(normalized)
    
    @staticmethod
    def _calculate_spin_vector(text: str) -> Tuple[float, float, float]:
        """
        Calculate 3D spin vector from text semantic analysis.
        
        Spin represents the semantic "direction" of the text:
        - X: Topic/domain direction
        - Y: Emotional valence direction
        - Z: Complexity/abstraction direction
        """
        # X: Topic direction based on word categories
        technical_words = ['code', 'function', 'api', 'data', 'system', 'algorithm']
        creative_words = ['idea', 'story', 'design', 'art', 'imagine', 'create']
        tech_count = sum(1 for w in technical_words if w in text.lower())
        creative_count = sum(1 for w in creative_words if w in text.lower())
        x_spin = (tech_count - creative_count) * 0.1
        
        # Y: Emotional valence
        y_spin = (ResonanceHasher._calculate_spin(text) - 0.5) * 0.2
        
        # Z: Complexity based on word length and sentence structure
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        z_spin = (avg_word_len - 5) * 0.05  # Normalize around average word length of 5
        
        # Normalize to unit vector if magnitude > 1
        vec = np.array([x_spin, y_spin, z_spin])
        mag = np.linalg.norm(vec)
        if mag > 1:
            vec = vec / mag
        
        return tuple(vec.tolist())
    
    @staticmethod
    def _calculate_meaning_score(text: str) -> float:
        """
        Calculate meaning score based on content richness.
        
        Higher scores for:
        - Longer content
        - More unique words
        - More complex sentences
        """
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Vocabulary richness
        if word_count > 0:
            richness = unique_words / word_count
        else:
            richness = 0
        
        # Length factor (logarithmic scaling)
        length_factor = min(1.0, np.log1p(word_count) / 5)
        
        # Combined score
        meaning_score = (richness * 0.5 + length_factor * 0.5)
        return float(meaning_score)
    
    @staticmethod
    def _calculate_energy(text: str) -> float:
        """Calculate energy level (0-1) based on emotional/intensity indicators"""
        # Simple heuristic: count intensity words, punctuation, caps
        intensity_words = ['very', 'extremely', 'highly', 'critical', 'urgent', 'important', 'essential']
        intensity_count = sum(1 for word in intensity_words if word in text.lower())
        
        # Punctuation intensity
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Caps intensity
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Normalize to 0-1
        energy = min(1.0, (intensity_count * 0.1 + exclamation_count * 0.05 + 
                          question_count * 0.03 + caps_ratio * 0.5))
        return round(energy, 3)
    
    @staticmethod
    def _calculate_spin(text: str) -> float:
        """Calculate spin (direction/intent) - positive/negative/neutral"""
        # Simple sentiment-based spin
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'fail', 'error', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        # Spin: -1 (negative) to +1 (positive), normalized to 0-1
        if positive_count + negative_count == 0:
            spin = 0.5  # Neutral
        else:
            spin_raw = (positive_count - negative_count) / (positive_count + negative_count)
            spin = (spin_raw + 1) / 2  # Normalize to 0-1
        
        return round(spin, 3)
    
    @staticmethod
    def calculate_resonance(hash1: str, hash2: str) -> float:
        """Calculate resonance score between two hashes (0-1)"""
        # Simple hamming distance on hash strings
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        similarity = matches / len(hash1)
        return round(similarity, 3)
    
    @staticmethod
    def extract_anchors(text: str, max_anchors: int = 5) -> List[str]:
        """Extract key phrases/anchors from text"""
        # Simple extraction: key phrases, important words
        # In production, use NLP models for better extraction
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        important_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Extract phrases (2-3 word combinations)
        anchors = []
        for i in range(len(important_words) - 1):
            phrase = f"{important_words[i]} {important_words[i+1]}"
            if len(phrase) > 5:  # Minimum phrase length
                anchors.append(phrase)
        
        # Also add single important words
        anchors.extend([w for w in important_words if len(w) > 5][:max_anchors])
        
        # Remove duplicates and limit
        unique_anchors = list(dict.fromkeys(anchors))[:max_anchors]
        return unique_anchors
    
    @staticmethod
    def calculate_xyz_coordinates(embedding: List[float]) -> Tuple[float, float, float]:
        """
        Convert embedding vector to 3D XYZ coordinates in semantic space.
        
        Uses PCA (Principal Component Analysis) to reduce embedding dimensions to 3D.
        This creates a 3D semantic space where similar meanings are close together.
        
        Args:
            embedding: High-dimensional embedding vector (e.g., 384, 768, 1536 dimensions)
        
        Returns:
            Tuple of (x, y, z) coordinates in 3D semantic space
        """
        if not embedding or len(embedding) < 3:
            # Fallback: use hash-based coordinates if no embedding
            hash_val = hashlib.sha256(str(embedding).encode()).hexdigest()
            x = int(hash_val[:8], 16) / 0xFFFFFFFF  # Normalize to 0-1
            y = int(hash_val[8:16], 16) / 0xFFFFFFFF
            z = int(hash_val[16:24], 16) / 0xFFFFFFFF
            return (x, y, z)
        
        # Convert to numpy array
        embedding_array = np.array(embedding).reshape(1, -1)
        
        # If embedding is already 3D or less, use directly (normalized)
        if len(embedding) <= 3:
            coords = embedding_array[0]
            # Normalize to 0-1 range
            coords = (coords - coords.min()) / (coords.max() - coords.min() + 1e-10)
            # Pad or truncate to 3D
            if len(coords) < 3:
                coords = np.pad(coords, (0, 3 - len(coords)), mode='constant')
            return tuple(coords[:3].tolist())
        
        # Use PCA to reduce to 3D
        # Initialize PCA model if not exists
        if ResonanceHasher._pca_model is None:
            ResonanceHasher._pca_model = PCA(n_components=3)
            # Fit on a dummy embedding to initialize (will be refit on actual data)
            dummy = np.random.rand(1, len(embedding))
            ResonanceHasher._pca_model.fit(dummy)
        
        # Transform embedding to 3D
        coords_3d = ResonanceHasher._pca_model.transform(embedding_array)[0]
        
        # Normalize to 0-1 range for consistent semantic space
        # Use min-max normalization
        coords_3d = (coords_3d - coords_3d.min()) / (coords_3d.max() - coords_3d.min() + 1e-10)
        
        return tuple(coords_3d.tolist())
    
    @staticmethod
    def calculate_proximity(xyz1: Tuple[float, float, float], xyz2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points in 3D semantic space.
        
        Distance formula: √[(x1-x2)² + (y1-y2)² + (z1-z2)²]
        
        Args:
            xyz1: First point (x, y, z)
            xyz2: Second point (x, y, z)
        
        Returns:
            Distance (0 = same point, higher = more distant)
        """
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        return float(distance)
    
    @staticmethod
    def calculate_proximity_score(xyz1: Tuple[float, float, float], xyz2: Tuple[float, float, float]) -> float:
        """
        Calculate proximity score (0-1) where 1 = same point, 0 = very distant.
        
        Converts distance to similarity score.
        
        Args:
            xyz1: First point (x, y, z)
            xyz2: Second point (x, y, z)
        
        Returns:
            Proximity score (0-1, higher = more similar)
        """
        distance = ResonanceHasher.calculate_proximity(xyz1, xyz2)
        # Convert distance to similarity score (inverse relationship)
        # Using exponential decay: score = e^(-distance)
        # This gives 1.0 for distance 0, ~0.37 for distance 1, ~0.14 for distance 2
        score = np.exp(-distance)
        return float(score)
    
    @staticmethod
    def apply_spin(
        point: np.ndarray,
        spin_vector: Optional[np.ndarray] = None,
        spin_angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply spin (internal semantic rotation) to a point.
        
        Mathematical definition: s_{t+1} = R(ω) · s_t
        
        Where:
        - R(ω) = rotation matrix
        - ω = spin vector from semantic volatility
        - s_t = current point position
        
        Args:
            point: Current point in 3D space (x, y, z)
            spin_vector: Optional spin vector (if None, uses random small rotation)
            spin_angle: Optional rotation angle in radians (if None, uses spin_vector magnitude)
        
        Returns:
            Rotated point
        """
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        
        # If no spin vector provided, use small random rotation
        if spin_vector is None:
            # Small random rotation (semantic drift)
            spin_vector = np.random.normal(0, 0.01, size=3)
        
        if isinstance(spin_vector, (list, tuple)):
            spin_vector = np.array(spin_vector)
        
        # Calculate rotation angle
        if spin_angle is None:
            angle = np.linalg.norm(spin_vector)
            if angle < 1e-10:
                return point  # No rotation
            axis = spin_vector / angle
        else:
            angle = spin_angle
            if np.linalg.norm(spin_vector) > 1e-10:
                axis = spin_vector / np.linalg.norm(spin_vector)
            else:
                axis = np.array([0, 0, 1])  # Default axis
        
        # Rodrigues' rotation formula for 3D rotation
        # R = I + sin(θ)K + (1 - cos(θ))K²
        # where K is the cross-product matrix of axis
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Cross-product matrix K
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rotation matrix
        I = np.eye(3)
        R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # Apply rotation
        rotated_point = np.dot(R, point)
        
        # Normalize to keep on unit sphere
        norm = np.linalg.norm(rotated_point)
        if norm > 0:
            rotated_point = rotated_point / norm
        
        return rotated_point
    
    @staticmethod
    def apply_drift(
        point: np.ndarray,
        anchor: np.ndarray,
        gamma: float = 0.1
    ) -> np.ndarray:
        """
        Apply drift (decay toward anchor).
        
        Mathematical definition: s_{t+1} = s_t + γ(A_{j*} - s_t)
        
        Where:
        - s_t = current point position
        - A_{j*} = anchor position
        - γ ∈ [0,1] = drift coefficient (decay rate)
        
        Args:
            point: Current point in 3D space (x, y, z)
            anchor: Anchor point to drift toward
            gamma: Drift coefficient (0-1, higher = faster drift)
        
        Returns:
            Drifted point (closer to anchor)
        """
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        if isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        
        # Clamp gamma to [0, 1]
        gamma = max(0.0, min(1.0, gamma))
        
        # Apply drift: s_{t+1} = s_t + γ(A - s_t)
        drifted_point = point + gamma * (anchor - point)
        
        # Normalize to keep on unit sphere
        norm = np.linalg.norm(drifted_point)
        if norm > 0:
            drifted_point = drifted_point / norm
        
        return drifted_point
    
    @staticmethod
    def calculate_stability(
        point: np.ndarray,
        previous_points: List[np.ndarray],
        threshold: float = 0.1
    ) -> float:
        """
        Calculate stability metric for a point.
        
        Stability measures how much a point has moved over time.
        Lower values = more stable (less movement)
        Higher values = less stable (more movement)
        
        Args:
            point: Current point position
            previous_points: List of previous positions
            threshold: Distance threshold for stability
        
        Returns:
            Stability score (0-1, lower = more stable)
        """
        if not previous_points:
            return 0.0  # No history = stable
        
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        
        # Calculate average distance from previous points
        distances = []
        for prev_point in previous_points:
            if isinstance(prev_point, (list, tuple)):
                prev_point = np.array(prev_point)
            distance = np.linalg.norm(point - prev_point)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 (stability score)
        # Higher distance = less stable
        stability_score = min(1.0, avg_distance / threshold)
        
        return float(stability_score)
    
    @staticmethod
    def to_hyperspherical(coords: np.ndarray) -> Dict[str, float]:
        """
        Convert 3D coordinates to hyperspherical coordinates (latitude/longitude).
        
        Mathematical definition:
        For s = (x, y, z):
        φ = arcsin(z)  # Latitude
        θ = arctan2(y, x)  # Longitude
        r = ||s||  # Radius (should be 1 for unit sphere)
        
        Args:
            coords: 3D coordinates (x, y, z)
        
        Returns:
            Dictionary with 'r' (radius), 'phi' (latitude), 'theta' (longitude)
        """
        if isinstance(coords, (list, tuple)):
            coords = np.array(coords)
        
        x, y, z = coords[0], coords[1], coords[2]
        
        # Radius
        r = np.linalg.norm(coords)
        
        # Latitude (phi) - angle from equator
        phi = np.arcsin(np.clip(z / r if r > 0 else 0, -1, 1))
        
        # Longitude (theta) - angle around equator
        theta = np.arctan2(y, x)
        
        return {
            'r': float(r),
            'phi': float(phi),  # Latitude in radians
            'theta': float(theta),  # Longitude in radians
        }
    
    @staticmethod
    def from_hyperspherical(r: float, phi: float, theta: float) -> np.ndarray:
        """
        Convert hyperspherical coordinates back to 3D Cartesian coordinates.
        
        Mathematical definition:
        x = r * cos(φ) * cos(θ)
        y = r * cos(φ) * sin(θ)
        z = r * sin(φ)
        
        Args:
            r: Radius
            phi: Latitude (in radians)
            theta: Longitude (in radians)
        
        Returns:
            3D coordinates (x, y, z)
        """
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        return np.array([x, y, z])
    
    @staticmethod
    def fuse_hash_and_embedding(
        hash_vector: np.ndarray,
        embedding: np.ndarray,
        alpha: float = 0.9
    ) -> np.ndarray:
        """
        Fuse hash identity with semantic meaning.
        
        Mathematical definition: s₀ = α·ê + (1-α)·v_h
        Then: s = s₀ / ||s₀|| (project onto unit sphere)
        
        Where:
        - α ∈ [0,1] controls meaning vs identity (usually 0.85-0.95)
        - ê = normalized embedding (semantic meaning)
        - v_h = normalized hash vector (deterministic identity)
        
        Args:
            hash_vector: Normalized hash vector v_h
            embedding: Embedding vector (will be normalized)
            alpha: Fusion weight (default 0.9 = 90% meaning, 10% identity)
        
        Returns:
            Fused sphere point s (normalized to unit sphere)
        """
        # Normalize embedding
        embed_norm = np.array(embedding, dtype=np.float64)
        embed_norm = embed_norm / np.linalg.norm(embed_norm)
        
        # Ensure hash_vector is normalized
        hash_norm = hash_vector / np.linalg.norm(hash_vector) if np.linalg.norm(hash_vector) > 0 else hash_vector
        
        # Fusion: s₀ = α·ê + (1-α)·v_h
        s0 = alpha * embed_norm + (1 - alpha) * hash_norm
        
        # Project onto unit sphere: s = s₀ / ||s₀||
        s = s0 / np.linalg.norm(s0)
        
        return s
    
    @staticmethod
    def calculate_resonance_function(xyz: Tuple[float, float, float]) -> float:
        """
        Calculate resonance using the foundational architecture formula.
        
        Mathematical definition: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        
        Where:
        - a = π/4 (resonance coefficient for x)
        - b = e/3 (resonance coefficient for y)
        - c = φ/2 (resonance coefficient for z, where φ = golden ratio)
        
        Args:
            xyz: Tuple of (x, y, z) coordinates
        
        Returns:
            Resonance value (can be negative or positive)
        """
        x, y, z = xyz
        
        # Resonance function: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        resonance = (
            np.sin(ResonanceHasher.RESONANCE_A * x) +
            np.cos(ResonanceHasher.RESONANCE_B * y) +
            np.tan(ResonanceHasher.RESONANCE_C * z)
        )
        
        return float(resonance)
    
    @staticmethod
    def calculate_anchor_energy(
        point: np.ndarray,
        anchor: np.ndarray,
        beta: float = 1.0
    ) -> float:
        """
        Calculate anchor attraction energy.
        
        Mathematical definition: E_j(s) = exp(-β·||s - A_j||²)
        
        Where:
        - β controls "sharpness" of resonance
        - Higher E_j means stronger alignment with anchor A_j
        
        Args:
            point: Sphere point s (normalized vector)
            anchor: Anchor position A_j (normalized vector)
            beta: Sharpness parameter (default 1.0)
        
        Returns:
            Energy value (0-1, higher = stronger alignment)
        """
        # Ensure both are normalized
        point_norm = point / np.linalg.norm(point) if np.linalg.norm(point) > 0 else point
        anchor_norm = anchor / np.linalg.norm(anchor) if np.linalg.norm(anchor) > 0 else anchor
        
        # Calculate squared distance: ||s - A_j||²
        distance_sq = np.sum((point_norm - anchor_norm) ** 2)
        
        # Energy: E_j(s) = exp(-β·||s - A_j||²)
        energy = np.exp(-beta * distance_sq)
        
        return float(energy)
    
    @staticmethod
    def find_best_anchor(
        point: np.ndarray,
        anchors: List[np.ndarray],
        beta: float = 1.0
    ) -> Tuple[int, float]:
        """
        Find anchor with highest energy.
        
        Mathematical definition: j* = argmax_j E_j(s)
        
        Args:
            point: Sphere point s
            anchors: List of anchor positions A_j
            beta: Sharpness parameter
        
        Returns:
            Tuple of (best_anchor_index, best_energy_score)
        """
        if not anchors:
            return (-1, 0.0)
        
        energies = [
            ResonanceHasher.calculate_anchor_energy(point, anchor, beta)
            for anchor in anchors
        ]
        
        best_idx = int(np.argmax(energies))
        best_energy = energies[best_idx]
        
        return (best_idx, best_energy)
    
    @staticmethod
    def hash_to_coords(hash_str: str) -> Tuple[float, float, float]:
        """
        Convert a hash string to 3D XYZ coordinates.
        
        This is a deterministic mapping from hash to coordinates.
        Used for backward compatibility with existing code.
        
        Args:
            hash_str: Hash string (hex)
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # Use first 24 characters of hash for coordinates
        # Each coordinate gets 8 hex characters (32 bits)
        if len(hash_str) < 24:
            hash_str = hash_str.ljust(24, '0')
        
        x = int(hash_str[:8], 16) / 0xFFFFFFFF  # Normalize to 0-1
        y = int(hash_str[8:16], 16) / 0xFFFFFFFF
        z = int(hash_str[16:24], 16) / 0xFFFFFFFF
        
        return (x, y, z)
    
    @staticmethod
    def aggregate_evidence(
        memories: List[Dict],
        weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Layer 7: Evidence Aggregation.
        
        Mathematical definition: E* = Σ_{i∈R} w_i · s_i
        Then: Ê* = E* / ||E*|| (normalized)
        
        Combines multiple memory positions into a single evidence vector.
        
        Args:
            memories: List of memory dicts with 'xyz' or 'x','y','z' fields
            weights: Optional weights for each memory (default: equal weights)
        
        Returns:
            Tuple of (normalized_evidence_vector, total_weight)
        """
        if not memories:
            return (np.array([0.5, 0.5, 0.5]), 0.0)
        
        # Extract coordinates from memories
        points = []
        for mem in memories:
            if 'xyz' in mem and isinstance(mem['xyz'], (list, tuple)):
                points.append(np.array(mem['xyz']))
            elif all(k in mem for k in ['x', 'y', 'z']):
                points.append(np.array([mem['x'], mem['y'], mem['z']]))
            elif all(k in mem for k in ['xyz_x', 'xyz_y', 'xyz_z']):
                points.append(np.array([mem['xyz_x'], mem['xyz_y'], mem['xyz_z']]))
        
        if not points:
            return (np.array([0.5, 0.5, 0.5]), 0.0)
        
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(points)] * len(points)
        else:
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(points)] * len(points)
        
        # Weighted sum: E* = Σ w_i · s_i
        evidence = np.zeros(3)
        for point, weight in zip(points, weights):
            evidence += weight * point
        
        # Normalize: Ê* = E* / ||E*||
        norm = np.linalg.norm(evidence)
        if norm > 0:
            normalized = evidence / norm
        else:
            normalized = evidence
        
        return (normalized, sum(weights))
    
    @staticmethod
    def correct_output(
        llm_output_vector: np.ndarray,
        evidence_vector: np.ndarray,
        lambda_weight: float = 0.8
    ) -> np.ndarray:
        """
        Layer 9: Output Correction.
        
        Mathematical definition: o_corrected = λ·o_k* + (1-λ)·Ê*
        
        Blends LLM output with evidence to reduce hallucination.
        
        Args:
            llm_output_vector: LLM's output position in semantic space
            evidence_vector: Aggregated evidence vector from Layer 7
            lambda_weight: Weight for LLM output (0.7-0.9 typical)
        
        Returns:
            Corrected output vector
        """
        # Ensure both are numpy arrays
        if isinstance(llm_output_vector, (list, tuple)):
            llm_output_vector = np.array(llm_output_vector)
        if isinstance(evidence_vector, (list, tuple)):
            evidence_vector = np.array(evidence_vector)
        
        # Blend: o_corrected = λ·o_k* + (1-λ)·Ê*
        corrected = lambda_weight * llm_output_vector + (1 - lambda_weight) * evidence_vector
        
        # Normalize to unit sphere
        norm = np.linalg.norm(corrected)
        if norm > 0:
            corrected = corrected / norm
        
        return corrected
    
    @staticmethod
    def magnetic_pull(resonance_score: float) -> float:
        """
        Hash Sphere Magnetic Pull System (HS-MPS).
        
        Non-linear boost for strong memories:
        - Low resonance (0.3) → 0.135 (weaker)
        - Medium resonance (0.6) → 0.54 (moderate)
        - High resonance (0.9) → 1.0 (strong, capped)
        
        Args:
            resonance_score: Original resonance score (0-1)
        
        Returns:
            Boosted resonance score (0-1)
        """
        magnetic = (resonance_score ** 2) * 1.5
        return min(magnetic, 1.0)
    
    @staticmethod
    def compute_cluster_centroid(
        points: List[Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Compute the centroid of a cluster of points.
        
        Args:
            points: List of (x, y, z) coordinates
        
        Returns:
            Centroid (x, y, z)
        """
        if not points:
            return (0.5, 0.5, 0.5)
        
        arr = np.array(points)
        centroid = np.mean(arr, axis=0)
        return tuple(centroid.tolist())
    
    @staticmethod
    def assign_to_cluster(
        point: Tuple[float, float, float],
        cluster_centroids: List[Tuple[float, float, float]],
        cluster_names: Optional[List[str]] = None
    ) -> Tuple[int, str, float]:
        """
        Assign a point to the nearest cluster.
        
        Args:
            point: (x, y, z) coordinates
            cluster_centroids: List of cluster centroid coordinates
            cluster_names: Optional list of cluster names
        
        Returns:
            Tuple of (cluster_index, cluster_name, distance)
        """
        if not cluster_centroids:
            return (0, "default", 0.0)
        
        # Default cluster names
        if cluster_names is None:
            cluster_names = [f"cluster_{i}" for i in range(len(cluster_centroids))]
        
        # Find nearest cluster
        point_arr = np.array(point)
        min_dist = float('inf')
        best_idx = 0
        
        for i, centroid in enumerate(cluster_centroids):
            centroid_arr = np.array(centroid)
            dist = np.linalg.norm(point_arr - centroid_arr)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        return (best_idx, cluster_names[best_idx], float(min_dist))
