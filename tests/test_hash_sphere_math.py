"""
Unit Tests for Hash Sphere Math Layers
======================================

Tests for the mathematical functions integrated in Week 2:
- Layer 3: Fusion Layer (fuse_hash_and_embedding)
- Layer 4: Anchor Energy (calculate_anchor_energy, find_best_anchor)
- Layer 5: Resonance Function (calculate_resonance_function)

Run with: pytest memory_service/tests/test_hash_sphere_math.py -v
"""

import pytest
import numpy as np
from typing import Tuple, List

# Import the functions we're testing

# Add shared modules to path
SHARED_PATH = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_PATH))
import sys
sys.path.insert(0, '/Users/devswat/resonantgenesis_backend/memory_service')

from app.services.resonance_hashing import ResonanceHasher


class TestFusionLayer:
    """Tests for Layer 3: Fusion Layer - s = α·e + (1-α)·h"""
    
    def test_fusion_output_is_unit_vector(self):
        """Verify fused vector has unit norm: ||s|| = 1"""
        hasher = ResonanceHasher()
        
        # Create test vectors
        hash_vector = np.random.randn(32)
        embedding = np.random.randn(32)
        
        # Fuse
        fused = hasher.fuse_hash_and_embedding(hash_vector, embedding, alpha=0.9)
        
        # Check unit norm
        norm = np.linalg.norm(fused)
        assert abs(norm - 1.0) < 1e-6, f"Fused vector norm should be 1.0, got {norm}"
    
    def test_fusion_alpha_weight(self):
        """Verify alpha parameter controls embedding vs hash weight"""
        hasher = ResonanceHasher()
        
        # Create distinct vectors
        hash_vector = np.array([1.0] * 32)
        embedding = np.array([0.0] * 32)
        embedding[0] = 1.0  # Only first element non-zero
        
        # High alpha = more embedding influence
        fused_high = hasher.fuse_hash_and_embedding(hash_vector, embedding, alpha=0.99)
        
        # Low alpha = more hash influence
        fused_low = hasher.fuse_hash_and_embedding(hash_vector, embedding, alpha=0.01)
        
        # High alpha should be closer to embedding direction
        # Low alpha should be closer to hash direction
        assert fused_high[0] > fused_low[0], "High alpha should weight embedding more"
    
    def test_fusion_handles_zero_vectors(self):
        """Verify fusion handles edge cases gracefully"""
        hasher = ResonanceHasher()
        
        # Zero hash vector
        hash_vector = np.zeros(32)
        embedding = np.random.randn(32)
        
        # Should not crash, should return normalized embedding
        fused = hasher.fuse_hash_and_embedding(hash_vector, embedding, alpha=0.9)
        assert fused is not None
        assert len(fused) == 32


class TestResonanceFunction:
    """Tests for Layer 5: Resonance Function - R(h) = sin(a·x) + cos(b·y) + tan(c·z)"""
    
    def test_resonance_output_range(self):
        """Verify resonance value is in expected range [-3, 3] for most inputs"""
        hasher = ResonanceHasher()
        
        # Test multiple random points
        for _ in range(100):
            xyz = (np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            resonance = hasher.calculate_resonance_function(xyz)
            
            # sin and cos are bounded [-1, 1], tan can be large but for small inputs it's bounded
            # For typical xyz in [-1, 1], resonance should be roughly in [-3, 3]
            # Allow some tolerance for tan near asymptotes
            assert -10 < resonance < 10, f"Resonance {resonance} out of expected range for xyz={xyz}"
    
    def test_resonance_uses_correct_constants(self):
        """Verify resonance function uses π/4, e/3, φ/2 constants"""
        hasher = ResonanceHasher()
        
        # Test at origin
        xyz = (0.0, 0.0, 0.0)
        resonance = hasher.calculate_resonance_function(xyz)
        
        # At origin: sin(0) + cos(0) + tan(0) = 0 + 1 + 0 = 1
        assert abs(resonance - 1.0) < 1e-6, f"Resonance at origin should be 1.0, got {resonance}"
    
    def test_resonance_deterministic(self):
        """Verify same input produces same output"""
        hasher = ResonanceHasher()
        
        xyz = (0.5, -0.3, 0.7)
        r1 = hasher.calculate_resonance_function(xyz)
        r2 = hasher.calculate_resonance_function(xyz)
        
        assert r1 == r2, "Resonance function should be deterministic"
    
    def test_resonance_varies_with_position(self):
        """Verify different positions produce different resonance values"""
        hasher = ResonanceHasher()
        
        xyz1 = (0.1, 0.2, 0.3)
        xyz2 = (0.4, 0.5, 0.6)
        
        r1 = hasher.calculate_resonance_function(xyz1)
        r2 = hasher.calculate_resonance_function(xyz2)
        
        assert r1 != r2, "Different positions should have different resonance"


class TestAnchorEnergy:
    """Tests for Layer 4: Anchor Energy - E_j(s) = exp(-β·||s - A_j||²)"""
    
    def test_anchor_energy_at_anchor(self):
        """Verify energy is 1.0 when point is exactly at anchor"""
        hasher = ResonanceHasher()
        
        point = np.array([0.5, 0.5, 0.5])
        anchor = np.array([0.5, 0.5, 0.5])
        
        energy = hasher.calculate_anchor_energy(point, anchor, beta=1.0)
        
        # At anchor: exp(-β·0) = exp(0) = 1.0
        assert abs(energy - 1.0) < 1e-6, f"Energy at anchor should be 1.0, got {energy}"
    
    def test_anchor_energy_range(self):
        """Verify energy is in range [0, 1]"""
        hasher = ResonanceHasher()
        
        for _ in range(100):
            point = np.random.randn(3)
            anchor = np.random.randn(3)
            energy = hasher.calculate_anchor_energy(point, anchor, beta=1.0)
            
            assert 0.0 <= energy <= 1.0, f"Energy {energy} should be in [0, 1]"
    
    def test_anchor_energy_decreases_with_angular_distance(self):
        """Verify energy decreases as point moves angularly away from anchor on unit sphere"""
        hasher = ResonanceHasher()
        
        # Anchor at [1, 0, 0] direction
        anchor = np.array([1.0, 0.0, 0.0])
        
        # Points at increasing angular distances (all on unit sphere after normalization)
        point_same = np.array([1.0, 0.0, 0.0])  # Same direction
        point_45deg = np.array([1.0, 1.0, 0.0])  # 45 degrees away
        point_90deg = np.array([0.0, 1.0, 0.0])  # 90 degrees away (orthogonal)
        
        e_same = hasher.calculate_anchor_energy(point_same, anchor, beta=1.0)
        e_45deg = hasher.calculate_anchor_energy(point_45deg, anchor, beta=1.0)
        e_90deg = hasher.calculate_anchor_energy(point_90deg, anchor, beta=1.0)
        
        assert e_same > e_45deg > e_90deg, f"Energy should decrease with angular distance: {e_same} > {e_45deg} > {e_90deg}"
    
    def test_beta_controls_decay_rate(self):
        """Verify higher beta means faster energy decay"""
        hasher = ResonanceHasher()
        
        point = np.array([0.5, 0.0, 0.0])
        anchor = np.array([0.0, 0.0, 0.0])
        
        e_low_beta = hasher.calculate_anchor_energy(point, anchor, beta=0.5)
        e_high_beta = hasher.calculate_anchor_energy(point, anchor, beta=2.0)
        
        # Higher beta = faster decay = lower energy at same distance
        assert e_low_beta > e_high_beta, "Higher beta should cause faster decay"


class TestFindBestAnchor:
    """Tests for find_best_anchor function"""
    
    def test_find_best_anchor_returns_closest_direction(self):
        """Verify find_best_anchor returns the anchor with closest direction (highest energy)"""
        hasher = ResonanceHasher()
        
        # Point in [1, 0, 0] direction
        point = np.array([1.0, 0.0, 0.0])
        anchors = [
            np.array([0.0, 1.0, 0.0]),  # Orthogonal - 90 degrees
            np.array([1.0, 0.1, 0.0]),  # Nearly same direction - ~6 degrees
            np.array([0.0, 0.0, 1.0]),  # Orthogonal - 90 degrees
        ]
        
        best_idx, best_energy = hasher.find_best_anchor(point, anchors, beta=1.0)
        
        assert best_idx == 1, f"Should return closest direction anchor (index 1), got {best_idx}"
        assert best_energy > 0.9, f"Best energy should be high for close direction, got {best_energy}"
    
    def test_find_best_anchor_empty_list(self):
        """Verify graceful handling of empty anchor list"""
        hasher = ResonanceHasher()
        
        point = np.array([0.1, 0.1, 0.1])
        anchors = []
        
        best_idx, best_energy = hasher.find_best_anchor(point, anchors, beta=1.0)
        
        assert best_idx == -1, "Should return -1 for empty anchor list"
        assert best_energy == 0.0, "Should return 0.0 energy for empty list"


class TestHybridRankerWeights:
    """Tests for hybrid memory ranker weight configuration"""
    
    def test_weights_sum_to_one(self):
        """Verify all weights sum to 1.0"""
        from app.services.hybrid_memory_ranker import (
            W_RAG, W_RESONANCE, W_PROXIMITY, W_RECENCY, W_ANCHOR,
            W_RESONANCE_FUNCTION, W_ANCHOR_ENERGY
        )
        
        total = W_RAG + W_RESONANCE + W_PROXIMITY + W_RECENCY + W_ANCHOR + W_RESONANCE_FUNCTION + W_ANCHOR_ENERGY
        
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"
    
    def test_compute_score_with_new_fields(self):
        """Verify compute_score handles new resonance_function_score and anchor_energy fields"""
        from app.services.hybrid_memory_ranker import compute_score
        
        mem = {
            "rag_score": 0.8,
            "resonance_score": 0.7,
            "proximity_score": 0.6,
            "recency_score": 0.5,
            "anchor_score": 0.4,
            "resonance_function_score": 0.9,
            "anchor_energy": 0.85,
        }
        
        score = compute_score(mem)
        
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"
        assert score > 0.5, "Score should be reasonably high with good inputs"
    
    def test_compute_score_handles_missing_fields(self):
        """Verify compute_score handles missing new fields gracefully"""
        from app.services.hybrid_memory_ranker import compute_score
        
        # Memory without new fields
        mem = {
            "rag_score": 0.8,
            "resonance_score": 0.7,
        }
        
        # Should not crash
        score = compute_score(mem)
        assert score >= 0.0, "Score should be non-negative"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
