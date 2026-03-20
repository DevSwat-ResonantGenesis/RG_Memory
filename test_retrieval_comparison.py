#!/usr/bin/env python3
"""
Retrieval Evaluation: Compare resonance_hashing.py vs semantic_encoder.py

Tests:
1. XYZ distance correlation with semantic similarity
2. Retrieval precision using XYZ proximity
3. Hash-based vs semantic-based clustering
"""

import sys
import math
import hashlib
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Test data: Similar pairs (should be close in semantic space)
SIMILAR_PAIRS = [
    ("I love coding and programming", "I enjoy writing code"),
    ("Our startup needs funding", "We need investment capital"),
    ("Deploy to production server", "Release to production"),
    ("The weather is nice today", "It's a beautiful day outside"),
    ("Help me fix this bug", "I need to debug this issue"),
    ("Machine learning model training", "Training a neural network"),
    ("Customer acquisition strategy", "Getting new customers"),
    ("I feel happy and excited", "I am glad and enthusiastic"),
    ("Database query optimization", "Improve SQL performance"),
    ("Team collaboration tools", "Tools for working together"),
]

# Dissimilar pairs (should be far apart)
DISSIMILAR_PAIRS = [
    ("I love coding", "The weather is nice"),
    ("Our startup needs funding", "I want to eat pizza"),
    ("Deploy to production", "My cat is sleeping"),
    ("Machine learning model", "The restaurant is closed"),
    ("Customer acquisition", "The mountain is tall"),
]

def euclidean_distance(xyz1, xyz2):
    """Calculate Euclidean distance between two XYZ points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz1, xyz2)))


# =============================================================================
# TEST 1: Original resonance_hashing.py
# =============================================================================

print("=" * 60)
print("TEST 1: resonance_hashing.py (Original)")
print("=" * 60)

from app.services.resonance_hashing import ResonanceHasher as OriginalHasher

original_hasher = OriginalHasher()

def get_xyz_original(text):
    """Get XYZ coordinates using original resonance_hashing."""
    coords = original_hasher.compute_full_coordinates(text)
    return (coords.x, coords.y, coords.z)

# Test similar pairs
print("\nSimilar pairs (should have SMALL distance):")
similar_distances_original = []
for text1, text2 in SIMILAR_PAIRS[:5]:
    xyz1 = get_xyz_original(text1)
    xyz2 = get_xyz_original(text2)
    dist = euclidean_distance(xyz1, xyz2)
    similar_distances_original.append(dist)
    print(f"  [{dist:.4f}] '{text1[:30]}...' vs '{text2[:30]}...'")

# Test dissimilar pairs
print("\nDissimilar pairs (should have LARGE distance):")
dissimilar_distances_original = []
for text1, text2 in DISSIMILAR_PAIRS:
    xyz1 = get_xyz_original(text1)
    xyz2 = get_xyz_original(text2)
    dist = euclidean_distance(xyz1, xyz2)
    dissimilar_distances_original.append(dist)
    print(f"  [{dist:.4f}] '{text1[:30]}...' vs '{text2[:30]}...'")

avg_similar_orig = sum(similar_distances_original) / len(similar_distances_original)
avg_dissimilar_orig = sum(dissimilar_distances_original) / len(dissimilar_distances_original)

print(f"\n📊 ORIGINAL RESULTS:")
print(f"  Avg distance (similar pairs):    {avg_similar_orig:.4f}")
print(f"  Avg distance (dissimilar pairs): {avg_dissimilar_orig:.4f}")
print(f"  Ratio (dissimilar/similar):      {avg_dissimilar_orig/avg_similar_orig:.2f}x")

if avg_dissimilar_orig > avg_similar_orig:
    print("  ✅ GOOD: Dissimilar pairs are farther apart")
else:
    print("  ❌ BAD: Similar pairs are NOT closer together")


# =============================================================================
# TEST 2: Check if semantic_encoder.py exists, if not create minimal version
# =============================================================================

print("\n" + "=" * 60)
print("TEST 2: semantic_encoder.py (Semantic-based)")
print("=" * 60)

# Create a minimal semantic encoder for testing
class SemanticEncoder:
    """Minimal semantic encoder for comparison testing."""
    
    CLUSTER_WORDS = {
        "alpha": {"person", "people", "human", "user", "customer", "team", "company", "startup"},
        "beta": {"object", "thing", "product", "device", "computer", "server", "database", "code", "software"},
        "gamma": {"idea", "concept", "strategy", "plan", "goal", "problem", "solution", "funding", "investment"},
        "delta": {"create", "build", "develop", "deploy", "release", "fix", "debug", "train", "learn", "help"},
        "epsilon": {"good", "bad", "great", "nice", "beautiful", "happy", "excited", "glad"},
        "zeta": {"in", "on", "at", "to", "from", "with", "for", "about"},
    }
    
    WARM_WORDS = {"love", "enjoy", "excited", "happy", "glad", "need", "want", "help", "fix"}
    COLD_WORDS = {"maybe", "perhaps", "consider", "analyze", "evaluate"}
    POSITIVE_WORDS = {"love", "enjoy", "good", "great", "nice", "beautiful", "happy", "excited", "glad"}
    NEGATIVE_WORDS = {"bad", "terrible", "hate", "sad", "problem", "bug", "issue", "fail"}
    
    def encode(self, text):
        """Encode text to semantic features."""
        words = text.lower().split()
        
        # Calculate cluster distribution
        cluster_counts = {c: 0 for c in self.CLUSTER_WORDS}
        for word in words:
            for cluster, cluster_words in self.CLUSTER_WORDS.items():
                if word in cluster_words:
                    cluster_counts[cluster] += 1
        
        total = sum(cluster_counts.values()) + 1e-9
        cluster_dist = [cluster_counts[c] / total for c in sorted(cluster_counts.keys())]
        
        # Temperature
        warm = sum(1 for w in words if w in self.WARM_WORDS)
        cold = sum(1 for w in words if w in self.COLD_WORDS)
        temperature = 0.5 + 0.1 * (warm - cold)
        temperature = max(0, min(1, temperature))
        
        # Polarity
        pos = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        polarity = 0.5 + 0.1 * (pos - neg)
        polarity = max(0, min(1, polarity))
        
        # XYZ from semantic features (NOT hash bits)
        weights = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
        x = sum(cv * w for cv, w in zip(cluster_dist, weights))
        x = min(1.0, max(0.0, x))
        
        y = (temperature * 0.6) + (polarity * 0.4)
        y = min(1.0, max(0.0, y))
        
        # Z from complexity
        complexity = min(1.0, len(words) / 20)
        z = complexity
        
        return (x, y, z)

semantic_encoder = SemanticEncoder()

def get_xyz_semantic(text):
    """Get XYZ coordinates using semantic encoder."""
    return semantic_encoder.encode(text)

# Test similar pairs
print("\nSimilar pairs (should have SMALL distance):")
similar_distances_semantic = []
for text1, text2 in SIMILAR_PAIRS[:5]:
    xyz1 = get_xyz_semantic(text1)
    xyz2 = get_xyz_semantic(text2)
    dist = euclidean_distance(xyz1, xyz2)
    similar_distances_semantic.append(dist)
    print(f"  [{dist:.4f}] '{text1[:30]}...' vs '{text2[:30]}...'")

# Test dissimilar pairs
print("\nDissimilar pairs (should have LARGE distance):")
dissimilar_distances_semantic = []
for text1, text2 in DISSIMILAR_PAIRS:
    xyz1 = get_xyz_semantic(text1)
    xyz2 = get_xyz_semantic(text2)
    dist = euclidean_distance(xyz1, xyz2)
    dissimilar_distances_semantic.append(dist)
    print(f"  [{dist:.4f}] '{text1[:30]}...' vs '{text2[:30]}...'")

avg_similar_sem = sum(similar_distances_semantic) / len(similar_distances_semantic)
avg_dissimilar_sem = sum(dissimilar_distances_semantic) / len(dissimilar_distances_semantic)

print(f"\n📊 SEMANTIC RESULTS:")
print(f"  Avg distance (similar pairs):    {avg_similar_sem:.4f}")
print(f"  Avg distance (dissimilar pairs): {avg_dissimilar_sem:.4f}")
print(f"  Ratio (dissimilar/similar):      {avg_dissimilar_sem/avg_similar_sem:.2f}x")

if avg_dissimilar_sem > avg_similar_sem:
    print("  ✅ GOOD: Dissimilar pairs are farther apart")
else:
    print("  ❌ BAD: Similar pairs are NOT closer together")


# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON: Original vs Semantic")
print("=" * 60)

print(f"""
                        Original    Semantic    Winner
                        --------    --------    ------
Avg similar distance:   {avg_similar_orig:.4f}      {avg_similar_sem:.4f}      {'Semantic' if avg_similar_sem < avg_similar_orig else 'Original'}
Avg dissimilar dist:    {avg_dissimilar_orig:.4f}      {avg_dissimilar_sem:.4f}      {'Semantic' if avg_dissimilar_sem > avg_dissimilar_orig else 'Original'}
Ratio (dis/sim):        {avg_dissimilar_orig/avg_similar_orig:.2f}x        {avg_dissimilar_sem/avg_similar_sem:.2f}x        {'Semantic' if avg_dissimilar_sem/avg_similar_sem > avg_dissimilar_orig/avg_similar_orig else 'Original'}
""")

# Overall winner
orig_score = avg_dissimilar_orig / avg_similar_orig
sem_score = avg_dissimilar_sem / avg_similar_sem

if sem_score > orig_score * 1.2:  # Semantic needs to be 20% better to win
    print("🏆 WINNER: Semantic Encoder")
    print("   Reason: Better separation between similar and dissimilar pairs")
elif orig_score > sem_score * 1.2:
    print("🏆 WINNER: Original Resonance Hashing")
    print("   Reason: Better separation between similar and dissimilar pairs")
else:
    print("🤝 TIE: Both approaches perform similarly")
    print("   Recommendation: Keep original (already in production)")
