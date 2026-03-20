"""
Semantic Encoder - Hybrid Hash Sphere Semantic Encoding System

This module provides semantic XYZ coordinate calculation for the Hash Sphere.
Used alongside resonance_hashing.py which handles hash generation.

HYBRID ARCHITECTURE:
    - semantic_encoder.py: XYZ coordinates (semantic-based, better clustering)
    - resonance_hashing.py: Hash generation (original approach)

XYZ Calculation:
    X = f(cluster_vector) - weighted combination of semantic clusters
    Y = f(temperature, polarity) - emotional/sentiment space
    Z = f(complexity) - text complexity

Based on: Hash Sphere Memory System Vision
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SEMANTIC CLUSTERS (Alpha, Beta, Gamma, Delta, Epsilon, Zeta)
# ============================================================================

class SemanticCluster(Enum):
    """Six fundamental semantic clusters for word classification."""
    ALPHA = 1    # Living/Animate
    BETA = 2     # Inanimate/Objects
    GAMMA = 3    # Abstract/Concepts
    DELTA = 4    # Actions/Processes
    EPSILON = 5  # Qualities/Properties
    ZETA = 6     # Relations/Connections


# Cluster word dictionaries
CLUSTER_WORDS = {
    SemanticCluster.ALPHA: {
        "person", "people", "human", "man", "woman", "child", "user", "customer",
        "client", "investor", "founder", "developer", "engineer", "designer",
        "manager", "leader", "team", "employee", "partner", "friend", "family",
        "doctor", "teacher", "student", "artist", "scientist", "expert",
        "animal", "dog", "cat", "bird", "fish", "horse", "lion", "tiger",
        "company", "startup", "organization", "business", "corporation", "agency",
        "government", "institution", "community", "group", "society",
    },
    SemanticCluster.BETA: {
        "object", "thing", "item", "product", "device", "machine", "tool",
        "computer", "phone", "car", "building", "house", "room", "table",
        "chair", "book", "paper", "document", "file", "folder", "box",
        "material", "metal", "wood", "plastic", "glass", "stone", "water",
        "food", "money", "currency", "asset", "resource", "equipment",
        "code", "software", "app", "application", "website", "database",
        "server", "api", "interface", "system", "platform", "framework",
    },
    SemanticCluster.GAMMA: {
        "idea", "concept", "theory", "principle", "strategy", "plan", "goal",
        "vision", "mission", "purpose", "meaning", "value", "belief", "opinion",
        "project", "program", "initiative", "campaign", "effort", "work",
        "task", "job", "assignment", "challenge", "problem", "solution",
        "opportunity", "risk", "issue", "concern", "question", "answer",
        "emotion", "feeling", "mood", "happiness", "sadness", "anger", "fear",
        "love", "hate", "hope", "trust", "confidence", "doubt", "anxiety",
        "market", "industry", "sector", "niche", "segment", "audience",
        "revenue", "profit", "growth", "success", "failure", "performance",
        "investment", "funding", "capital", "budget", "cost", "price",
    },
    SemanticCluster.DELTA: {
        "do", "make", "create", "build", "develop", "design", "implement",
        "execute", "run", "start", "stop", "begin", "end", "finish", "complete",
        "find", "search", "look", "discover", "explore", "investigate", "analyze",
        "think", "consider", "decide", "choose", "select", "pick", "prefer",
        "say", "tell", "speak", "write", "read", "listen", "hear", "see", "watch",
        "go", "come", "move", "travel", "walk", "fly", "drive",
        "give", "take", "send", "receive", "share", "transfer", "exchange",
        "buy", "sell", "pay", "invest", "spend", "save", "earn", "lose",
        "help", "support", "assist", "guide", "lead", "manage", "control",
        "learn", "teach", "train", "study", "practice", "improve", "grow",
        "change", "update", "modify", "edit", "fix", "repair", "solve",
        "connect", "link", "join", "combine", "merge", "integrate", "sync",
        "launch", "deploy", "release", "publish", "ship", "deliver",
    },
    SemanticCluster.EPSILON: {
        "good", "bad", "great", "excellent", "amazing", "wonderful", "terrible",
        "big", "small", "large", "tiny", "huge", "massive", "little",
        "fast", "slow", "quick", "rapid", "instant", "gradual",
        "new", "old", "young", "ancient", "modern", "current", "recent",
        "easy", "hard", "difficult", "simple", "complex", "complicated",
        "important", "critical", "essential", "vital", "key", "main", "primary",
        "best", "worst", "better", "worse", "optimal", "ideal", "perfect",
        "high", "low", "top", "bottom", "upper", "lower", "middle",
        "first", "last", "next", "previous", "final", "initial",
        "true", "false", "real", "fake", "genuine", "authentic",
        "positive", "negative", "neutral", "active", "passive",
        "open", "closed", "public", "private", "free", "paid",
        "strong", "weak", "powerful", "effective", "efficient",
        "nice", "beautiful", "pretty", "ugly",
    },
    SemanticCluster.ZETA: {
        "in", "on", "at", "to", "from", "with", "without", "by", "for",
        "about", "through", "during", "before", "after", "between", "among",
        "under", "over", "above", "below", "inside", "outside", "within",
        "and", "or", "but", "so", "because", "if", "when", "while", "although",
        "however", "therefore", "thus", "hence", "moreover", "furthermore",
        "like", "as", "than", "versus", "against", "toward", "towards",
        "into", "onto", "upon", "across", "along", "around", "behind",
    },
}

# Temperature words
WARM_WORDS = {
    "love", "enjoy", "excited", "happy", "glad", "need", "want", "help", "fix",
    "urgent", "now", "immediately", "asap", "quickly", "hurry", "rush",
    "amazing", "awesome", "hate", "angry", "frustrated",
}

COLD_WORDS = {
    "maybe", "perhaps", "consider", "analyze", "evaluate", "assess", "review",
    "think", "ponder", "reflect", "contemplate", "study", "research",
    "sometime", "eventually", "later", "whenever", "someday",
}

# Polarity words
POSITIVE_WORDS = {
    "love", "enjoy", "good", "great", "nice", "beautiful", "happy", "excited",
    "glad", "amazing", "wonderful", "excellent", "perfect", "success", "win",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "hate", "sad", "problem", "bug", "issue", "fail",
    "error", "wrong", "broken", "difficult", "hard", "frustrating",
}


@dataclass
class SemanticXYZ:
    """Semantic XYZ coordinates."""
    x: float
    y: float
    z: float
    dominant_cluster: SemanticCluster
    temperature: float
    polarity: float


class SemanticEncoder:
    """
    Semantic encoder for XYZ coordinate calculation.
    
    Used alongside resonance_hashing.py for hybrid approach:
    - This module: XYZ coordinates (semantic-based)
    - resonance_hashing.py: Hash generation
    """
    
    def __init__(self):
        self.cluster_words = CLUSTER_WORDS
        self.warm_words = WARM_WORDS
        self.cold_words = COLD_WORDS
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
    
    def calculate_xyz(self, text: str) -> Tuple[float, float, float]:
        """
        Calculate semantic XYZ coordinates from text.
        
        Returns:
            Tuple of (x, y, z) coordinates in range [0, 1]
        """
        words = text.lower().split()
        
        # Calculate cluster distribution
        cluster_counts = {c: 0 for c in SemanticCluster}
        for word in words:
            for cluster, cluster_words in self.cluster_words.items():
                if word in cluster_words:
                    cluster_counts[cluster] += 1
        
        total = sum(cluster_counts.values()) + 1e-9
        cluster_dist = [cluster_counts[c] / total for c in sorted(cluster_counts.keys(), key=lambda x: x.value)]
        
        # Temperature
        warm = sum(1 for w in words if w in self.warm_words)
        cold = sum(1 for w in words if w in self.cold_words)
        temperature = 0.5 + 0.1 * (warm - cold)
        temperature = max(0, min(1, temperature))
        
        # Polarity
        pos = sum(1 for w in words if w in self.positive_words)
        neg = sum(1 for w in words if w in self.negative_words)
        polarity = 0.5 + 0.1 * (pos - neg)
        polarity = max(0, min(1, polarity))
        
        # X: Weighted cluster position
        weights = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
        x = sum(cv * w for cv, w in zip(cluster_dist, weights))
        x = min(1.0, max(0.0, x))
        
        # Y: Temperature-Polarity space
        y = (temperature * 0.6) + (polarity * 0.4)
        y = min(1.0, max(0.0, y))
        
        # Z: Complexity
        complexity = min(1.0, len(words) / 20)
        z = complexity
        
        return (x, y, z)
    
    def encode(self, text: str) -> SemanticXYZ:
        """
        Full semantic encoding of text.
        
        Returns:
            SemanticXYZ with coordinates and metadata
        """
        words = text.lower().split()
        x, y, z = self.calculate_xyz(text)
        
        # Find dominant cluster
        cluster_counts = {c: 0 for c in SemanticCluster}
        for word in words:
            for cluster, cluster_words in self.cluster_words.items():
                if word in cluster_words:
                    cluster_counts[cluster] += 1
        
        dominant = max(cluster_counts.keys(), key=lambda c: cluster_counts[c])
        
        # Temperature and polarity
        warm = sum(1 for w in words if w in self.warm_words)
        cold = sum(1 for w in words if w in self.cold_words)
        temperature = 0.5 + 0.1 * (warm - cold)
        temperature = max(0, min(1, temperature))
        
        pos = sum(1 for w in words if w in self.positive_words)
        neg = sum(1 for w in words if w in self.negative_words)
        polarity = 0.5 + 0.1 * (pos - neg)
        polarity = max(0, min(1, polarity))
        
        return SemanticXYZ(
            x=x,
            y=y,
            z=z,
            dominant_cluster=dominant,
            temperature=temperature,
            polarity=polarity,
        )


# Global instance
_semantic_encoder = None

def get_semantic_encoder() -> SemanticEncoder:
    """Get or create the global semantic encoder instance."""
    global _semantic_encoder
    if _semantic_encoder is None:
        _semantic_encoder = SemanticEncoder()
    return _semantic_encoder
