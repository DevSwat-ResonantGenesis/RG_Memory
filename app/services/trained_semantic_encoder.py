"""
Trained Semantic Encoder — ML-based cluster/temperature/polarity prediction
=============================================================================

Replaces the hardcoded word lists in semantic_encoder.py with trained sklearn models.
Uses the same Nomic Embed embeddings that production already generates.

Pipeline:
    embedding (512-dim) → cluster_classifier → ALPHA/BETA/GAMMA/DELTA/EPSILON
    embedding (512-dim) → temperature_regressor → 0.0–1.0
    embedding (512-dim) → polarity_regressor → 0.0–1.0
    embedding (512-dim) → embedding_to_xyz → (x, y, z)

Model file: app/data/models/trained_semantic_encoder.pkl
Training:  python -m app.data.models.train_semantic_model
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Path to the trained model file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "models", "trained_semantic_encoder.pkl"
)


@dataclass
class TrainedSemanticResult:
    """Result from the trained semantic encoder."""
    x: float
    y: float
    z: float
    cluster: str
    cluster_confidence: float
    temperature: float
    polarity: float


class TrainedSemanticEncoder:
    """ML-trained semantic encoder using embedding-based classifiers.
    
    This is what semantic_encoder_model.pkl was meant to become:
    - learned_vocab → cluster_classifier (KNN on embeddings)
    - sentiment_classifier → temperature_regressor + polarity_regressor (Ridge)
    - word_clusterer → cluster_centroids (mean embeddings per cluster)
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._load_attempted = False
    
    def _ensure_loaded(self) -> bool:
        """Lazy-load the trained model. Returns True if loaded successfully."""
        if self._loaded:
            return True
        if self._load_attempted:
            return False
        
        self._load_attempted = True
        
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Trained semantic encoder model not found at {self.model_path}. "
                f"Run: python -m app.data.models.train_semantic_model"
            )
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            
            # Validate model structure
            required_keys = [
                "cluster_classifier", "label_encoder",
                "temperature_regressor", "polarity_regressor",
                "cluster_centroids",
            ]
            for key in required_keys:
                if key not in self._model:
                    logger.error(f"Trained model missing key: {key}")
                    return False
            
            self._loaded = True
            logger.info(
                f"Loaded trained semantic encoder v{self._model.get('version', '?')} "
                f"({self._model.get('training_samples', '?')} samples, "
                f"clusters: {self._model.get('cluster_names', [])})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load trained semantic encoder: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if the trained model is available."""
        return self._ensure_loaded()
    
    def predict_cluster(self, embedding: List[float]) -> Tuple[str, float]:
        """Predict semantic cluster from embedding.
        
        Args:
            embedding: 512-dim embedding vector
        
        Returns:
            (cluster_name, confidence) where confidence is from KNN distance weighting
        """
        if not self._ensure_loaded():
            return ("GAMMA", 0.0)  # Default fallback
        
        emb = np.array(embedding, dtype=np.float64).reshape(1, -1)
        
        # Predict cluster
        cluster_idx = self._model["cluster_classifier"].predict(emb)[0]
        cluster_name = self._model["label_encoder"].inverse_transform([cluster_idx])[0]
        
        # Get prediction probabilities for confidence
        probs = self._model["cluster_classifier"].predict_proba(emb)[0]
        confidence = float(probs.max())
        
        return (cluster_name, confidence)
    
    def predict_temperature(self, embedding: List[float]) -> float:
        """Predict temperature (urgency/warmth) from embedding.
        
        Returns:
            Temperature in [0, 1] range
        """
        if not self._ensure_loaded():
            return 0.5
        
        emb = np.array(embedding, dtype=np.float64).reshape(1, -1)
        temp = float(self._model["temperature_regressor"].predict(emb)[0])
        return max(0.0, min(1.0, temp))
    
    def predict_polarity(self, embedding: List[float]) -> float:
        """Predict polarity (positive/negative sentiment) from embedding.
        
        Returns:
            Polarity in [0, 1] range (0=negative, 1=positive)
        """
        if not self._ensure_loaded():
            return 0.5
        
        emb = np.array(embedding, dtype=np.float64).reshape(1, -1)
        pol = float(self._model["polarity_regressor"].predict(emb)[0])
        return max(0.0, min(1.0, pol))
    
    def get_cluster_centroids(self) -> Dict[str, np.ndarray]:
        """Get learned cluster centroids.
        
        Returns:
            Dict mapping cluster name to mean embedding vector
        """
        if not self._ensure_loaded():
            return {}
        return self._model.get("cluster_centroids", {})
    
    def encode(self, embedding: List[float]) -> TrainedSemanticResult:
        """Full semantic encoding from embedding.
        
        Takes an already-computed embedding and predicts all semantic properties.
        
        Args:
            embedding: 512-dim embedding vector (from Nomic Embed)
        
        Returns:
            TrainedSemanticResult with xyz, cluster, temperature, polarity
        """
        from .resonance_hashing import ResonanceHasher
        
        # Get XYZ from embedding projection
        x, y, z = ResonanceHasher.embedding_to_xyz(embedding)
        
        # Predict semantic properties
        cluster, confidence = self.predict_cluster(embedding)
        temperature = self.predict_temperature(embedding)
        polarity = self.predict_polarity(embedding)
        
        return TrainedSemanticResult(
            x=x, y=y, z=z,
            cluster=cluster,
            cluster_confidence=confidence,
            temperature=temperature,
            polarity=polarity,
        )
    
    def calculate_xyz(self, text: str) -> Tuple[float, float, float]:
        """Compatibility method for semantic_encoder.py interface.
        
        NOTE: This method generates its own embedding for the text.
        Prefer encode(embedding) when you already have the embedding.
        """
        # Generate embedding inline (slower — prefer passing embedding directly)
        try:
            from ..embeddings import NomicEmbeddings
            nomic = NomicEmbeddings(matryoshka_dim=512)
            embeddings = nomic.encode([text], task="search_document")
            if embeddings:
                x, y, z = self.encode(embeddings[0]).x, self.encode(embeddings[0]).y, self.encode(embeddings[0]).z
                return (x, y, z)
        except Exception as e:
            logger.warning(f"Trained encoder calculate_xyz failed: {e}")
        
        # Fallback to hardcoded encoder
        from .semantic_encoder import get_semantic_encoder
        return get_semantic_encoder().calculate_xyz(text)


# Global singleton
_trained_encoder: Optional[TrainedSemanticEncoder] = None


def get_trained_encoder() -> TrainedSemanticEncoder:
    """Get or create the global trained semantic encoder."""
    global _trained_encoder
    if _trained_encoder is None:
        _trained_encoder = TrainedSemanticEncoder()
    return _trained_encoder
