"""
MiniLM-L6-v2 Embeddings for Hash Sphere Memory
===============================================

Replaces Nomic Embed v1.5 with MiniLM-L6-v2 based on LOCOMO benchmark results.
MiniLM achieved 0.597 vs Nomic's 0.472 on the full 10-conversation evaluation.
"""
import logging
from typing import List, Literal
import numpy as np
import torch
import torch.nn.functional as F

from .config import settings

logger = logging.getLogger(__name__)


class MiniLMEmbeddings:
    """MiniLM-L6-v2 embeddings for Hash Sphere memory.
    
    Based on LOCOMO benchmark results, MiniLM-L6-v2 (384-dim) outperformed
    Nomic Embed v1.5 (512-dim) with a score of 0.597 vs 0.472.
    The key advantage is better deduplication balance at threshold 0.85.
    """
    
    def __init__(self):
        self.model = None
        self._initialized = False
        self.embedding_dim = 384
    
    def _ensure_initialized(self):
        """Lazy load the model to avoid startup delays."""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("🚀 Loading MiniLM-L6-v2 model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._initialized = True
                logger.info(f"✅ MiniLM-L6-v2 loaded (dim={self.embedding_dim})")
            except Exception as e:
                logger.error(f"❌ Failed to load MiniLM-L6-v2: {e}")
                self._initialized = False
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            List of normalized embeddings (384-dim)
        """
        self._ensure_initialized()
        
        if not self.model:
            return []
        
        try:
            # MiniLM doesn't need task prefixes
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True  # L2 normalization
            )
            
            # Convert to list of lists
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"❌ Failed to encode texts: {e}")
            return []
    
    def encode_single(self, text: str) -> List[float]:
        """Encode a single text to embedding.
        
        Args:
            text: Single text to encode
        
        Returns:
            Normalized embedding (384-dim)
        """
        result = self.encode([text])
        return result[0] if result else []
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def is_healthy(self) -> bool:
        """Check if the embedding service is healthy."""
        return self._initialized and self.model is not None


# Singleton instance for production
_minilm_instance = None

def get_minilm_embeddings() -> MiniLMEmbeddings:
    """Get singleton MiniLM embeddings instance."""
    global _minilm_instance
    if _minilm_instance is None:
        _minilm_instance = MiniLMEmbeddings()
    return _minilm_instance


# Backward compatibility aliases
def get_embeddings() -> MiniLMEmbeddings:
    """Get embeddings instance (backward compatibility)."""
    return get_minilm_embeddings()


# Factory function for easy switching
def create_embeddings(model_type: Literal["minilm", "nomic"] = "minilm"):
    """Create embeddings instance based on model type.
    
    Args:
        model_type: Type of embedding model to create
    
    Returns:
        Embeddings instance
    """
    if model_type == "minilm":
        return MiniLMEmbeddings()
    elif model_type == "nomic":
        # Import Nomic for backward compatibility
        from .embeddings import NomicEmbeddings
        return NomicEmbeddings()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
