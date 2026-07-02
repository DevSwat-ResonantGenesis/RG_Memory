"""
Generic SentenceTransformer embedding backend (base-embedder upgrade)
=====================================================================

Loads any sentence-transformers model by name (default BAAI/bge-base-en-v1.5,
768-dim) — a much stronger retrieval embedder than MiniLM-L6 (384). Handles the
BGE-family query instruction (BGE recommends prefixing the QUERY side only).

Configurable via env:
  MEMORY_EMBEDDING_MODEL   (default "BAAI/bge-base-en-v1.5")
  MEMORY_EMBEDDING_DIM     (default 768)
  MEMORY_EMBEDDING_QUERY_INSTRUCTION (default the BGE retrieval instruction)
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
_DEFAULT_DIM = int(os.getenv("MEMORY_EMBEDDING_DIM", "768"))
_DEFAULT_QUERY_INSTRUCTION = os.getenv(
    "MEMORY_EMBEDDING_QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)


class STEmbeddings:
    """SentenceTransformer embedder with BGE-style query instruction."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, dim: int = _DEFAULT_DIM):
        self.model_name = model_name
        self.embedding_dim = dim
        self.query_instruction = _DEFAULT_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""
        self.model = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("🚀 Loading embedder %s ...", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("✅ Embedder loaded: %s (dim=%d)", self.model_name, self.embedding_dim)
        except Exception as e:
            logger.error("❌ Failed to load embedder %s: %s", self.model_name, e)
            self._initialized = False

    def encode(self, texts: List[str], task: str = "search_document") -> List[List[float]]:
        self._ensure_initialized()
        if not self.model:
            return []
        try:
            prefix = self.query_instruction if task == "search_query" else ""
            inputs = [prefix + (t or "") for t in texts]
            vecs = self.model.encode(inputs, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        except Exception as e:
            logger.error("Embedding failed (%s): %s", self.model_name, e)
            return []
