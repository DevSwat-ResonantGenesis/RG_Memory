"""Embeddings generator for memory service.

Supports multiple embedding providers:
1. MiniLM-L6-v2 (local, free, LOCOMO-optimized) - DEFAULT
2. Nomic Embed v1.5 (local, free, 512-dim) - optional
3. OpenAI API (cloud, paid)
4. Hash-based fallback (development only)

MiniLM-L6-v2 features:
- 384 dimensions
- Optimized for LOCOMO benchmark (0.597 vs Nomic's 0.472)
- Better deduplication balance at threshold 0.85
- Faster inference than Nomic
"""

import hashlib
import logging
import math
from typing import List, Optional, Literal

import httpx
import torch
import torch.nn.functional as F

from .config import settings

logger = logging.getLogger(__name__)

# Import MiniLM for LOCOMO-optimized embeddings
from .embeddings_minilm import MiniLMEmbeddings

# Task prefixes for Nomic Embed (kept for backward compatibility)
TASK_PREFIXES = {
    "search_document": "search_document: ",
    "search_query": "search_query: ",
    "clustering": "clustering: ",
    "classification": "classification: ",
}


class NomicEmbeddings:
    """Nomic Embed v1.5 - Local semantic embeddings.
    
    Uses sentence-transformers with Matryoshka support for
    flexible dimension sizes (768, 512, 256, 128, 64).
    """
    
    def __init__(self, matryoshka_dim: int = 512):
        self.model = None
        self.matryoshka_dim = matryoshka_dim
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy load the model to avoid startup delays."""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("🚀 Loading Nomic Embed v1.5 model...")
                self.model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True
                )
                self._initialized = True
                logger.info(f"✅ Nomic Embed loaded (dim={self.matryoshka_dim})")
            except Exception as e:
                logger.error(f"❌ Failed to load Nomic Embed: {e}")
                self._initialized = False
    
    def encode(
        self,
        texts: List[str],
        task: Literal["search_document", "search_query", "clustering", "classification"] = "search_document"
    ) -> List[List[float]]:
        """Encode texts to embeddings with task-specific prefix.
        
        Args:
            texts: List of texts to encode
            task: Task type for prefix (affects embedding quality)
        
        Returns:
            List of normalized embeddings
        """
        self._ensure_initialized()
        
        if not self.model:
            return []
        
        # Add task prefix
        prefix = TASK_PREFIXES.get(task, "search_document: ")
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=True)
        
        # Apply Matryoshka dimension reduction
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :self.matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()
    
    @property
    def dimensions(self) -> int:
        return self.matryoshka_dim


class EmbeddingsGenerator:
    """Generate embeddings using MiniLM, Nomic, OpenAI, or fallback.
    
    Priority order (based on LOCOMO benchmark results):
    1. MiniLM-L6-v2 (local, free, LOCOMO-optimized) - DEFAULT
    2. Nomic Embed v1.5 (local, free) - optional
    3. OpenAI API - if API key configured
    4. Hash-based fallback - development only
    """

    def __init__(self):
        self.openai_api_key = getattr(settings, "OPENAI_API_KEY", None)
        self.openai_model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_dimensions = getattr(settings, "EMBEDDING_DIMENSIONS", 1536)
        
        # MiniLM - LOCOMO optimized (384-dim, 0.597 score)
        self.minilm = MiniLMEmbeddings()
        self._use_minilm = getattr(settings, "USE_MINILM_EMBED", True)
        
        # Nomic Embed with configurable dimensions (512-dim, 0.472 score)
        matryoshka_dim = getattr(settings, "NOMIC_MATRYOSHKA_DIM", 512)
        self.nomic = NomicEmbeddings(matryoshka_dim=matryoshka_dim)
        self._use_nomic = getattr(settings, "USE_NOMIC_EMBED", False)  # Disabled by default
    
    @property
    def dimensions(self) -> int:
        """Return current embedding dimensions."""
        if self._use_minilm and self.minilm._initialized:
            return self.minilm.embedding_dim
        if self._use_nomic and self.nomic._initialized:
            return self.nomic.dimensions
        return self.openai_dimensions

    async def generate(
        self,
        texts: List[str],
        task: str = "search_document"
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            task: Task type for Nomic (ignored for MiniLM/OpenAI)
        """
        # Try MiniLM first (LOCOMO optimized, 0.597 score)
        if self._use_minilm:
            try:
                embeddings = self.minilm.encode(texts)
                if embeddings:
                    logger.debug(f"✅ Generated {len(embeddings)} MiniLM embeddings")
                    return embeddings
            except Exception as e:
                logger.warning(f"⚠️ MiniLM failed: {e}, falling back...")
                self._use_minilm = False
        
        # Try Nomic Embed next (512-dim, 0.472 score)
        if self._use_nomic:
            try:
                embeddings = self.nomic.encode(texts, task=task)
                if embeddings:
                    logger.debug(f"✅ Generated {len(embeddings)} Nomic embeddings")
                    return embeddings
            except Exception as e:
                logger.warning(f"⚠️ Nomic Embed failed: {e}, falling back...")
                self._use_nomic = False
        
        # Fallback to OpenAI
        if self.openai_api_key:
            return await self._generate_openai(texts)
        
        # Last resort: hash-based (not semantic!)
        logger.warning("⚠️ Using hash-based embeddings (not semantic!)")
        return [self._generate_simple(text) for text in texts]

    async def generate_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        embeddings = await self.generate([query], task="search_query")
        return embeddings[0] if embeddings else []
    
    async def generate_document(self, document: str) -> List[float]:
        """Generate embedding for a document."""
        embeddings = await self.generate([document], task="search_document")
        return embeddings[0] if embeddings else []

    async def _generate_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "input": texts,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return [item["embedding"] for item in data["data"]]
        except httpx.RequestError as e:
            logger.error(f"OpenAI API error: {e}")

        # Fallback to simple embeddings
        return [self._generate_simple(text) for text in texts]

    def _generate_simple(self, text: str) -> List[float]:
        """Generate simple hash-based embeddings for development.
        
        WARNING: This is NOT semantic - just deterministic noise.
        Only use for development when no embedding model is available.
        """
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Use current dimensions
        dim = self.dimensions
        embedding = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            value = value * math.cos(i * 0.01)
            embedding.append(value)
        
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            # Handle dimension mismatch by truncating to smaller
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
    
    def semantic_search(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        top_k: int = 10
    ) -> List[tuple]:
        """Find most similar documents to query.
        
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        for i, doc_emb in enumerate(document_embeddings):
            sim = self.cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def generate_batch(
        self,
        texts: List[str],
        task: str = "search_document"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with caching.
        
        Checks cache first, generates only for uncached texts,
        then caches new embeddings.
        
        Args:
            texts: List of texts to embed
            task: Task type for Nomic (search_document, search_query, etc.)
            
        Returns:
            List of embeddings in same order as input texts
        """
        from .services.embedding_cache import embedding_cache
        
        if not texts:
            return []
        
        # Check cache for each text
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached:
                cached_results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self.generate(uncached_texts, task=task)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cached_results[idx] = embedding
                embedding_cache.set(texts[idx], embedding)
        
        # Return in original order
        return [cached_results[i] for i in range(len(texts))]


embeddings_generator = EmbeddingsGenerator()
