"""
Hash Sphere Model — vocab→12-axis prototype model (RFC-0002 Wave 2)
====================================================================

Wave 1 assigned α…ζ / temperature / polarity by EXACT word match against the
seed CLUSTER_WORDS dictionaries — so any word not in the ~600-word seed got a
uniform prior. Wave 2 generalizes those seeds to the ENTIRE vocabulary using the
frozen MiniLM embedding space:

  - Each of the 6 world axes (α…ζ) is defined by the CENTROID of its seed words'
    embeddings. Any text's α…ζ = softmax of cosine similarity to the 6 centroids.
  - Temperature = warm-centroid vs cold-centroid similarity.
  - Polarity    = positive-centroid vs negative-centroid similarity.

This is the "train on world vocabulary, seeded by CLUSTER_WORDS" step in its
robust, stable form: MiniLM supplies knowledge of every word; the seed lists
define the axis DIRECTIONS. The model is frozen (stable) — only the runtime
physics field changes live. Centroids are persisted to a JSON artifact so
restarts load instantly.

"physician" (not in seed) embeds near "doctor" (ALPHA seed) → classified ALPHA.
That is the generalization Wave 1 lacked.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .semantic_encoder import (
    CLUSTER_WORDS,
    SemanticCluster,
    WARM_WORDS,
    COLD_WORDS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
)

logger = logging.getLogger(__name__)

_ARTIFACT = Path(__file__).resolve().parents[2] / "data" / "models" / "hash_sphere_prototypes.json"

_CLUSTER_ORDER = [
    SemanticCluster.ALPHA, SemanticCluster.BETA, SemanticCluster.GAMMA,
    SemanticCluster.DELTA, SemanticCluster.EPSILON, SemanticCluster.ZETA,
]
_CLUSTER_KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
_SOFTMAX_TEMP = 12.0  # sharpness of the α…ζ softmax over cosine sims


def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def _mean(vectors: List[List[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return _normalize([a / len(vectors) for a in acc])


def _cos(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))  # inputs are pre-normalized


class HashSphereModel:
    """Frozen prototype model: MiniLM embedding → 12 semantic axes."""

    def __init__(self) -> None:
        self._cluster_centroids: Dict[str, List[float]] = {}
        self._warm: Optional[List[float]] = None
        self._cold: Optional[List[float]] = None
        self._positive: Optional[List[float]] = None
        self._negative: Optional[List[float]] = None
        self._built = False

    @property
    def ready(self) -> bool:
        return self._built

    def _load_artifact(self) -> bool:
        try:
            if _ARTIFACT.exists():
                data = json.loads(_ARTIFACT.read_text())
                self._cluster_centroids = {k: data["clusters"][k] for k in _CLUSTER_KEYS}
                self._warm = data.get("warm")
                self._cold = data.get("cold")
                self._positive = data.get("positive")
                self._negative = data.get("negative")
                self._built = all([
                    len(self._cluster_centroids) == 6,
                    self._warm, self._cold, self._positive, self._negative,
                ])
                if self._built:
                    logger.info("HashSphereModel: loaded prototype artifact")
                return self._built
        except Exception as e:
            logger.warning("HashSphereModel: artifact load failed: %s", e)
        return False

    def _save_artifact(self) -> None:
        try:
            _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
            _ARTIFACT.write_text(json.dumps({
                "clusters": self._cluster_centroids,
                "warm": self._warm, "cold": self._cold,
                "positive": self._positive, "negative": self._negative,
            }))
            logger.info("HashSphereModel: saved prototype artifact")
        except Exception as e:
            logger.warning("HashSphereModel: artifact save failed: %s", e)

    async def ensure_built(self, embeddings_generator) -> bool:
        """Build centroids from seed words (idempotent). Loads artifact if present."""
        if self._built:
            return True
        if self._load_artifact():
            return True
        try:
            async def embed(words: List[str]) -> List[List[float]]:
                if not words:
                    return []
                vecs = await embeddings_generator.generate(list(words), task="search_document")
                return [_normalize(v) for v in vecs]

            for key, cluster in zip(_CLUSTER_KEYS, _CLUSTER_ORDER):
                centroid = _mean(await embed(sorted(CLUSTER_WORDS[cluster])))
                if centroid is None:
                    return False
                self._cluster_centroids[key] = centroid

            self._warm = _mean(await embed(sorted(WARM_WORDS)))
            self._cold = _mean(await embed(sorted(COLD_WORDS)))
            self._positive = _mean(await embed(sorted(POSITIVE_WORDS)))
            self._negative = _mean(await embed(sorted(NEGATIVE_WORDS)))

            self._built = all([
                len(self._cluster_centroids) == 6,
                self._warm, self._cold, self._positive, self._negative,
            ])
            if self._built:
                self._save_artifact()
                logger.info("HashSphereModel: built prototypes from seed vocabulary")
            return self._built
        except Exception as e:
            logger.warning("HashSphereModel: build failed: %s", e)
            return False

    def predict(self, embedding: Optional[List[float]]) -> Optional[Dict]:
        """embedding → {clusters: {α…ζ}, temperature, polarity}. None if unavailable."""
        if not self._built or not embedding:
            return None
        e = _normalize(list(embedding))

        # α…ζ via softmax over cosine similarity to cluster centroids
        sims = [_cos(e, self._cluster_centroids[k]) for k in _CLUSTER_KEYS]
        mx = max(sims)
        exps = [math.exp(_SOFTMAX_TEMP * (s - mx)) for s in sims]
        total = sum(exps) or 1.0
        clusters = {k: exps[i] / total for i, k in enumerate(_CLUSTER_KEYS)}

        # temperature / polarity via bipolar centroid contrast → [0,1]
        def bipolar(pos_c, neg_c) -> float:
            if not pos_c or not neg_c:
                return 0.5
            d = _cos(e, pos_c) - _cos(e, neg_c)
            return 1.0 / (1.0 + math.exp(-6.0 * d))  # logistic squash

        temperature = bipolar(self._warm, self._cold)
        polarity = bipolar(self._positive, self._negative)
        return {"clusters": clusters, "temperature": temperature, "polarity": polarity}


hash_sphere_model = HashSphereModel()
