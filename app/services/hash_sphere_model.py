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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_WORD_RE = re.compile(r"[a-z']{2,}")

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
# Sentence-vs-word-centroid cosines are compressed into a narrow band, so we
# STANDARDIZE the 6 sims (z-score) before softmax to amplify RELATIVE cluster
# affinity — otherwise unrelated texts get near-identical distributions and thus
# spuriously high gravity. Temp applies to the standardized scores.
_SOFTMAX_TEMP = 2.5


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
        # word → {"cluster": idx, "warm": diff, "pos": diff}. Grows as memories arrive;
        # a word's semantic axis is stable, so this is a cheap permanent cache.
        self._word_cache: Dict[str, Dict] = {}

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
        """embedding → {clusters: {α…ζ}, temperature, polarity}. None if unavailable.

        Wave 2.5: if the trained projection head is active (flag on + weights
        present), it supersedes the fixed seed-centroid mapping below. When it is
        off or returns None, we fall through to the stable Wave-2 path — so the
        live behaviour is unchanged unless the head is explicitly enabled.
        """
        if not embedding:
            return None
        try:
            from .hash_sphere_projection import projection_head
            learned = projection_head.predict(embedding)
            if learned is not None:
                return learned
        except Exception:
            pass
        if not self._built:
            return None
        e = _normalize(list(embedding))

        # α…ζ via softmax over STANDARDIZED cosine similarity to cluster centroids.
        # Standardizing (z-score) turns the compressed absolute cosines into
        # relative affinities, so distinct topics land in distinct distributions.
        sims = [_cos(e, self._cluster_centroids[k]) for k in _CLUSTER_KEYS]
        mean = sum(sims) / len(sims)
        var = sum((s - mean) ** 2 for s in sims) / len(sims)
        std = math.sqrt(var) or 1e-6
        z = [(s - mean) / std for s in sims]
        mx = max(z)
        exps = [math.exp(_SOFTMAX_TEMP * (zi - mx)) for zi in z]
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

    def _classify_word(self, e: List[float]) -> Dict:
        """Classify a single (normalized) word embedding into a SOFT cluster
        distribution + bipolar temperature/polarity diffs.

        Word-vs-word-centroid is in-distribution (unlike sentence-vs-word-centroid).
        A soft distribution (standardized softmax) per word, averaged over the
        sentence, gives a smooth, robust α…ζ — far less noisy than a hard-argmax
        histogram over a handful of words."""
        sims = [_cos(e, self._cluster_centroids[k]) for k in _CLUSTER_KEYS]
        mean = sum(sims) / len(sims)
        var = sum((s - mean) ** 2 for s in sims) / len(sims)
        std = math.sqrt(var) or 1e-6
        z = [(s - mean) / std for s in sims]
        mx = max(z)
        exps = [math.exp(_SOFTMAX_TEMP * (zi - mx)) for zi in z]
        tot = sum(exps) or 1.0
        dist = [x / tot for x in exps]
        warm = (_cos(e, self._warm) - _cos(e, self._cold)) if (self._warm and self._cold) else 0.0
        pos = (_cos(e, self._positive) - _cos(e, self._negative)) if (self._positive and self._negative) else 0.0
        return {"dist": dist, "warm": warm, "pos": pos}

    async def axes_for_text(self, text: str, embeddings_generator) -> Optional[Dict]:
        """Word-level α…ζ / temperature / polarity for a text (the faithful design).

        Each content word is classified to its nearest cluster centroid; the
        sentence distribution is the normalized histogram of word clusters.
        Words are embedded once and cached. Returns None if the model isn't built.
        """
        if not self._built:
            return None
        words = _WORD_RE.findall((text or "").lower())
        if not words:
            return None
        uniq = list(dict.fromkeys(words))
        missing = [w for w in uniq if w not in self._word_cache]
        if missing:
            try:
                vecs = await embeddings_generator.generate(missing, task="search_document")
            except Exception as e:
                logger.warning("axes_for_text: word embed failed: %s", e)
                return None
            for w, v in zip(missing, vecs):
                self._word_cache[w] = self._classify_word(_normalize(v))

        acc = [0.0] * 6
        warm_sum = 0.0
        pos_sum = 0.0
        n = 0
        for w in words:
            wc = self._word_cache.get(w)
            if not wc:
                continue
            for i in range(6):
                acc[i] += wc["dist"][i]
            warm_sum += wc["warm"]
            pos_sum += wc["pos"]
            n += 1
        if n == 0:
            return None
        total = sum(acc) or 1.0
        clusters = {k: acc[i] / total for i, k in enumerate(_CLUSTER_KEYS)}

        def squash(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-8.0 * x))

        return {
            "clusters": clusters,
            "temperature": squash(warm_sum / n),
            "polarity": squash(pos_sum / n),
        }


hash_sphere_model = HashSphereModel()
