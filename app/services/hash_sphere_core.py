"""
Hash Sphere Core — the 12-D semantic manifold (RFC-0002 Wave 1)
================================================================

A memory is a POINT in a fixed 12-dimensional semantic space. Its hash is its
quantized position (Text ≡ 12-D coordinate ≡ hash). Retrieval ranks by gravity
(RBF proximity) in this space — the hash sphere is the primary brain; pgvector
cosine + BM25 are only the candidate-recall floor. XYZ is visualization only and
is NOT computed here.

The 12 dimensions:
    0-5  α β γ δ ε ζ   world-vocabulary probability (Living/Inanimate/Abstract/
                        Action/Quality/Relation), sums to 1
    6    temperature   urgency / linguistic intensity (0-1)
    7    polarity      sentiment valence (0-1)
    8-10 spin (3-D)    intensity / complexity / abstraction of the thought
    11   resonance     derived harmonic signature — NOT a metric axis

Distance/gravity run on the 11-D METRIC core (dims 0-10). Resonance (dim 11) is a
derived scalar used as a corroborating score and folded into the hash; it is
never an L2 coordinate (its classic tan term is unbounded, so we use a bounded
harmonic here).

Wave 1 seeds α…ζ / temperature / polarity from the existing CLUSTER_WORDS
dictionaries so the space works UNTRAINED and can be measured before the
vocab→12-D model (Wave 2) replaces the seed lookup.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .semantic_encoder import (
    CLUSTER_WORDS,
    SemanticCluster,
    WARM_WORDS,
    COLD_WORDS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
)

# Fixed dimensionality (locked — no sprouting axes).
METRIC_DIMS = 11          # dims 0-10 used for distance/gravity
CORE_DIMS = 12            # incl. derived resonance (dim 11)
_GRAVITY_BETA = 7.0       # RBF sharpness for gravity(): exp(-β·d²)
_QUANTIZE_LEVELS = 12     # buckets per axis for the positional hash

# Ordered cluster axes → dims 0-5
_CLUSTER_ORDER = [
    SemanticCluster.ALPHA,
    SemanticCluster.BETA,
    SemanticCluster.GAMMA,
    SemanticCluster.DELTA,
    SemanticCluster.EPSILON,
    SemanticCluster.ZETA,
]

_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class HashSphereCore:
    """A memory's position in the 12-D semantic manifold."""
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    epsilon: float = 0.0
    zeta: float = 0.0
    temperature: float = 0.5
    polarity: float = 0.5
    spin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # intensity, complexity, abstraction
    resonance: float = 0.0                              # derived, bounded [-1,1]-ish
    energy: float = 0.0                                 # ±resonance (signed sentiment strength)

    def metric_vector(self) -> List[float]:
        """The 11-D vector used for distance/gravity (excludes derived resonance)."""
        return [
            self.alpha, self.beta, self.gamma, self.delta, self.epsilon, self.zeta,
            self.temperature, self.polarity,
            self.spin[0], self.spin[1], self.spin[2],
        ]

    def to_dict(self) -> Dict:
        return {
            "version": "v-true-1",
            "clusters": {
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "delta": self.delta, "epsilon": self.epsilon, "zeta": self.zeta,
            },
            "temperature": self.temperature,
            "polarity": self.polarity,
            "spin": {"intensity": self.spin[0], "complexity": self.spin[1], "abstraction": self.spin[2]},
            "resonance": self.resonance,
            "energy": self.energy,
            "metric_vector": self.metric_vector(),
        }

    @property
    def dominant_cluster(self) -> str:
        vals = {
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "delta": self.delta, "epsilon": self.epsilon, "zeta": self.zeta,
        }
        return max(vals, key=vals.get)

    def hash(self) -> str:
        """Quantized position → deterministic id. Similar cores → nearby ids."""
        buckets = [
            min(_QUANTIZE_LEVELS - 1, max(0, int(round(v * (_QUANTIZE_LEVELS - 1)))))
            for v in self.metric_vector()
        ]
        # resonance folded in as a coarse bucket so identity reflects phase too
        res_bucket = min(7, max(0, int(round((self.resonance + 1.0) / 2.0 * 7))))
        raw = "-".join(str(b) for b in buckets) + f"-r{res_bucket}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return f"hs1_{digest}"


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _cluster_distribution(tokens: List[str]) -> Dict[SemanticCluster, float]:
    """Normalized word-cluster probabilities over α…ζ (sums to 1)."""
    counts = {c: 0.0 for c in _CLUSTER_ORDER}
    hits = 0
    for tok in tokens:
        for cluster in _CLUSTER_ORDER:
            if tok in CLUSTER_WORDS[cluster]:
                counts[cluster] += 1.0
                hits += 1
                break
    if hits == 0:
        # No seed-vocabulary hits → uniform prior (Wave 2's model will fix coverage).
        return {c: 1.0 / len(_CLUSTER_ORDER) for c in _CLUSTER_ORDER}
    return {c: counts[c] / hits for c in _CLUSTER_ORDER}


def _temperature(tokens: List[str], raw_text: str) -> float:
    warm = sum(1 for t in tokens if t in WARM_WORDS)
    cold = sum(1 for t in tokens if t in COLD_WORDS)
    punct = raw_text.count("!") + raw_text.count("?")
    score = 0.5 + 0.12 * (warm - cold) + 0.05 * punct
    return max(0.0, min(1.0, score))


def _polarity(tokens: List[str]) -> float:
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    score = 0.5 + 0.12 * (pos - neg)
    return max(0.0, min(1.0, score))


def _spin(tokens: List[str], raw_text: str, dist: Dict[SemanticCluster, float]) -> Tuple[float, float, float]:
    """Spin = (intensity, complexity, abstraction) per the Spin↔Energy swap.

    intensity  — urgency signal: caps ratio + exclamation density + warm words
    complexity — lexical diversity (unique/total) scaled by length
    abstraction— abstract/quality mass (γ+ε) vs concrete mass (α+β)
    """
    n = max(1, len(tokens))
    letters = [ch for ch in raw_text if ch.isalpha()]
    caps_ratio = (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0
    excl = (raw_text.count("!") / n)
    warm = sum(1 for t in tokens if t in WARM_WORDS) / n
    intensity = max(0.0, min(1.0, 0.5 * caps_ratio + 0.3 * min(1.0, excl * 5) + 0.6 * min(1.0, warm * 5)))

    unique_ratio = len(set(tokens)) / n
    length_factor = min(1.0, n / 40.0)
    complexity = max(0.0, min(1.0, 0.6 * unique_ratio + 0.4 * length_factor))

    concrete = dist[SemanticCluster.ALPHA] + dist[SemanticCluster.BETA]
    abstract = dist[SemanticCluster.GAMMA] + dist[SemanticCluster.EPSILON]
    abstraction = max(0.0, min(1.0, 0.5 + 0.5 * (abstract - concrete)))
    return (intensity, complexity, abstraction)


def _resonance(dist: Dict[SemanticCluster, float], temperature: float,
               polarity: float, spin: Tuple[float, float, float]) -> float:
    """Derived harmonic signature over the semantic core (dims 0-10).

    Classic form R = sin(a·x)+cos(b·y)+tan(c·z) has an unbounded tan term, so we
    replace it with a bounded third harmonic. Range ~[-1, 1] after /3 scaling.
    NOTE: this is a SIGNATURE, not a distance axis.
    """
    a, b, c = math.pi / 4.0, math.e / 3.0, 1.618 / 2.0
    # collapse the semantic mass to three phase inputs
    x = dist[SemanticCluster.ALPHA] + dist[SemanticCluster.DELTA]   # agents/actions
    y = temperature * 0.5 + polarity * 0.5                          # affect
    z = spin[1] * 0.5 + spin[2] * 0.5                               # complexity/abstraction
    r = math.sin(a * x * math.pi) + math.cos(b * y * math.pi) + math.sin(c * z * math.pi * 2)
    return max(-1.0, min(1.0, r / 3.0))


def encode_core(text: str, embedding: Optional[List[float]] = None,
                axes: Optional[dict] = None) -> HashSphereCore:
    """Compute the 12-D semantic core for a piece of text.

    Wave 2 axis priority for α…ζ / temperature / polarity:
      1. `axes` — precomputed WORD-LEVEL classification (the robust, faithful path;
         async callers pass hash_sphere_model.axes_for_text()).
      2. sentence-embedding prototype similarity (model.predict) — coarser fallback.
      3. Wave-1 seed-dictionary word counting (works untrained).
    Spin is always text-statistic based (intensity/complexity/abstraction).
    """
    tokens = _tokenize(text)

    model_axes = axes
    if model_axes is None:
        try:
            from .hash_sphere_model import hash_sphere_model
            model_axes = hash_sphere_model.predict(embedding)
        except Exception:
            model_axes = None

    if model_axes:
        c = model_axes["clusters"]
        dist = {
            SemanticCluster.ALPHA: c["alpha"], SemanticCluster.BETA: c["beta"],
            SemanticCluster.GAMMA: c["gamma"], SemanticCluster.DELTA: c["delta"],
            SemanticCluster.EPSILON: c["epsilon"], SemanticCluster.ZETA: c["zeta"],
        }
        temperature = model_axes["temperature"]
        polarity = model_axes["polarity"]
    else:
        dist = _cluster_distribution(tokens)
        temperature = _temperature(tokens, text or "")
        polarity = _polarity(tokens)

    spin = _spin(tokens, text or "", dist)
    resonance = _resonance(dist, temperature, polarity, spin)
    # Energy = ±resonance (signed sentiment strength): polarity centered at 0.5
    energy = (polarity - 0.5) * 2.0 * (0.5 + 0.5 * abs(resonance))

    return HashSphereCore(
        alpha=dist[SemanticCluster.ALPHA],
        beta=dist[SemanticCluster.BETA],
        gamma=dist[SemanticCluster.GAMMA],
        delta=dist[SemanticCluster.DELTA],
        epsilon=dist[SemanticCluster.EPSILON],
        zeta=dist[SemanticCluster.ZETA],
        temperature=temperature,
        polarity=polarity,
        spin=spin,
        resonance=resonance,
        energy=energy,
    )


def gravity(a_metric: List[float], b_metric: List[float], beta: float = _GRAVITY_BETA) -> float:
    """RBF gravity between two 11-D metric cores: exp(-β·||a-b||²). Range (0,1].

    1.0 = identical position; decays with semantic distance. This is the primary
    hash-sphere ranking signal.
    """
    if not a_metric or not b_metric:
        return 0.0
    n = min(len(a_metric), len(b_metric))
    dist_sq = sum((a_metric[i] - b_metric[i]) ** 2 for i in range(n))
    return math.exp(-beta * dist_sq)


def core_from_stored(coords: Optional[Dict]) -> Optional[List[float]]:
    """Extract the 11-D metric vector from a stored hash_sphere_coords dict."""
    if not coords or not isinstance(coords, dict):
        return None
    mv = coords.get("metric_vector")
    if isinstance(mv, list) and len(mv) >= METRIC_DIMS:
        return [float(x) for x in mv[:METRIC_DIMS]]
    return None
