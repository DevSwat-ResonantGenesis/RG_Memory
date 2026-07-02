"""
Hash Sphere Projection Head (RFC-0002 Wave 2.5) — GATED, off by default
=======================================================================

The Wave-2 model maps a 384-d MiniLM embedding to the 12-D core via *fixed*
seed-centroid cosine + softmax (see hash_sphere_model.predict). That mapping is
stable but not learned — it can't capture axis structure the seed words miss.

This module adds an OPTIONAL trained projection head:

    embedding(384)  --W,b-->  cluster logits(6) + temp logit + polarity logit
    clusters = softmax(logits) ; temperature = σ(t) ; polarity = σ(p)

so it is a drop-in replacement for predict()'s output contract
({"clusters": {α…ζ}, "temperature", "polarity"}).

SAFETY / NO-REGRESSION CONTRACT
-------------------------------
- Disabled unless BOTH: env MEMORY_PROJECTION_HEAD=1  AND  weights load cleanly.
- Weights live in an .npz at MEMORY_PROJECTION_HEAD_WEIGHTS (default under
  data/models/). If the file is missing/corrupt, we return None and the caller
  falls back to the fixed Wave-2 mapping — the live path is byte-for-byte
  unchanged when the flag is off or weights are absent.
- Inference is pure NumPy (no torch at serve time). Training (see
  benchmark/train_projection_head.py) writes the .npz; promotion is gated on a
  LOCOMO non-regression check, so bad weights never reach production.

Weight file schema (np.savez):
    W  : float32 [8, 384]   # rows 0..5 = α…ζ, row 6 = temperature, row 7 = polarity
    b  : float32 [8]
    meta (optional): saved separately; not required at inference.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CLUSTER_KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "models" / "projection_head.npz"


def _enabled() -> bool:
    return os.getenv("MEMORY_PROJECTION_HEAD", "0").strip().lower() in ("1", "true", "yes", "on")


class ProjectionHead:
    """Trained 384→(6+2) linear projection. Loads lazily; safe no-op when off."""

    def __init__(self) -> None:
        self._W = None
        self._b = None
        self._np = None
        self._tried = False
        self._active = False

    def _ensure_loaded(self) -> bool:
        if self._tried:
            return self._active
        self._tried = True
        if not _enabled():
            return False
        try:
            import numpy as np  # local import: numpy ships with sentence-transformers
            path = Path(os.getenv("MEMORY_PROJECTION_HEAD_WEIGHTS", str(_DEFAULT_WEIGHTS)))
            if not path.exists():
                logger.info("ProjectionHead: enabled but no weights at %s — using fixed mapping", path)
                return False
            data = np.load(str(path))
            W, b = data["W"].astype("float32"), data["b"].astype("float32")
            if W.shape[0] != 8 or b.shape[0] != 8:
                logger.warning("ProjectionHead: bad weight shape %s / %s — disabled", W.shape, b.shape)
                return False
            self._np, self._W, self._b, self._active = np, W, b, True
            logger.info("ProjectionHead: ACTIVE (weights %s, in-dim %d)", path.name, W.shape[1])
            return True
        except Exception as e:  # never break ingest on a bad artifact
            logger.warning("ProjectionHead: load failed (%s) — using fixed mapping", e)
            return False

    @property
    def active(self) -> bool:
        return self._ensure_loaded()

    def predict(self, embedding: Optional[List[float]]) -> Optional[Dict]:
        """embedding → {clusters, temperature, polarity}; None → caller falls back."""
        if not embedding or not self._ensure_loaded():
            return None
        try:
            np = self._np
            e = np.asarray(embedding, dtype="float32")
            n = float(np.linalg.norm(e))
            if n > 0:
                e = e / n
            if e.shape[0] != self._W.shape[1]:
                return None  # dim mismatch → safe fallback
            z = self._W @ e + self._b            # [8]
            logits = z[:6]
            logits = logits - float(logits.max())
            exps = np.exp(logits)
            probs = exps / (float(exps.sum()) or 1.0)
            clusters = {k: float(probs[i]) for i, k in enumerate(_CLUSTER_KEYS)}
            temperature = 1.0 / (1.0 + math.exp(-float(z[6])))
            polarity = 1.0 / (1.0 + math.exp(-float(z[7])))
            return {"clusters": clusters, "temperature": temperature, "polarity": polarity}
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("ProjectionHead.predict failed: %s", e)
            return None


projection_head = ProjectionHead()
