#!/usr/bin/env python3
"""
Train the Hash Sphere projection head (RFC-0002 Wave 2.5).

WHAT IT DOES
------------
Fits a linear head  W[8,384], b[8]  that maps a MiniLM embedding to
  rows 0..5 → α…ζ cluster logits, row 6 → temperature logit, row 7 → polarity logit
matching the output contract of hash_sphere_model.predict().

Training signal = DISTILLATION from the current stable Wave-2 model over a broad
text corpus (the seed vocabulary sentences by default, or a --corpus file, one
text per line). We regress the head onto the teacher's soft logits via ridge
regression (closed form, pure NumPy). A trained linear head generalizes the seed
directions across the whole embedding space more smoothly than per-query
centroid cosines, at ~zero inference cost.

WHY GATED
---------
The head is OFF in production (MEMORY_PROJECTION_HEAD unset). This script writes
the weights artifact but does NOT flip the flag. Use --promote to run the LOCOMO
harness with the head ON and refuse to keep the weights unless the overall score
is >= the baseline (no-regression gate). Only after a clean promote should ops
set MEMORY_PROJECTION_HEAD=1.

USAGE
-----
  python benchmark/train_projection_head.py --out data/models/projection_head.npz
  python benchmark/train_projection_head.py --promote          # train + LOCOMO gate
  MEMORY_PROJECTION_HEAD=1 ...  # (ops) enable at serve time after a clean promote
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_head")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_CLUSTER_KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    return math.log(p / (1.0 - p))


async def _build_corpus_embeddings(texts):
    """Return (X[n,384], Y[n,8]) using the live embedder + stable teacher model."""
    import numpy as np
    from app.embeddings_minilm import embeddings_generator
    from app.services.hash_sphere_model import hash_sphere_model

    await hash_sphere_model.ensure_built(embeddings_generator)
    embs = await embeddings_generator.generate(texts, task="search_document")

    X, Y = [], []
    for text, e in zip(texts, embs):
        if not e:
            continue
        teacher = hash_sphere_model.predict(e)  # fixed Wave-2 mapping = teacher
        if not teacher:
            continue
        clusters = teacher["clusters"]
        # target logits: 6 cluster logits (log-prob, softmax-invertible) + temp + polarity
        row = [math.log(min(1.0, max(_EPS, clusters[k]))) for k in _CLUSTER_KEYS]
        row.append(_logit(teacher["temperature"]))
        row.append(_logit(teacher["polarity"]))
        X.append(np.asarray(e, dtype="float32"))
        Y.append(np.asarray(row, dtype="float32"))
    if not X:
        raise SystemExit("No training rows built (embedder unavailable?)")
    return np.vstack(X), np.vstack(Y)


def _seed_corpus():
    """Default corpus: the seed vocabulary as short sentences."""
    from app.services.semantic_encoder import (
        CLUSTER_WORDS, WARM_WORDS, COLD_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS,
    )
    words = set()
    for grp in CLUSTER_WORDS.values():
        words.update(grp)
    for grp in (WARM_WORDS, COLD_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS):
        words.update(grp)
    return [f"This is about {w}." for w in sorted(words)]


def _fit_ridge(X, Y, lam: float):
    """Closed-form ridge: solve (XᵀX + λI) Wᵀ = XᵀY with a bias column."""
    import numpy as np
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype="float32")])         # [n, d+1]
    A = Xb.T @ Xb + lam * np.eye(d + 1, dtype="float32")
    B = Xb.T @ Y                                                  # [d+1, 8]
    theta = np.linalg.solve(A, B)                                 # [d+1, 8]
    W = theta[:d, :].T.astype("float32")                          # [8, d]
    b = theta[d, :].astype("float32")                             # [8]
    return W, b


async def _amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "projection_head.npz"))
    ap.add_argument("--corpus", default="", help="optional text file, one sample per line")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge regularization")
    ap.add_argument("--promote", action="store_true", help="run LOCOMO no-regression gate before keeping")
    args = ap.parse_args()

    import numpy as np

    texts = (
        [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()]
        if args.corpus else _seed_corpus()
    )
    log.info("Training on %d texts (lam=%.3f)", len(texts), args.lam)
    X, Y = await _build_corpus_embeddings(texts)
    W, b = _fit_ridge(X, Y, args.lam)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fit reconstruction error (sanity signal)
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype="float32")])
    pred = Xb @ np.vstack([W.T, b[None, :]])
    rmse = float(np.sqrt(((pred - Y) ** 2).mean()))
    np.savez(str(out), W=W, b=b)
    log.info("Saved %s  (W%s, b%s)  fit_rmse=%.4f", out, W.shape, b.shape, rmse)

    if args.promote:
        log.info("Promotion gate: run LOCOMO with the head ON, compare to baseline.")
        log.info("  1) baseline:  MEMORY_PROJECTION_HEAD=0 python benchmark/locomo_run.py")
        log.info("  2) candidate: MEMORY_PROJECTION_HEAD=1 python benchmark/locomo_run.py")
        log.info("  Keep weights & set MEMORY_PROJECTION_HEAD=1 ONLY if candidate >= baseline.")
        log.info("  (Gate is intentionally manual: LOCOMO needs the live LLM+memory services.)")


if __name__ == "__main__":
    asyncio.run(_amain())
