"""
Cross-Encoder Reranker (relevance precision layer)
==================================================

Bi-encoder cosine (MiniLM embeddings) is fast but coarse — it embeds query and
memory INDEPENDENTLY. A cross-encoder jointly encodes (query, memory) pairs and
scores true relevance, which is markedly sharper (this is what closes the gap
with the frontier labs' rerankers).

Used as a precision re-rank over the top candidate pool: cosine/gravity/BM25/mesh
RECALL candidates cheaply; the cross-encoder then RE-SCORES the top ~N for final
ordering. Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (trained for passage
ranking, ~80MB, CPU-fast). Lazy-loaded once; degrades gracefully to no-op if
unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None
_load_failed = False


def _load():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME, max_length=256)
        logger.info("Cross-encoder reranker loaded: %s", _MODEL_NAME)
    except Exception as e:
        _load_failed = True
        logger.warning("Cross-encoder unavailable (%s) — rerank disabled", e)
    return _model


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _predict_sync(query: str, docs: List[str]) -> List[float]:
    model = _load()
    if model is None:
        return []
    pairs = [[query, d] for d in docs]
    raw = model.predict(pairs)
    return [_sigmoid(float(s)) for s in raw]


async def rerank_scores(query: str, docs: List[str]) -> List[float]:
    """Return a 0-1 relevance score per doc for (query, doc). [] if unavailable.
    Runs the CPU model in a threadpool so it doesn't block the event loop."""
    if not query or not docs:
        return []
    try:
        return await asyncio.to_thread(_predict_sync, query, docs)
    except Exception as e:
        logger.debug("Rerank failed: %s", e)
        return []


def available() -> bool:
    return _load() is not None
