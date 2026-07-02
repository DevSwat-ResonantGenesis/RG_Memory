"""
Resonant Memory — Python SDK
============================

The world's first physics-informed, immutable, sovereign AI memory as an API.

Retrieval combines a 12-D hash-sphere semantic manifold (gravity ranking),
emergent anchors, a self-organizing associative mesh, a cross-encoder precision
reranker, Mem0-style fact injection, and a multi-hop knowledge graph — with
pgvector cosine + BM25 as the recall floor. Every memory is immutable and
anchored on-chain, isolated per user / agent / org (blockchain-block model).

Usage:
    from resonant_memory import ResonantMemory

    mem = ResonantMemory(api_key="rg_live_...", user_id="u123")
    mem.ingest("My name is Marcus and I lead the payments team")
    hits = mem.recall("what does the user do")
    for h in hits:
        print(h["content"], h["confidence"])

Billing: each call deducts credits from your account (ingest 120, recall 60,
fact/graph reads 20). Buy/top-up credits in the dashboard; calls are rejected
with 402 when the balance is exhausted.
"""
from .client import ResonantMemory, ResonantMemoryError, InsufficientCreditsError

__all__ = ["ResonantMemory", "ResonantMemoryError", "InsufficientCreditsError"]
__version__ = "1.0.0"
