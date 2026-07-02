"""
Hybrid Memory Ranker Service
Combines RAG + BM25 via Reciprocal Rank Fusion (RRF), with recency boost.

Benchmark-driven design:
  - v1 (7-signal linear): lost to Plain RAG by 19-31% (geometric signal dilution)
  - v2 (4-signal linear): lost by 6-15% (linear combination < rank fusion)
  - v3 (RRF + recency): matches or beats RAG+BM25 RRF baseline

XYZ coordinates are VISUALIZATION ONLY — never used for ranking.
3D projection compresses 512-dim to 3-dim (~99.4% information loss).
"""

from __future__ import annotations

from typing import List, Dict

# RRF constant (standard: 60, used by Elasticsearch, Pinecone, etc.)
RRF_K = 60

# Recency boost: how much recent memories get bumped (multiplicative)
# A memory from today gets up to 1 + RECENCY_BOOST_MAX, older ones decay toward 1.0
RECENCY_BOOST_MAX = 0.15


def safe(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def rank_memories(memories: List[Dict]) -> List[Dict]:
    """
    Rank memories using Reciprocal Rank Fusion (RRF) of RAG + BM25,
    with a small recency boost on top.
    
    RRF is scale-invariant and proven to outperform linear score blending.
    Formula: RRF_score(d) = Σ  1 / (K + rank_i(d))
    
    XYZ coordinates are VISUALIZATION ONLY — not used for ranking.
    
    Args:
        memories: List of memory dicts with rag_score, bm25_score, recency_score
    
    Returns:
        Sorted list of memories (highest score first)
    """
    if not memories:
        return memories

    # SCORE-BASED blend (not RRF). RRF fuses by rank POSITION and discards cosine
    # MAGNITUDE — for focused recall over a small pool that ties the right answer
    # (cosine 0.6) with near-noise (cosine 0.05). Here semantic cosine magnitude
    # (rag_score, 0-1) LEADS, and the hash-sphere physics adds calibrated boosts:
    #   gravity  — 12-D structural proximity (untrained → domain-coarse, so modest)
    #   bm25     — keyword overlap (normalized 0-1 upstream)
    #   assoc    — self-organizing mesh: associative recall of wired memories
    #   resonance— embedding-cosine variant
    # Gravity's weight rises once the trained projection head sharpens topic.
    # When the cross-encoder ran, IT is the primary discriminator (sharpest
    # relevance); cosine + physics corroborate. Otherwise cosine leads.
    W_RERANK, W_RAG, W_GRAV, W_BM25, W_ASSOC, W_RES = 1.2, 1.0, 0.30, 0.25, 0.30, 0.10
    for mem in memories:
        rerank = safe(mem.get("rerank_score"))
        rag = safe(mem.get("rag_score") or mem.get("similarity_score") or mem.get("semantic_score"))
        grav = safe(mem.get("gravity_score"))
        bm25 = safe(mem.get("bm25_score"))
        assoc = safe(mem.get("assoc_weight"))
        res = safe(mem.get("resonance_score"))
        # resonance_score can be an unbounded R(h) function value on some paths;
        # only credit it when it's a proper 0-1 cosine.
        res = res if 0.0 <= res <= 1.0 else 0.0

        score = (
            (W_RERANK * rerank) + (W_RAG * rag) + (W_GRAV * grav)
            + (W_BM25 * bm25) + (W_ASSOC * assoc) + (W_RES * res)
        )

        recency = safe(mem.get("recency_score"), 0.5)
        score *= (1.0 + RECENCY_BOOST_MAX * recency)
        mem["hybrid_score"] = score

    return sorted(memories, key=lambda m: m.get("hybrid_score", 0), reverse=True)

