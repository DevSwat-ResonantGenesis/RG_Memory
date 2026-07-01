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
    
    # Step 0: Rank by HASH SPHERE gravity (12-D) — the PRIMARY signal (RFC-0002).
    # Weighted highest in the fusion; cosine/BM25 are the recall floor.
    grav_ranked = sorted(memories, key=lambda m: safe(m.get("gravity_score")), reverse=True)
    grav_rank = {id(m): rank for rank, m in enumerate(grav_ranked)}

    # Step 1: Rank by RAG score (descending)
    rag_ranked = sorted(memories, key=lambda m: safe(m.get("rag_score") or m.get("similarity_score") or m.get("semantic_score")), reverse=True)
    rag_rank = {id(m): rank for rank, m in enumerate(rag_ranked)}

    # Step 2: Rank by BM25 score (descending)
    bm25_ranked = sorted(memories, key=lambda m: safe(m.get("bm25_score")), reverse=True)
    bm25_rank = {id(m): rank for rank, m in enumerate(bm25_ranked)}

    # Step 3: Rank by resonance/embedding cosine (descending)
    res_ranked = sorted(memories, key=lambda m: safe(m.get("resonance_score")), reverse=True)
    res_rank = {id(m): rank for rank, m in enumerate(res_ranked)}

    # Step 4: RRF fusion — gravity (weight 2.0, primary) + RAG + BM25 + resonance
    GRAVITY_WEIGHT = 2.0
    for mem in memories:
        mid = id(mem)
        rrf_score = (
            GRAVITY_WEIGHT * (1.0 / (RRF_K + grav_rank.get(mid, len(memories)))) +
            1.0 / (RRF_K + rag_rank.get(mid, len(memories))) +
            1.0 / (RRF_K + bm25_rank.get(mid, len(memories))) +
            1.0 / (RRF_K + res_rank.get(mid, len(memories)))
        )
        
        # Step 5: Recency boost (multiplicative, not additive)
        recency = safe(mem.get("recency_score"), 0.5)
        boost = 1.0 + (RECENCY_BOOST_MAX * recency)
        
        mem["hybrid_score"] = rrf_score * boost
    
    return sorted(memories, key=lambda m: m.get("hybrid_score", 0), reverse=True)

