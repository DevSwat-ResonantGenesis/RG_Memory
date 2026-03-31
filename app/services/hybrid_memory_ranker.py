"""
Hybrid Memory Ranker Service
Combines RAG, Hash Sphere, and History signals into one final ranking score.

PATCH #11: Multi-score hybrid AI memory selector, similar to DeepMind / Anthropic agent memory layers.
"""

from __future__ import annotations

from typing import List, Dict

# Weight configuration for hybrid scoring
# Rebalanced after benchmark v2 (conversation-based ground truth):
#   Plain RAG P@5=0.446, BM25 P@5=0.440, old hybrid P@5=0.356
#   Old hybrid lost 19-31% vs baselines because geometric signals (proximity,
#   resonance function, anchor energy) diluted the strong embedding signal.
#   XYZ coordinates are VISUALIZATION ONLY — not used for retrieval ranking.
#   3D projection compresses 512-dim to 3-dim (~99.4% information loss).
W_RAG = 0.55               # pgvector cosine similarity — strongest signal
W_BM25 = 0.20              # BM25 keyword score — competitive with RAG for conversation retrieval
W_RECENCY = 0.15           # Timestamp-based decay — recent memories matter
W_RESONANCE = 0.10         # Embedding cosine resonance (tiebreaker for RAG)


def safe(v, default=0.0):
    """
    Safely convert value to float, returning default if conversion fails.
    
    Args:
        v: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def compute_score(mem: Dict) -> float:
    """
    Compute final hybrid score combining 4 signals:
    - RAG cosine similarity (0.55) — strongest signal
    - BM25 keyword score (0.20) — complements RAG for exact matches
    - Recency decay (0.15) — recent memories preferred
    - Embedding resonance (0.10) — tiebreaker
    
    XYZ coordinates are VISUALIZATION ONLY — not used for ranking.
    Benchmark v2 showed geometric signals (proximity, resonance function,
    anchor energy) diluted retrieval quality by 19-31%.
    
    Args:
        mem: Memory dict with scoring fields
    
    Returns:
        Final hybrid score (0-1)
    """
    rag_score = safe(mem.get("rag_score") or mem.get("similarity_score") or mem.get("semantic_score"))
    bm25_score = safe(mem.get("bm25_score"))
    recency_score = safe(mem.get("recency_score"))
    resonance_score = safe(mem.get("resonance_score"))
    
    final = (
        rag_score * W_RAG +
        bm25_score * W_BM25 +
        recency_score * W_RECENCY +
        resonance_score * W_RESONANCE
    )
    
    return final


def rank_memories(memories: List[Dict]) -> List[Dict]:
    """
    Sort memories based on hybrid score.
    
    PATCH #11: Applies multi-factor scoring to rank memories by relevance.
    
    Args:
        memories: List of memory dicts with scoring fields
    
    Returns:
        Sorted list of memories (highest score first)
    """
    for mem in memories:
        mem["hybrid_score"] = compute_score(mem)
    
    return sorted(memories, key=lambda m: m.get("hybrid_score", 0), reverse=True)

