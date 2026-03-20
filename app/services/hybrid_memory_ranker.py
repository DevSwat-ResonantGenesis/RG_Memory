"""
Hybrid Memory Ranker Service
Combines RAG, Hash Sphere, and History signals into one final ranking score.

PATCH #11: Multi-score hybrid AI memory selector, similar to DeepMind / Anthropic agent memory layers.
"""

from __future__ import annotations

from typing import List, Dict

# Weight configuration for hybrid scoring
# Updated to include Layer 4 (Anchor Energy) and Layer 5 (Resonance Function)
W_RAG = 0.30
W_RESONANCE = 0.25
W_PROXIMITY = 0.10
W_RECENCY = 0.05
W_ANCHOR = 0.05
W_RESONANCE_FUNCTION = 0.15  # Layer 5: R(h) = sin(a·x) + cos(b·y) + tan(c·z)
W_ANCHOR_ENERGY = 0.10       # Layer 4: E_j(s) = exp(-β·||s - A_j||²)


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
    Compute final hybrid score combining:
    - RAG semantic score
    - Hash Sphere resonance (hash similarity)
    - XYZ proximity score
    - Recency (timestamp-based)
    - Anchor importance
    - Resonance function score (Layer 5: R(h) = sin(a·x) + cos(b·y) + tan(c·z))
    - Anchor energy (Layer 4: E_j(s) = exp(-β·||s - A_j||²))
    
    PATCH #11: This is how Anthropic agent memory, Google's ReAct-RAG, 
    and DeepMind's TREE memory scoring work internally.
    
    WEEK 2 UPDATE: Added Layer 4 (Anchor Energy) and Layer 5 (Resonance Function)
    from Hash Sphere foundational architecture.
    
    Args:
        mem: Memory dict with scoring fields
    
    Returns:
        Final hybrid score (0-1)
    """
    rag_score = safe(mem.get("rag_score") or mem.get("similarity_score") or mem.get("semantic_score"))  # 0–1
    resonance_score = safe(mem.get("resonance_score"))  # 0–1
    proximity_score = safe(mem.get("proximity_score"))  # 0–1
    recency_score = safe(mem.get("recency_score"))  # 0–1
    anchor_score = safe(mem.get("anchor_score") or mem.get("importance_score"))  # 0–1
    
    # Layer 5: Resonance Function - R(h) = sin(a·x) + cos(b·y) + tan(c·z)
    resonance_function_score = safe(mem.get("resonance_function_score"))  # 0–1
    
    # Layer 4: Anchor Energy - E_j(s) = exp(-β·||s - A_j||²)
    anchor_energy = safe(mem.get("anchor_energy"))  # 0–1
    
    final = (
        rag_score * W_RAG +
        resonance_score * W_RESONANCE +
        proximity_score * W_PROXIMITY +
        recency_score * W_RECENCY +
        anchor_score * W_ANCHOR +
        resonance_function_score * W_RESONANCE_FUNCTION +
        anchor_energy * W_ANCHOR_ENERGY
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

