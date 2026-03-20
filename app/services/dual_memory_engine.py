"""
Dual-Layer Long-Term Memory (DLLM)
===================================

Patch #44: Introduces two layers of long-term memory:

Layer 1 — Episodic Memory (short-term, with decay)
- Everything from chat history
- Stored with timestamps
- Slowly decays over time
- Recent = high weight
- Old = low weight

Layer 2 — Semantic Memory (long-term crystallized facts)
- Summaries of repeated facts
- Stable personality knowledge
- Conclusions, rules, patterns
- Never decays
- Updated using reinforcement logic

This patch mirrors hippocampus + neocortex.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DualMemoryEngine:
    """
    Dual-Layer Long-Term Memory Engine
    
    Combines episodic (decaying) and semantic (stable) memory layers
    to create a brain-like memory system.
    """
    
    def __init__(self, episodic_decay_days: int = 30):
        """
        Initialize the Dual Memory Engine.
        
        Args:
            episodic_decay_days: Half-life for episodic memory decay (default: 30 days)
        """
        self.EPISODIC_DECAY_DAYS = episodic_decay_days
    
    def compute_decay(self, timestamp: datetime) -> float:
        """
        Reduce importance of episodic memory as it gets older.
        
        Uses exponential decay model:
        - Day 0: weight = 1.0
        - Day 30: weight = 0.0 (half-life)
        - Minimum weight: 0.1 (never fully decays)
        
        Args:
            timestamp: When the memory was created
        
        Returns:
            Decay weight (0.1 to 1.0)
        """
        if not timestamp:
            return 0.5  # Default weight if no timestamp
        
        try:
            days_old = (datetime.utcnow() - timestamp).days
            # Linear decay: 1.0 at day 0, 0.0 at day 30, minimum 0.1
            decay = max(0.1, 1.0 - (days_old / self.EPISODIC_DECAY_DAYS))
            return decay
        except Exception as e:
            logger.warning(f"Error computing decay: {e}")
            return 0.5  # Default weight on error
    
    def build_dual_memory(
        self,
        episodic_msgs: List[Any],
        semantic_anchors: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Combine episodic chat messages + long-term anchors.
        
        Args:
            episodic_msgs: List of chat messages (with timestamp attribute)
            semantic_anchors: List of memory anchors (with anchor_text and importance_score)
        
        Returns:
            Combined and sorted memory list (top 10 by weight)
        """
        episodic_context = []
        
        # Process episodic memories (decaying)
        for msg in episodic_msgs:
            # Extract timestamp (handle different message formats)
            timestamp = None
            if hasattr(msg, 'created_at'):
                timestamp = msg.created_at
            elif hasattr(msg, 'timestamp'):
                timestamp = msg.timestamp
            elif isinstance(msg, dict):
                timestamp_str = msg.get('timestamp') or msg.get('created_at')
                if timestamp_str:
                    try:
                        if isinstance(timestamp_str, str):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = timestamp_str
                    except Exception:
                        timestamp = None
            
            # Extract content
            content = None
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content') or msg.get('text')
            
            if content:
                decay = self.compute_decay(timestamp) if timestamp else 0.5
                episodic_context.append({
                    "content": content,
                    "weight": decay,
                    "type": "episodic",
                    "timestamp": timestamp.isoformat() if timestamp else None
                })
        
        # Process semantic memories (stable, never decay)
        semantic_context = []
        for anchor in semantic_anchors:
            # Extract anchor text
            anchor_text = None
            if hasattr(anchor, 'anchor_text'):
                anchor_text = anchor.anchor_text
            elif hasattr(anchor, 'context'):
                anchor_text = anchor.context
            elif isinstance(anchor, dict):
                anchor_text = anchor.get('anchor_text') or anchor.get('content') or anchor.get('context')
            
            # Extract importance score
            importance_score = 0.5
            if hasattr(anchor, 'importance_score'):
                importance_score = anchor.importance_score or 0.5
            elif isinstance(anchor, dict):
                importance_score = anchor.get('importance_score') or anchor.get('score') or 0.5
            
            if anchor_text:
                # Semantic memories get 1.2x boost (they're more stable)
                semantic_context.append({
                    "content": anchor_text,
                    "weight": importance_score * 1.2,  # Semantic > episodic
                    "type": "semantic",
                    "importance_score": importance_score
                })
        
        # Merge and sort by weight
        combined = episodic_context + semantic_context
        combined = sorted(combined, key=lambda x: x["weight"], reverse=True)
        
        # Return top 10 memories
        return combined[:10]
    
    def get_memory_summary(self, memories: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable summary of dual-layer memories.
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            Formatted memory summary string
        """
        if not memories:
            return "No memories available."
        
        parts = []
        for i, mem in enumerate(memories, 1):
            mem_type = mem.get("type", "unknown")
            weight = mem.get("weight", 0.0)
            content = mem.get("content", "")[:200]  # First 200 chars
            parts.append(f"{i}. [{mem_type}] (w={weight:.2f}): {content}")
        
        return "\n".join(parts)

