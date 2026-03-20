"""
Temporal Memory Layer (TML) Service
Enables AI to understand time: "last week", "yesterday", "earlier today", etc.

PATCH #21: Your AI understands human time expressions and retrieves memories by temporal queries.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from uuid import UUID

from sqlmodel import Session, select


TIME_PATTERNS = {
    "yesterday": 1,
    "last week": 7,
    "a week ago": 7,
    "last month": 30,
    "a month ago": 30,
    "two months ago": 60,
    "three months ago": 90,
    "earlier today": 0,
    "this morning": 0,
    "last year": 365,
    "a year ago": 365,
    "recently": 7,
    "a few days ago": 3,
    "last few days": 3,
}


def detect_temporal_query(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Detect temporal query from user text.
    
    PATCH #21: Identifies time expressions like "last week", "yesterday", etc.
    
    Args:
        text: User message text
    
    Returns:
        Tuple of (time_key, days_ago) or (None, None) if no temporal query detected
    """
    if not text:
        return None, None
    
    text_lower = text.lower()
    
    for key, days in TIME_PATTERNS.items():
        if key in text_lower:
            return key, days
    
    return None, None


def extract_temporal_memories(
    session: Session,
    user_id: str,
    org_id: Optional[str],
    text: str
) -> List[Dict]:
    """
    Extract memories based on temporal query.
    
    PATCH #21: Retrieves memories from a specific time period based on user's temporal query.
    
    Args:
        session: Database session
        user_id: User ID
        org_id: Optional organization ID
        text: User message text
    
    Returns:
        List of memory dicts with temporal relevance
    """
    key, days = detect_temporal_query(text)
    
    if not key:
        return []
    
    target_date = datetime.utcnow() - timedelta(days=days)
    
    # Query memories for this user (using RAG engine)
    try:
        from ..services.rag import rag_engine
        # Get all memories for this user
        memories_raw = rag_engine.retrieve_memories(
            session=session,
            user_id=user_id,
            org_id=org_id,
            query="",  # Empty query to get all memories
            top_k=100,  # Get many to filter by time
            use_embedding=False
        )
        memories = memories_raw
    except Exception:
        memories = []
    
    # Filter by timestamp closeness
    result = []
    for mem in memories:
        mem_time = None
        mem_content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
        
        if not mem_content:
            continue
        
        # Try to get timestamp from metadata
        mem_meta = mem.get("metadata", {}) if isinstance(mem, dict) else getattr(mem, "meta_data", {})
        if mem_meta and isinstance(mem_meta, dict):
            mem_time_str = mem_meta.get("created_at")
            if mem_time_str:
                try:
                    from dateutil import parser as date_parser
                    mem_time = date_parser.parse(mem_time_str)
                except Exception:
                    pass
        
        # Fallback to created_at if available
        if not mem_time:
            if isinstance(mem, dict):
                mem_time = mem.get("created_at")
            else:
                mem_time = getattr(mem, 'created_at', None)
        
        if not mem_time:
            continue
        
        # Calculate time difference
        if isinstance(mem_time, str):
            try:
                from dateutil import parser as date_parser
                mem_time = date_parser.parse(mem_time)
            except Exception:
                continue
        
        delta = abs((mem_time - target_date).days)
        
        # Include memories within 3 days of target (for "yesterday", "last week", etc.)
        if delta <= 3:
            result.append({
                "content": mem_content,
                "delta_days": delta,
                "timestamp": mem_time.isoformat() if hasattr(mem_time, 'isoformat') else str(mem_time),
                "type": "temporal"
            })
    
    # Sort by closeness (closer = better)
    result.sort(key=lambda x: x["delta_days"])
    
    return result[:5]

