"""
Temporal Memory Layer (TML) Service
Enables AI to understand time: "last week", "yesterday", "earlier today", etc.

Detects temporal expressions in queries and filters memories by created_at timestamp.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


TIME_PATTERNS = {
    "yesterday": (1, 1),
    "last week": (7, 3),
    "a week ago": (7, 3),
    "last month": (30, 7),
    "a month ago": (30, 7),
    "two months ago": (60, 7),
    "three months ago": (90, 7),
    "earlier today": (0, 0),
    "this morning": (0, 0),
    "last year": (365, 30),
    "a year ago": (365, 30),
    "recently": (3, 3),
    "a few days ago": (3, 2),
    "last few days": (3, 3),
}


def detect_temporal_query(text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Detect temporal expression in user text.
    
    Returns:
        Tuple of (time_key, days_ago, window_days) or (None, None, None)
    """
    if not text:
        return None, None, None
    
    text_lower = text.lower()
    
    for key, (days, window) in TIME_PATTERNS.items():
        if key in text_lower:
            return key, days, window
    
    return None, None, None


async def extract_temporal_memories(
    session: AsyncSession,
    user_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Retrieve memories from a time window matching the user's temporal query.
    
    Uses direct DB query on created_at — no rag_engine dependency.
    
    Args:
        session: Async database session
        user_id: User UUID string
        query: User message text
        limit: Max results
    
    Returns:
        List of memory dicts with temporal scores, or [] if no temporal expression found
    """
    from ..models import MemoryRecord
    from .memory_encryption import decrypt_memory_content
    import uuid as _uuid

    key, days_ago, window = detect_temporal_query(query)
    if key is None:
        return []

    now = datetime.now(timezone.utc)
    if days_ago == 0:
        range_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        range_end = now
    else:
        target = now - timedelta(days=days_ago)
        range_start = target - timedelta(days=window)
        range_end = target + timedelta(days=window)

    try:
        user_uuid = _uuid.UUID(user_id)
    except Exception:
        return []

    try:
        stmt = (
            select(MemoryRecord)
            .where(
                MemoryRecord.user_id == user_uuid,
                MemoryRecord.created_at >= range_start,
                MemoryRecord.created_at <= range_end,
            )
            .order_by(MemoryRecord.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
    except Exception as e:
        logger.warning(f"Temporal query failed: {e}")
        return []

    memories = []
    for rec in records:
        content = decrypt_memory_content(rec.content)
        if not content or len(content) < 10 or content.startswith("ENC2:"):
            continue
        memories.append({
            "id": str(rec.id),
            "content": content,
            "type": "temporal",
            "timestamp": rec.created_at.isoformat() if rec.created_at else None,
            "temporal_key": key,
        })

    logger.debug(f"Temporal search '{key}' ({range_start.date()} → {range_end.date()}) returned {len(memories)} memories")
    return memories

