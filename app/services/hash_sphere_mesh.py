"""
Hash Sphere Mesh — self-organizing associative memory (RFC-0002 Wave 3c)
=========================================================================

"Fire together, wire together." When memories are retrieved together they
reinforce the edge between them; edges decay with age; strongly-connected
memories can be pulled into recall associatively even when direct 12-D gravity
and cosine miss them. This is the V0.1 SelfOrganizingMemoryMesh, on the 12-D space.

- reinforce_coretrieval(): strengthen edges among a co-retrieved set (weight
  grows toward 1.0, count++). Runs best-effort after each retrieval.
- associative_neighbors(): given seed memory ids, return strongly-connected
  neighbor ids (with age-decayed weight) for associative recall.

Decay is applied LAZILY at read time from last_reinforced age, so no scheduler
is required: effective_weight = weight · 0.97^(days_since_reinforced).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import MemoryEdge

logger = logging.getLogger(__name__)

REINFORCE_STEP = 0.15      # edge weight moves this fraction toward 1.0 per co-retrieval
DECAY_PER_DAY = 0.97       # lazy multiplicative decay
NEIGHBOR_MIN_WEIGHT = 0.2  # associative recall ignores edges weaker than this


def _ordered(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _decayed(weight: float, last_reinforced: Optional[datetime]) -> float:
    if not last_reinforced:
        return weight
    try:
        now = datetime.now(last_reinforced.tzinfo or timezone.utc)
        days = max(0.0, (now - last_reinforced).total_seconds() / 86400.0)
    except Exception:
        return weight
    return weight * (DECAY_PER_DAY ** days)


async def reinforce_coretrieval(
    session: AsyncSession,
    *,
    user_uuid,
    org_uuid,
    agent_hash: Optional[str],
    memory_ids: List[str],
) -> int:
    """Strengthen edges among co-retrieved memories. Returns edges touched."""
    if not user_uuid or not memory_ids or len(memory_ids) < 2:
        return 0
    # de-dup + valid uuids, cap the clique size to bound work
    ids = []
    for m in memory_ids[:8]:
        try:
            ids.append(str(uuid.UUID(m)))
        except (ValueError, TypeError):
            continue
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        return 0

    touched = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s, d = _ordered(ids[i], ids[j])
            row = await session.execute(
                select(MemoryEdge).where(
                    MemoryEdge.user_id == user_uuid,
                    MemoryEdge.src_id == uuid.UUID(s),
                    MemoryEdge.dst_id == uuid.UUID(d),
                ).limit(1)
            )
            edge = row.scalar_one_or_none()
            if edge is None:
                session.add(MemoryEdge(
                    user_id=user_uuid, org_id=org_uuid, agent_hash=agent_hash,
                    src_id=uuid.UUID(s), dst_id=uuid.UUID(d),
                    weight=REINFORCE_STEP, coretrieval_count=1,
                ))
            else:
                base = _decayed(edge.weight or 0.0, edge.last_reinforced)
                edge.weight = min(1.0, base + REINFORCE_STEP * (1.0 - base))
                edge.coretrieval_count = (edge.coretrieval_count or 0) + 1
                edge.last_reinforced = datetime.now(timezone.utc)
            touched += 1
    await session.commit()
    return touched


async def associative_neighbors(
    session: AsyncSession,
    *,
    user_uuid,
    seed_ids: List[str],
    exclude_ids: Optional[set] = None,
    limit: int = 5,
) -> List[Tuple[str, float]]:
    """Return (memory_id, decayed_weight) neighbors strongly linked to the seeds,
    excluding seeds/exclude_ids, best first."""
    if not user_uuid or not seed_ids:
        return []
    seeds = []
    for s in seed_ids[:5]:
        try:
            seeds.append(uuid.UUID(str(s)))
        except (ValueError, TypeError):
            continue
    if not seeds:
        return []
    exclude = set(exclude_ids or set()) | {str(s) for s in seeds}

    rows = await session.execute(
        select(MemoryEdge).where(
            MemoryEdge.user_id == user_uuid,
            (MemoryEdge.src_id.in_(seeds)) | (MemoryEdge.dst_id.in_(seeds)),
        ).limit(200)
    )
    best: Dict[str, float] = {}
    for edge in rows.scalars().all():
        w = _decayed(edge.weight or 0.0, edge.last_reinforced)
        if w < NEIGHBOR_MIN_WEIGHT:
            continue
        for nid in (str(edge.src_id), str(edge.dst_id)):
            if nid in exclude:
                continue
            if w > best.get(nid, 0.0):
                best[nid] = w
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:limit]
