"""
Hash Sphere Anchors — the live gravity field (RFC-0002 Wave 3b)
================================================================

Emergent anchors are the "gravity wells" of the hash sphere. They are NOT stored
dimensions — they are a live field that forms and moves as memories arrive:

  - On ingest, the new memory's 12-D core is compared (gravity) to the user's
    existing emergent anchors.
  - If it falls into an existing well (gravity >= JOIN), it JOINS: the anchor
    DRIFTS toward the memory (A ← A + γ·(m − A)), its member_count and importance
    grow. This is biological consolidation — related memories pull the well.
  - Otherwise a NEW anchor SPAWNS at the memory's position (a fresh well).

Anchors are stored in memory_anchors (anchor_type='emergent'), with the 12-D
core in extra_metadata. They are immutable in the delete sense (is_archived only)
but their POSITION is live (drift) — exactly "identity constant, gravity live".

Retrieval uses them as a secondary pull: candidates that sit in a strong well the
query also falls into get a gravity boost (dense/important regions surface).
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import MemoryAnchor
from .hash_sphere_core import gravity as hs_gravity, HashSphereCore

logger = logging.getLogger(__name__)

JOIN_GRAVITY = 0.55      # memory joins an anchor if gravity to it >= this
DRIFT_GAMMA = 0.25       # anchor drifts this fraction toward each joining memory
BOOST_GRAVITY = 0.45     # query must be at least this close to a well to boost


def _anchor_core(anchor: MemoryAnchor) -> Optional[List[float]]:
    meta = anchor.extra_metadata or {}
    mv = meta.get("metric_vector")
    if isinstance(mv, list) and mv:
        return [float(x) for x in mv]
    return None


async def _active_emergent(session: AsyncSession, user_uuid, agent_hash) -> List[MemoryAnchor]:
    stmt = select(MemoryAnchor).where(
        MemoryAnchor.user_id == user_uuid,
        MemoryAnchor.anchor_type == "emergent",
        MemoryAnchor.is_archived == False,  # noqa: E712
    )
    if agent_hash:
        stmt = stmt.where(MemoryAnchor.agent_hash == agent_hash)
    result = await session.execute(stmt.limit(500))
    return list(result.scalars().all())


async def reinforce(
    session: AsyncSession,
    *,
    user_uuid,
    org_uuid,
    agent_hash: Optional[str],
    memory_id,
    core: HashSphereCore,
) -> str:
    """Join the nearest well (drift it) or spawn a new one. Returns action label."""
    if not user_uuid or not org_uuid:
        return "skipped"
    m = core.metric_vector()
    anchors = await _active_emergent(session, user_uuid, agent_hash)

    best = None
    best_g = 0.0
    for a in anchors:
        ac = _anchor_core(a)
        if ac is None:
            continue
        g = hs_gravity(m, ac)
        if g > best_g:
            best_g, best = g, a

    if best is not None and best_g >= JOIN_GRAVITY:
        # DRIFT the well toward the new memory (live consolidation)
        ac = _anchor_core(best)
        drifted = [ac[i] + DRIFT_GAMMA * (m[i] - ac[i]) for i in range(min(len(ac), len(m)))]
        meta = dict(best.extra_metadata or {})
        members = int(meta.get("member_count", 1)) + 1
        meta["metric_vector"] = drifted
        meta["member_count"] = members
        best.extra_metadata = meta
        best.importance_score = min(1.0, 0.3 + 0.05 * members)
        await session.commit()
        return "joined"

    # SPAWN a new well at this memory's position
    meta = {"metric_vector": m, "member_count": 1, "clusters": core.to_dict()["clusters"]}
    anchor = MemoryAnchor(
        user_id=user_uuid,
        org_id=org_uuid,
        message_id=memory_id,
        anchor_text=f"emergent:{core.dominant_cluster}",
        anchor_hash=core.hash(),
        context="",
        importance_score=0.3,
        anchor_type="emergent",
        agent_hash=agent_hash,
        extra_metadata=meta,
    )
    session.add(anchor)
    await session.commit()
    return "spawned"


async def gravity_boosts(
    session: AsyncSession,
    *,
    user_uuid,
    agent_hash: Optional[str],
    query_core_metric: List[float],
) -> List[Dict]:
    """Return wells the query falls into, as {core, weight}. Used to pull dense
    regions into retrieval. weight scales with query-well gravity × well mass."""
    if not user_uuid:
        return []
    anchors = await _active_emergent(session, user_uuid, agent_hash)
    wells = []
    for a in anchors:
        ac = _anchor_core(a)
        if ac is None:
            continue
        g = hs_gravity(query_core_metric, ac)
        if g >= BOOST_GRAVITY:
            members = int((a.extra_metadata or {}).get("member_count", 1))
            mass = min(1.0, 0.5 + 0.1 * members)
            wells.append({"core": ac, "weight": g * mass})
    return wells
