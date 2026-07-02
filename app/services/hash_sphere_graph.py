"""
Hash Sphere Knowledge Graph — multi-hop entity reasoning (best-in-world, GAP 3)
================================================================================

Facts are already a latent knowledge graph: a fact's VALUE can be another fact's
ENTITY. e.g.  (user, child, "Lily")  and  ("Lily", grade, "kindergarten")  chain
through "Lily". Single-shot RAG cannot answer "what grade is my daughter in" — it
needs a 2-hop traversal user→child→Lily→grade→kindergarten. This is what OMEGA /
Hindsight / Zep do and plain vector memory cannot.

At retrieval we take the query-relevant seed facts, then TRAVERSE the fact graph
up to K hops, pulling in connected facts. Combined with the cross-encoder + the
associative mesh, this gives multi-hop reasoning grounded in immutable facts.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import MemoryFact

logger = logging.getLogger(__name__)

MAX_HOPS = 3
MAX_FACTS_PER_HOP = 12


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


async def _facts_for_entities(
    session: AsyncSession, user_uuid, entities: Set[str], exclude_ids: Set[str]
) -> List[MemoryFact]:
    """Active facts whose ENTITY or VALUE matches any of `entities` (case-insensitive)."""
    if not entities:
        return []
    ents = [e for e in entities if e]
    if not ents:
        return []
    stmt = select(MemoryFact).where(
        MemoryFact.user_id == user_uuid,
        MemoryFact.status == "active",
        (func.lower(MemoryFact.entity).in_(ents)) | (func.lower(MemoryFact.value).in_(ents)),
    ).limit(MAX_FACTS_PER_HOP * 4)
    rows = await session.execute(stmt)
    out = []
    for f in rows.scalars().all():
        if str(f.id) in exclude_ids:
            continue
        out.append(f)
    return out


async def traverse(
    session: AsyncSession,
    *,
    user_uuid,
    seed_facts: List[MemoryFact],
    max_hops: int = MAX_HOPS,
) -> List[Tuple[MemoryFact, int]]:
    """BFS over the fact graph from seed facts. Returns (fact, hop_distance) for
    facts reachable within max_hops that were NOT already seeds. hop=1 = directly
    connected to a seed, etc. This is the multi-hop expansion."""
    if not user_uuid or not seed_facts:
        return []

    seen_ids: Set[str] = {str(f.id) for f in seed_facts}
    # frontier entities = the values (and entities) of the seed facts
    frontier: Set[str] = set()
    for f in seed_facts:
        if f.value:
            frontier.add(_norm(f.value))
        if f.entity:
            frontier.add(_norm(f.entity))

    results: List[Tuple[MemoryFact, int]] = []
    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        hop_facts = await _facts_for_entities(session, user_uuid, frontier, seen_ids)
        next_frontier: Set[str] = set()
        added = 0
        for f in hop_facts:
            if str(f.id) in seen_ids:
                continue
            seen_ids.add(str(f.id))
            results.append((f, hop))
            added += 1
            # expand: the new fact's value/entity become next-hop frontier
            if f.value:
                next_frontier.add(_norm(f.value))
            if f.entity:
                next_frontier.add(_norm(f.entity))
            if added >= MAX_FACTS_PER_HOP:
                break
        frontier = next_frontier - {_norm(f.value) for f, _ in results}
    return results
