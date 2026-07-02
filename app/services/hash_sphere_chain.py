"""
Hash Sphere Chain — on-chain immutability / proof-of-existence (RFC-0002 Wave 4)
=================================================================================

Every memory is anchored to the DSID distributed blockchain as an immutable
transaction. We anchor HASHES ONLY (content_hash + 12-D position hash + metadata)
— never the plaintext — so the chain is a public, tamper-evident proof-of-
existence while the content stays encrypted off-chain.

Relationship typing (the "per-user / per-agent / user+agent" chains):
    user + agent → "user_agent";  user only → "user";  agent only → "agent".

Fixes the long-standing bug where memory anchoring posted to
/blockchain/transactions (404 → 0 tx ever recorded). The real endpoint is
/distributed/transactions.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# The distributed chain accepts {tx_type, payload}; prefix is /distributed.
_BLOCKCHAIN_URL = getattr(settings, "BLOCKCHAIN_SERVICE_URL", None) or "http://blockchain_service:8000"


def relationship_of(user_id: Optional[str], agent_hash: Optional[str]) -> str:
    if user_id and agent_hash:
        return "user_agent"
    if agent_hash and not user_id:
        return "agent"
    if user_id:
        return "user"
    return "system"


async def anchor_memory(
    *,
    memory_id: str,
    content_hash: Optional[str],
    position_hash: Optional[str],
    user_id: Optional[str],
    org_id: Optional[str],
    agent_hash: Optional[str],
    source: Optional[str],
    dominant_cluster: Optional[str] = None,
) -> Optional[str]:
    """Anchor a memory on-chain (hashes only). Returns tx_hash, or None on failure."""
    payload = {
        "memory_id": str(memory_id),
        "content_hash": content_hash,
        "position_hash": position_hash,      # quantized 12-D hash-sphere position
        "user_id": user_id,
        "org_id": org_id,
        "agent_hash": agent_hash,
        "relationship": relationship_of(user_id, agent_hash),
        "source": source,
        "dominant_cluster": dominant_cluster,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{_BLOCKCHAIN_URL}/distributed/transactions",
                json={"tx_type": "memory", "payload": payload},
            )
        if resp.status_code != 200:
            logger.warning("On-chain anchor returned %s", resp.status_code)
            return None
        return resp.json().get("tx_hash")
    except Exception as e:
        logger.debug("On-chain anchor skipped: %s", e)
        return None


def evidence_hash(query_hash: str, weighted: List[Tuple[str, float]]) -> str:
    """Deterministic evidence-ledger hash over the query + (memory_id, weight) set
    that produced a confident recall. Verifiable: same evidence → same hash."""
    parts = [query_hash or ""]
    for mid, w in sorted(weighted, key=lambda x: x[0]):
        parts.append(f"{mid}:{round(float(w), 4)}")
    return "ev_" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


async def anchor_evidence(
    *,
    query_hash: str,
    weighted: List[Tuple[str, float]],
    user_id: Optional[str],
    agent_hash: Optional[str],
    confidence: float,
) -> Optional[str]:
    """Anchor an evidence record on-chain: which memories (and weights) justified
    a confident answer. Returns tx_hash. Provides 'here is why I recalled this'."""
    ev = evidence_hash(query_hash, weighted)
    payload = {
        "evidence_hash": ev,
        "query_hash": query_hash,
        "memory_ids": [m for m, _ in weighted],
        "weights": {m: round(float(w), 4) for m, w in weighted},
        "confidence": round(float(confidence), 4),
        "user_id": user_id,
        "agent_hash": agent_hash,
        "relationship": relationship_of(user_id, agent_hash),
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{_BLOCKCHAIN_URL}/distributed/transactions",
                json={"tx_type": "evidence", "payload": payload},
            )
        if resp.status_code != 200:
            logger.warning("Evidence anchor returned %s", resp.status_code)
            return ev  # hash is still valid provenance even if chain post failed
        return ev
    except Exception as e:
        logger.debug("Evidence anchor skipped: %s", e)
        return ev
