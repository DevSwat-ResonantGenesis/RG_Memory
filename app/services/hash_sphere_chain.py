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

import logging
from typing import Optional

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
