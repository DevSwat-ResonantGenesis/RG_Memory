"""Resonant Memory API client."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = "https://api.dev-swat.com"  # gateway (meters + isolates + bills)


class ResonantMemoryError(Exception):
    """Base SDK error."""


class InsufficientCreditsError(ResonantMemoryError):
    """Raised on 402 — top up credits to continue."""


class ResonantMemory:
    """Client for the Resonant Memory API.

    Auth: an org API key (``rg_live_...``) created in the dashboard. All calls
    are scoped to your org; pass ``user_id`` / ``agent_hash`` to isolate memories
    to a specific user or agent (the blockchain-block model):
      - user_id only              → the user's private memory block
      - agent_hash only           → the agent's global memory block
      - user_id + agent_hash      → the user+agent shared block
    """

    def __init__(
        self,
        api_key: str,
        *,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        agent_hash: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ResonantMemoryError("api_key is required")
        self.api_key = api_key
        self.user_id = user_id
        self.org_id = org_id
        self.agent_hash = agent_hash
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    # ---- internals -------------------------------------------------------
    def _scope(self, **overrides) -> Dict[str, Any]:
        s = {"user_id": self.user_id, "org_id": self.org_id, "agent_hash": self.agent_hash}
        s.update({k: v for k, v in overrides.items() if v is not None})
        return {k: v for k, v in s.items() if v is not None}

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = self._client.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as e:
            raise ResonantMemoryError(f"request failed: {e}") from e
        if r.status_code == 402:
            raise InsufficientCreditsError("out of credits — top up to continue")
        if r.status_code >= 400:
            raise ResonantMemoryError(f"{r.status_code}: {r.text[:200]}")
        return r.json()

    # ---- public API ------------------------------------------------------
    def ingest(
        self,
        content: str,
        *,
        user_id: Optional[str] = None,
        agent_hash: Optional[str] = None,
        event_timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "sdk",
    ) -> Dict[str, Any]:
        """Store a memory. Returns the created record (id, hash, coordinates).

        `event_timestamp` (ISO or natural date) sets the memory's event time for
        temporal recall ("when did X happen"). Fact extraction, on-chain anchoring,
        and the associative mesh update automatically. Cost: 120 credits.
        """
        payload = self._scope(user_id=user_id, agent_hash=agent_hash)
        payload.update({
            "content": content, "source": source, "generate_embedding": True,
        })
        if event_timestamp:
            payload["event_timestamp"] = event_timestamp
        if metadata:
            payload["metadata"] = metadata
        return self._post("/memory/ingest", payload)

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: Optional[str] = None,
        agent_hash: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant memories for a query (the full hash-sphere
        pipeline: gravity + anchors + mesh + cross-encoder + facts + knowledge
        graph, cosine/BM25 floor). Returns a list of memory dicts. Cost: 60 credits.
        """
        payload = self._scope(user_id=user_id, agent_hash=agent_hash)
        payload.update({"query": query, "limit": limit})
        if session_id:
            payload["session_id"] = session_id
        if min_score is not None:
            payload["min_score"] = min_score
        return self._post("/memory/hash-sphere/extract", payload).get("memories", [])

    def recall_full(self, query: str, **kw) -> Dict[str, Any]:
        """Like `recall` but returns the full response including `confidence`,
        `answer_from_memory` (no-LLM-recall signal), `evidence_hash`
        (on-chain provenance), and the extraction methods used."""
        payload = self._scope(user_id=kw.get("user_id"), agent_hash=kw.get("agent_hash"))
        payload.update({"query": query, "limit": kw.get("limit", 10)})
        if kw.get("session_id"):
            payload["session_id"] = kw["session_id"]
        if kw.get("min_score") is not None:
            payload["min_score"] = kw["min_score"]
        return self._post("/memory/hash-sphere/extract", payload)

    def facts(self, *, user_id: Optional[str] = None, entity: Optional[str] = None,
              attribute: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List the distilled atomic facts for a user (name/role/preferences/…),
        entity-resolved and contradiction-superseded. Cost: 20 credits."""
        params = {"user_id": user_id or self.user_id, "limit": limit}
        if entity:
            params["entity"] = entity
        if attribute:
            params["attribute"] = attribute
        try:
            r = self._client.get(f"{self.base_url}/memory/facts", params=params)
        except httpx.HTTPError as e:
            raise ResonantMemoryError(f"request failed: {e}") from e
        if r.status_code == 402:
            raise InsufficientCreditsError("out of credits — top up to continue")
        if r.status_code >= 400:
            raise ResonantMemoryError(f"{r.status_code}: {r.text[:200]}")
        return r.json().get("facts", [])

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
