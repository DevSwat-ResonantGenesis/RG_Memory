"""
Fact Extraction Service (Tier 2, Step 5 + 6)
============================================

Extracts atomic facts from memory content via the LLM service, and applies
contradiction detection so a newer fact supersedes an older one asserting a
different value for the same (user, entity, attribute).

A fact is a subject-attribute-value triple with a confidence:
    {"entity": "user", "attribute": "name", "value": "Louie",
     "fact": "The user's name is Louie", "confidence": 0.95}

Design notes:
- Fact extraction runs best-effort and MUST NOT block or fail an ingest.
- Uses LLM_Service POST /chat/completions (OpenAI-format), the only real
  endpoint (the old summarization service called a nonexistent /generate).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import MemoryFact

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You extract durable, atomic facts about the user and their world from a "
    "message. Return ONLY a compact JSON array. Each item has keys: entity "
    "(the subject, e.g. 'user', a person, a project), attribute (a short "
    "predicate, e.g. 'name', 'location', 'goal', 'preference'), value (the "
    "object), fact (a short natural-language statement), confidence (0-1). "
    "Extract only stable, reusable facts — NOT questions, greetings, chit-chat, "
    "or transient state. If there are no durable facts, return []."
)


def _fact_hash(user_id: Optional[str], entity: str, attribute: str, value: str) -> str:
    key = f"{user_id or ''}|{(entity or '').lower()}|{(attribute or '').lower()}|{(value or '').lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _parse_facts(raw: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of fact dicts, tolerant of extra prose."""
    if not raw:
        return []
    # Pull out the first JSON array in the response.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []

    facts: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        attribute = str(item.get("attribute") or "").strip()
        value = str(item.get("value") or "").strip()
        fact_text = str(item.get("fact") or "").strip()
        if not fact_text and not (entity and attribute and value):
            continue
        if not fact_text:
            fact_text = f"{entity} {attribute} {value}".strip()
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        facts.append({
            "entity": entity[:255],
            "attribute": attribute[:255],
            "value": value,
            "fact": fact_text,
            "confidence": confidence,
        })
    return facts


class FactExtractionService:
    def __init__(self) -> None:
        self.llm_url = settings.LLM_SERVICE_URL
        self.model = settings.FACT_EXTRACTION_MODEL or None

    async def extract(self, content: str) -> List[Dict[str, Any]]:
        """Call the LLM to extract facts. Returns [] on any failure."""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        if self.model:
            payload["model"] = self.model
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(f"{self.llm_url}/llm/chat/completions", json=payload)
            if resp.status_code != 200:
                logger.warning("Fact extraction LLM returned %s", resp.status_code)
                return []
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return []
            raw = (choices[0].get("message") or {}).get("content") or ""
            return _parse_facts(raw)
        except Exception as e:  # network / parse / timeout — never propagate
            logger.warning("Fact extraction failed: %s", e)
            return []

    async def store_facts(
        self,
        session: AsyncSession,
        facts: List[Dict[str, Any]],
        *,
        memory_id: Optional[uuid.UUID],
        user_id: Optional[uuid.UUID],
        org_id: Optional[uuid.UUID],
        agent_hash: Optional[str] = None,
    ) -> int:
        """Persist facts with dedup + contradiction detection. Returns count stored."""
        stored = 0
        for f in facts:
            entity, attribute, value = f["entity"], f["attribute"], f["value"]
            fhash = _fact_hash(str(user_id) if user_id else None, entity, attribute, value)

            # Exact dedup: same triple already present and active → skip
            existing = await session.execute(
                select(MemoryFact.id).where(
                    MemoryFact.fact_hash == fhash,
                    MemoryFact.status == "active",
                ).limit(1)
            )
            if existing.scalar() is not None:
                continue

            new_fact = MemoryFact(
                memory_id=memory_id,
                user_id=user_id,
                org_id=org_id,
                agent_hash=agent_hash,
                fact=f["fact"],
                entity=entity or None,
                attribute=attribute or None,
                value=value or None,
                confidence=f["confidence"],
                status="active",
                fact_hash=fhash,
            )
            session.add(new_fact)
            await session.flush()  # get new_fact.id for superseding link

            # Contradiction detection: supersede older active facts about the same
            # (user, entity, attribute) that assert a DIFFERENT value.
            if user_id and entity and attribute:
                await session.execute(
                    update(MemoryFact)
                    .where(
                        MemoryFact.user_id == user_id,
                        MemoryFact.entity == entity,
                        MemoryFact.attribute == attribute,
                        MemoryFact.status == "active",
                        MemoryFact.id != new_fact.id,
                    )
                    .values(status="superseded", superseded_by=new_fact.id)
                )
            stored += 1

        if stored:
            await session.commit()
        return stored


fact_extraction_service = FactExtractionService()
