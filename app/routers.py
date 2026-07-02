import os
import time
import uuid
import math
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_session, async_session
from .embeddings import embeddings_generator
from .models import MemoryRecord, MemoryEmbedding, MemoryAnchor, MemoryFact, EMBEDDING_DIM
from .services import resonance_hasher
from .services.resonance_hashing import ResonanceHasher
from .services.memory_encryption import memory_encryption, encrypt_memory_content, decrypt_memory_content
from .services.embedding_cache import embedding_cache
from .services.performance_logger import perf_tracker, TimingContext
from .services.semantic_cache import semantic_cache
from .services.pgvector_search import pgvector_search, VectorSearchResult
from .services.fact_extraction import fact_extraction_service
from .services.hash_sphere_core import encode_core, gravity as hs_gravity, core_from_stored
from .services.hash_sphere_model import hash_sphere_model
from .services import hash_sphere_anchors, hash_sphere_chain
from .schemas import (
    MemoryIngestRequest,
    MemoryRecordResponse,
    MemoryRetrieveRequest,
    MemorySearchResponse,
    ProjectSummaryResponse,
    ProjectsResponse,
    ProjectFileResponse,
    ProjectFilesResponse,
)

logger = logging.getLogger(__name__)

BLOCKCHAIN_SERVICE_URL = os.getenv("BLOCKCHAIN_SERVICE_URL", "http://blockchain_service:8000")
BILLING_SERVICE_URL = os.getenv("BILLING_SERVICE_URL", "http://billing_service:8000")
PREMIUM_AGENT_GLOBAL_FEATURE = "hash_sphere_access"


router = APIRouter(prefix="/memory", tags=["memory"])


async def _anchor_onchain_task(
    memory_id: str,
    content_hash: Optional[str],
    position_hash: Optional[str],
    user_id: Optional[str],
    org_id: Optional[str],
    agent_hash: Optional[str],
    source: Optional[str],
    dominant_cluster: Optional[str],
) -> None:
    """Background task: anchor the memory on-chain (hashes only) and store the
    resulting tx_hash on the record as proof-of-existence. Best-effort."""
    try:
        tx_hash = await hash_sphere_chain.anchor_memory(
            memory_id=memory_id,
            content_hash=content_hash,
            position_hash=position_hash,
            user_id=user_id,
            org_id=org_id,
            agent_hash=agent_hash,
            source=source,
            dominant_cluster=dominant_cluster,
        )
        if not tx_hash:
            logger.warning("On-chain anchor for %s returned no tx_hash", memory_id)
            return
        # extra_metadata is a JSON (not JSONB) column, so read-modify-write via ORM
        # rather than the jsonb || operator.
        async with async_session() as s:
            res = await s.execute(select(MemoryRecord).where(MemoryRecord.id == uuid.UUID(memory_id)))
            rec = res.scalar_one_or_none()
            if rec is not None:
                meta = dict(rec.extra_metadata or {})
                meta["blockchain_tx"] = tx_hash
                meta["onchain"] = True
                rec.extra_metadata = meta
                await s.commit()
        logger.info("Anchored memory %s on-chain: %s", memory_id, tx_hash[:16])
    except Exception as e:
        logger.debug("On-chain anchor task failed: %s", e)


async def _extract_facts_task(
    content: str,
    memory_id: Optional[uuid.UUID],
    user_uuid: Optional[uuid.UUID],
    org_uuid: Optional[uuid.UUID],
    agent_hash: Optional[str],
) -> None:
    """Background task: extract atomic facts from content and store them with
    contradiction detection. Best-effort — opens its own session and never raises."""
    try:
        facts = await fact_extraction_service.extract(content)
        if not facts:
            return
        async with async_session() as fact_session:
            stored = await fact_extraction_service.store_facts(
                fact_session,
                facts,
                memory_id=memory_id,
                user_id=user_uuid,
                org_id=org_uuid,
                agent_hash=agent_hash,
            )
        if stored:
            logger.info("Extracted and stored %d fact(s) from memory %s", stored, memory_id)
    except Exception as e:
        logger.warning("Fact extraction task failed (non-critical): %s", e)


@router.post("/ingest", response_model=MemoryRecordResponse)
async def ingest_memory(
    payload: MemoryIngestRequest,
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_session),
):
    """Ingest a memory record with FULL Hash Sphere coordinate system.

    background_tasks is optional so internal callers (e.g. the credited
    /memory/ingest wrapper in main.py) can invoke this directly; when absent,
    fact extraction is skipped for that call.
    """
    perf_tracker.increment("total_ingests")
    
    # Invalidate semantic cache for this user (new memory = stale cache)
    if payload.user_id:
        semantic_cache.invalidate_user(payload.user_id)
    
    user_uuid = None
    org_uuid = None
    chat_uuid = None
    try:
        user_uuid = uuid.UUID(payload.user_id) if payload.user_id else None
    except Exception:
        user_uuid = None
    try:
        org_uuid = uuid.UUID(payload.org_id) if payload.org_id else None
    except Exception:
        org_uuid = None
    try:
        chat_uuid = uuid.UUID(payload.chat_id) if payload.chat_id else None
    except Exception:
        chat_uuid = None

    # Generate embedding FIRST so it can drive XYZ coordinate calculation
    # This ensures Hash Sphere coordinates are SEMANTIC (similar text → similar xyz)
    ingest_embedding = None
    if payload.generate_embedding:
        ingest_embeddings = await embeddings_generator.generate([payload.content], task="search_document")
        if ingest_embeddings:
            ingest_embedding = ingest_embeddings[0]
    
    # Generate XYZ / sphere coordinates for the VISUALIZER only (not retrieval).
    coords = ResonanceHasher.compute_full_coordinates(
        text=payload.content,
        embedding=ingest_embedding,
        context=payload.metadata.get("context") if payload.metadata else None
    )

    # RFC-0002 Wave 1/2: compute the 12-D SEMANTIC CORE — the memory's real
    # position in the hash sphere; drives retrieval. hash = quantized position.
    # Wave 2: ensure the prototype model is built so α…ζ generalize via embeddings.
    hs_axes = None
    try:
        await hash_sphere_model.ensure_built(embeddings_generator)
        hs_axes = await hash_sphere_model.axes_for_text(payload.content, embeddings_generator)
    except Exception:
        hs_axes = None
    hs_core = encode_core(payload.content, embedding=ingest_embedding, axes=hs_axes)
    core_dict = hs_core.to_dict()

    # Deduplication: skip exact duplicate content for same user
    from .services.memory_deduplication import memory_deduplication
    content_hash = memory_deduplication.compute_content_hash(payload.content)
    if user_uuid:
        dup_result = await session.execute(
            select(MemoryRecord.id).where(
                MemoryRecord.user_id == user_uuid,
                MemoryRecord.content_hash == content_hash,
            ).limit(1)
        )
        dup_id = dup_result.scalar_one_or_none()
        if dup_id:
            logger.debug(f"[Dedup] Skipping exact duplicate for user {user_uuid}")
            perf_tracker.increment("dedup_skipped")
            return {"id": str(dup_id), "status": "duplicate", "message": "Exact duplicate — not stored"}
    
    # Encrypt content if encryption is enabled
    stored_content = encrypt_memory_content(payload.content)
    
    record = MemoryRecord(
        chat_id=chat_uuid,
        user_id=user_uuid,
        org_id=org_uuid,
        source=payload.source,
        content=stored_content,  # Store encrypted content
        content_hash=content_hash,
        extra_metadata=payload.metadata,
        agent_hash=payload.agent_hash,
        # ========== HASH SPHERE — 12-D SEMANTIC CORE (retrieval truth) ==========
        # hash IS the quantized 12-D position (Text ≡ core ≡ hash).
        hash=hs_core.hash(),
        meaning_hash=coords.meaning_hash,
        energy_hash=coords.energy_hash,
        spin_hash=coords.spin_hash,
        universe_id=coords.universe_id,
        # XYZ / hypersphere — VISUALIZATION ONLY, never used in retrieval.
        xyz_x=coords.x,
        xyz_y=coords.y,
        xyz_z=coords.z,
        sphere_r=coords.r,
        sphere_phi=coords.phi,
        sphere_theta=coords.theta,
        # Retrieval-relevant scalars come from the 12-D CORE:
        resonance_score=hs_core.resonance,
        normalized_resonance=(hs_core.resonance + 1.0) / 2.0,
        anchor_energy=hs_core.energy,                 # Energy = ±resonance (swap)
        spin_x=hs_core.spin[0],                       # Spin = intensity
        spin_y=hs_core.spin[1],                       # complexity
        spin_z=hs_core.spin[2],                       # abstraction
        spin_magnitude=sum(s * s for s in hs_core.spin) ** 0.5,
        meaning_score=coords.meaning_score,
        intensity_score=hs_core.spin[0],
        sentiment_score=hs_core.polarity,
        # Full 12-D core + viz coords as JSON (metric_vector read by retrieval).
        hash_sphere_coords={**coords.to_dict(), **core_dict},
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)

    # Overwrite search_tsv with plaintext — the INSERT trigger used the encrypted content
    await session.execute(
        text("UPDATE memory_records SET search_tsv = to_tsvector('english', :txt) WHERE id = :rid"),
        {"txt": payload.content, "rid": record.id},
    )
    await session.commit()

    # RFC-0002 Wave 3b: reinforce the live gravity field — join the nearest well
    # (drift it toward this memory) or spawn a new one. Best-effort.
    try:
        await hash_sphere_anchors.reinforce(
            session,
            user_uuid=user_uuid,
            org_uuid=org_uuid,
            agent_hash=payload.agent_hash,
            memory_id=record.id,
            core=hs_core,
        )
    except Exception as e:
        logger.debug("Anchor reinforce skipped: %s", e)

    # Store the embedding in MemoryEmbedding table for vector search.
    # The column is a fixed-width pgvector(EMBEDDING_DIM); skip any embedding whose
    # dimension doesn't match (e.g. a fallback provider kicked in) rather than crash.
    if ingest_embedding and len(ingest_embedding) == EMBEDDING_DIM:
        embedding_record = MemoryEmbedding(
            memory_id=record.id,
            user_id=user_uuid,
            org_id=org_uuid,
            embedding=ingest_embedding,
        )
        session.add(embedding_record)
        await session.commit()
    elif ingest_embedding:
        logger.warning(
            "Skipping embedding store: dimension %d != %d (fallback provider?)",
            len(ingest_embedding), EMBEDDING_DIM,
        )
    
    # Auto-create memory anchors from meaningful content
    # Anchors enable fast keyword-based memory lookup (PRIORITY 1 in extraction)
    if len(payload.content) >= 15 and user_uuid and org_uuid:
        try:
            anchor_keywords = resonance_hasher.extract_anchors(payload.content, max_anchors=3)
            for keyword in anchor_keywords:
                anchor_coords = ResonanceHasher.compute_full_coordinates(keyword)
                anchor = MemoryAnchor(
                    user_id=user_uuid,
                    org_id=org_uuid,
                    chat_id=chat_uuid,
                    message_id=record.id,
                    anchor_text=keyword,
                    anchor_hash=ResonanceHasher.hash_text_deterministic(keyword),
                    context=payload.content[:500],
                    importance_score=coords.meaning_score or 0.5,
                    xyz_x=anchor_coords.x,
                    xyz_y=anchor_coords.y,
                    xyz_z=anchor_coords.z,
                    anchor_type="chat" if payload.source in ("resonant-chat", "resonant-chat-history") else "memory",
                    agent_hash=payload.agent_hash,
                )
                session.add(anchor)
            await session.commit()

            # (On-chain anchoring is now per-memory via _anchor_onchain_task,
            # scheduled below — proof-of-existence for the whole memory, not just
            # keyword anchors, and posted to the correct /distributed endpoint.)

        except Exception as e:
            logger.warning(f"Anchor creation failed (non-critical): {e}")

    # Tier 2: extract atomic facts from this memory in the background (best-effort,
    # never blocks or fails the ingest response).
    if (
        background_tasks is not None
        and not payload.skip_enrichment
        and settings.ENABLE_FACT_EXTRACTION
        and payload.content
        and len(payload.content) >= settings.FACT_EXTRACTION_MIN_CHARS
    ):
        background_tasks.add_task(
            _extract_facts_task,
            payload.content,
            record.id,
            user_uuid,
            org_uuid,
            payload.agent_hash,
        )

    # RFC-0002 Wave 4: anchor this memory on-chain (hashes only) for immutable
    # proof-of-existence. Best-effort background task.
    if background_tasks is not None and not payload.skip_enrichment:
        background_tasks.add_task(
            _anchor_onchain_task,
            str(record.id),
            content_hash,
            hs_core.hash(),
            str(user_uuid) if user_uuid else None,
            str(org_uuid) if org_uuid else None,
            payload.agent_hash,
            payload.source,
            hs_core.dominant_cluster,
        )

    # Return decrypted content in response with FULL Hash Sphere coordinates
    return MemoryRecordResponse(
        id=str(record.id),
        chat_id=str(record.chat_id) if record.chat_id else None,
        user_id=str(record.user_id) if record.user_id else None,
        org_id=str(record.org_id) if record.org_id else None,
        agent_hash=record.agent_hash,
        source=record.source,
        content=payload.content,  # Return original plaintext, not encrypted
        metadata=record.extra_metadata,
        # Full Hash Sphere coordinates
        hash=record.hash,
        meaning_hash=record.meaning_hash,
        energy_hash=record.energy_hash,
        spin_hash=record.spin_hash,
        universe_id=record.universe_id,
        xyz=[record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x is not None else None,
        xyz_x=record.xyz_x,
        xyz_y=record.xyz_y,
        xyz_z=record.xyz_z,
        sphere_r=record.sphere_r,
        sphere_phi=record.sphere_phi,
        sphere_theta=record.sphere_theta,
        resonance_score=record.resonance_score,
        normalized_resonance=record.normalized_resonance,
        anchor_energy=record.anchor_energy,
        spin={"x": record.spin_x, "y": record.spin_y, "z": record.spin_z, "magnitude": record.spin_magnitude} if record.spin_x is not None else None,
        semantic={"meaning": record.meaning_score, "intensity": record.intensity_score, "sentiment": record.sentiment_score} if record.meaning_score is not None else None,
        cluster=record.cluster_name,
        hash_sphere_coords=record.hash_sphere_coords,
    )


@router.post("/retrieve", response_model=List[MemoryRecordResponse])
async def retrieve_memory(
    payload: MemoryRetrieveRequest,
    request: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """Retrieve memories using vector similarity search or metadata filtering."""
    retrieval_start = time.perf_counter()
    perf_tracker.increment("total_retrievals")

    # If client omitted user/org in body (common for browser visualizers),
    # fall back to gateway-injected headers.
    if request:
        if not payload.user_id:
            payload.user_id = request.headers.get("x-user-id")
        if not payload.org_id:
            payload.org_id = request.headers.get("x-org-id")

    effective_agent_hash: Optional[str] = None
    if payload.team_id:
        effective_agent_hash = f"team_{payload.team_id}"
    elif payload.agent_hash:
        effective_agent_hash = payload.agent_hash

    user_uuid = None
    org_uuid = None
    try:
        user_uuid = uuid.UUID(payload.user_id) if payload.user_id else None
    except Exception:
        user_uuid = None
    try:
        org_uuid = uuid.UUID(payload.org_id) if payload.org_id else None
    except Exception:
        org_uuid = None

    async def _allow_premium_agent_global(req: Optional[Request], user_id: Optional[str]) -> bool:
        if not req or not user_id:
            return False

        if (req.headers.get("x-is-dev-override") or "").lower() == "true":
            return True

        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{BILLING_SERVICE_URL}/economic-state/{user_id}/check-feature/{PREMIUM_AGENT_GLOBAL_FEATURE}"
                )
            if resp.status_code != 200:
                return False
            data = resp.json()
            return bool(data.get("allowed", False))
        except Exception:
            return False

    allow_premium_agent_global = False
    if effective_agent_hash and payload.user_id:
        allow_premium_agent_global = await _allow_premium_agent_global(request, payload.user_id)

    def _tier_for(scope: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if scope == "agent_global":
            if metadata and isinstance(metadata, dict) and metadata.get("tier") in {"public", "premium"}:
                return metadata.get("tier")
            return "public"
        return "private"

    retrieval_mode = (payload.retrieval_mode or "embedding").strip().lower()
    if retrieval_mode not in {"embedding", "hash_sphere", "hybrid"}:
        retrieval_mode = "embedding"

    def _recency_score(created_at: Optional[datetime]) -> float:
        if not created_at:
            return 0.5
        try:
            now = datetime.now(timezone.utc)
            ts = created_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            return float(math.exp(-age_days / 30.0))
        except Exception:
            return 0.5

    def _hash_sphere_rank(
        *,
        responses: List[MemoryRecordResponse],
        query_xyz: tuple[float, float, float],
        query_hash: str,
        query_resonance: float,
    ) -> List[MemoryRecordResponse]:
        from .services.hybrid_memory_ranker import rank_memories

        prepared: List[Dict[str, Any]] = []
        for r in responses:
            xyz = None
            if r.xyz and isinstance(r.xyz, list) and len(r.xyz) == 3 and all(v is not None for v in r.xyz):
                xyz = (float(r.xyz[0]), float(r.xyz[1]), float(r.xyz[2]))

            # Use embedding similarity as resonance_score (semantic)
            # r.similarity comes from pgvector cosine — it IS the embedding cosine
            resonance_score = 0.0
            if r.similarity and r.similarity > 0:
                resonance_score = float(r.similarity)
            elif r.hash:
                # FALLBACK: hash Hamming (legacy, only when no embedding similarity)
                try:
                    resonance_score = float(resonance_hasher.calculate_resonance(query_hash, r.hash))
                except Exception:
                    resonance_score = 0.0

            proximity_score = 0.0
            if xyz:
                try:
                    proximity_score = float(ResonanceHasher.calculate_proximity_score(query_xyz, xyz))
                except Exception:
                    proximity_score = 0.0

            resonance_function_score = 0.0
            if xyz:
                try:
                    mem_resonance = float(ResonanceHasher.calculate_resonance_function(xyz))
                    resonance_diff = abs(float(query_resonance) - mem_resonance)
                    resonance_function_score = max(0.0, 1.0 - (resonance_diff / 6.0))
                except Exception:
                    resonance_function_score = 0.0

            anchor_energy = 0.0
            if xyz:
                try:
                    import numpy as np

                    anchor_energy = float(
                        ResonanceHasher.calculate_anchor_energy(
                            np.array(query_xyz),
                            np.array(xyz),
                        )
                    )
                except Exception:
                    anchor_energy = 0.0

            prepared.append(
                {
                    "id": r.id,
                    "response": r,
                    "rag_score": float(r.similarity or 0.0),
                    "resonance_score": float(r.resonance_score or resonance_score or 0.0),
                    "proximity_score": float(proximity_score),
                    "recency_score": float(_recency_score(getattr(r, "created_at", None))),
                    "anchor_score": 0.0,
                    "resonance_function_score": float(resonance_function_score),
                    "anchor_energy": float(anchor_energy),
                }
            )

        ranked = rank_memories(prepared)
        ranked.sort(key=lambda m: float(m.get("hybrid_score", 0.0)), reverse=True)
        return [m["response"] for m in ranked]
    
    if retrieval_mode in {"embedding", "hybrid"} and payload.use_vector_search and payload.query:
        # Check semantic cache first (full query results)
        if payload.user_id:
            cache_query = (
                f"{payload.query}||org:{payload.org_id}||agent:{effective_agent_hash}"
                f"||agprem:{int(allow_premium_agent_global)}||mode:{retrieval_mode}||v3"
            )
            cached_results = semantic_cache.get(payload.user_id, cache_query)
            if cached_results:
                perf_tracker.log_cache_hit()
                retrieval_duration = (time.perf_counter() - retrieval_start) * 1000
                perf_tracker.log_timing("retrieval", retrieval_duration)
                return [MemoryRecordResponse(**r) for r in cached_results]
        
        # Check embedding cache
        cached_embedding = embedding_cache.get(payload.query)
        
        if cached_embedding:
            query_embedding = cached_embedding
            perf_tracker.log_cache_hit()
        else:
            # Generate query embedding with search_query task prefix for better retrieval
            async with TimingContext("embedding_generation"):
                query_embeddings = await embeddings_generator.generate([payload.query], task="search_query")
            if query_embeddings:
                query_embedding = query_embeddings[0]
                # Cache the embedding for future use
                embedding_cache.set(payload.query, query_embedding)
                perf_tracker.log_cache_miss()
            else:
                query_embedding = None
        
        if query_embedding:
            # Single UNION ALL query instead of 3 separate DB round-trips
            scope_results = await pgvector_search.search_multi_scope(
                session=session,
                query_embedding=query_embedding,
                user_id=user_uuid,
                org_id=org_uuid,
                agent_hash=effective_agent_hash,
                limit=payload.limit,
            )

            overlay_results = scope_results.get("user_overlay", [])
            user_global_results = scope_results.get("user_global", [])
            agent_global_results = scope_results.get("agent_global", [])

            merged: Dict[str, Dict[str, Any]] = {}

            def _add_results(results: List[VectorSearchResult], scope: str, boost: float) -> None:
                for r in results:
                    score = (r.similarity or 0.0) + boost
                    existing = merged.get(r.memory_id)
                    if existing and (existing.get("_score") or 0.0) >= score:
                        continue

                    record_user_id = None
                    record_agent_hash = None
                    if r.metadata and isinstance(r.metadata, dict):
                        record_user_id = r.metadata.get("record_user_id")
                        record_agent_hash = r.metadata.get("record_agent_hash")

                    tier = _tier_for(scope, r.metadata)
                    if scope == "agent_global" and tier == "premium" and not allow_premium_agent_global:
                        continue

                    merged[r.memory_id] = {
                        "_score": score,
                        "response": MemoryRecordResponse(
                            id=r.memory_id,
                            chat_id=None,
                            user_id=record_user_id,
                            org_id=payload.org_id,
                            agent_hash=record_agent_hash,
                            source="memory",
                            content=decrypt_memory_content(r.content),
                            metadata=r.metadata,
                            similarity=r.similarity,
                            hash=r.hash,
                            xyz=list(r.xyz) if r.xyz else None,
                            resonance_score=r.resonance_score,
                            scope=scope,
                            tier=tier,
                        ),
                    }

            _add_results(overlay_results, scope="user_overlay", boost=0.02)
            _add_results(user_global_results, scope="user_global", boost=0.01)
            _add_results(agent_global_results, scope="agent_global", boost=0.0)

            if merged:
                sorted_items = sorted(merged.values(), key=lambda x: x.get("_score", 0.0), reverse=True)
                final_results = [item["response"] for item in sorted_items]

                if retrieval_mode == "hybrid":
                    query_coords = ResonanceHasher.compute_full_coordinates(
                        payload.query,
                        embedding=query_embedding,
                    )
                    query_xyz = (float(query_coords.x), float(query_coords.y), float(query_coords.z))
                    query_hash = ResonanceHasher.hash_text(payload.query)
                    query_resonance = float(getattr(query_coords, "resonance_score", 0.0) or 0.0)
                    final_results = _hash_sphere_rank(
                        responses=final_results,
                        query_xyz=query_xyz,
                        query_hash=query_hash,
                        query_resonance=query_resonance,
                    )

                final_results = final_results[: payload.limit]

                if payload.user_id:
                    semantic_cache.set(
                        payload.user_id,
                        cache_query,
                        [r.dict() for r in final_results],
                    )

                retrieval_duration = (time.perf_counter() - retrieval_start) * 1000
                perf_tracker.log_timing("retrieval", retrieval_duration)
                return final_results
            
            # Fallback to original linear scan if pgvector not available
            # Get all embeddings for user
            stmt = select(MemoryEmbedding)
            if user_uuid:
                stmt = stmt.where(MemoryEmbedding.user_id == user_uuid)

            result = await session.execute(stmt)
            embeddings = result.scalars().all()

            # Calculate similarities
            similarities = []
            for emb in embeddings:
                similarity = embeddings_generator.cosine_similarity(
                    query_embedding, emb.embedding
                )
                similarities.append((emb.memory_id, similarity))

            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memory_ids = [mid for mid, _ in similarities[:payload.limit]]
            similarity_map = {mid: sim for mid, sim in similarities[:payload.limit]}

            # Fetch memory records (excluding archived)
            if top_memory_ids:
                stmt = select(MemoryRecord).where(MemoryRecord.id.in_(top_memory_ids))
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                # Filter out archived records
                active_records = [
                    r for r in records 
                    if not (r.extra_metadata and r.extra_metadata.get("is_archived", False))
                ]

                # Log retrieval timing
                retrieval_duration = (time.perf_counter() - retrieval_start) * 1000
                perf_tracker.log_timing("retrieval", retrieval_duration)
                
                return [
                    MemoryRecordResponse(
                        id=str(r.id),
                        chat_id=str(r.chat_id) if r.chat_id else None,
                        user_id=str(r.user_id) if r.user_id else None,
                        source=r.source,
                        content=decrypt_memory_content(r.content),  # Decrypt on retrieval
                        metadata=r.extra_metadata,
                        similarity=similarity_map.get(r.id),
                        # Hash Sphere fields for Layer 7/9
                        hash=r.hash,
                        xyz=[r.xyz_x, r.xyz_y, r.xyz_z] if r.xyz_x is not None else None,
                        resonance_score=r.resonance_score,
                    )
                    for r in active_records
                ]

    # Fallback to simple metadata filter
    stmt = select(MemoryRecord).order_by(MemoryRecord.created_at.desc())
    if payload.chat_id:
        try:
            stmt = stmt.where(MemoryRecord.chat_id == uuid.UUID(payload.chat_id))
        except Exception:
            stmt = stmt.where(MemoryRecord.chat_id == None)

    scope_filters = []
    if user_uuid:
        if org_uuid:
            scope_filters.append(
                and_(
                    MemoryRecord.user_id == user_uuid,
                    MemoryRecord.org_id == org_uuid,
                    MemoryRecord.agent_hash.is_(None),
                )
            )
            if effective_agent_hash:
                scope_filters.append(
                    and_(
                        MemoryRecord.user_id == user_uuid,
                        MemoryRecord.org_id == org_uuid,
                        MemoryRecord.agent_hash == effective_agent_hash,
                    )
                )
        else:
            scope_filters.append(
                and_(
                    MemoryRecord.user_id == user_uuid,
                    MemoryRecord.agent_hash.is_(None),
                )
            )
            if effective_agent_hash:
                scope_filters.append(
                    and_(
                        MemoryRecord.user_id == user_uuid,
                        MemoryRecord.agent_hash == effective_agent_hash,
                    )
                )

    if effective_agent_hash and org_uuid:
        scope_filters.append(
            and_(
                MemoryRecord.user_id.is_(None),
                MemoryRecord.org_id == org_uuid,
                MemoryRecord.agent_hash == effective_agent_hash,
            )
        )

    if scope_filters:
        stmt = stmt.where(or_(*scope_filters))

    result = await session.execute(stmt.limit(payload.limit))
    records = result.scalars().all()
    
    # Filter out archived records
    active_records = [
        r for r in records 
        if not (r.extra_metadata and r.extra_metadata.get("is_archived", False))
    ]

    response_records: List[MemoryRecordResponse] = []
    for r in active_records:
        scope = "user_global"
        if effective_agent_hash and r.agent_hash == effective_agent_hash and r.user_id is not None:
            scope = "user_overlay"
        if effective_agent_hash and r.agent_hash == effective_agent_hash and r.user_id is None:
            scope = "agent_global"

        tier = _tier_for(scope, r.extra_metadata)
        if scope == "agent_global" and tier == "premium" and not allow_premium_agent_global:
            continue

        response_records.append(
            MemoryRecordResponse(
                id=str(r.id),
                chat_id=str(r.chat_id) if r.chat_id else None,
                user_id=str(r.user_id) if r.user_id else None,
                org_id=str(r.org_id) if r.org_id else None,
                agent_hash=r.agent_hash,
                source=r.source,
                content=decrypt_memory_content(r.content),  # Decrypt on retrieval
                metadata=r.extra_metadata,
                # Hash Sphere fields for Layer 7/9
                hash=r.hash,
                xyz=[r.xyz_x, r.xyz_y, r.xyz_z] if r.xyz_x is not None else None,
                resonance_score=r.resonance_score,
                scope=scope,
                tier=tier,
            )
        )

    if retrieval_mode == "hash_sphere" and payload.query:
        query_coords = ResonanceHasher.compute_full_coordinates(payload.query)
        query_xyz = (float(query_coords.x), float(query_coords.y), float(query_coords.z))
        query_hash = ResonanceHasher.hash_text(payload.query)
        query_resonance = float(getattr(query_coords, "resonance_score", 0.0) or 0.0)
        response_records = _hash_sphere_rank(
            responses=response_records,
            query_xyz=query_xyz,
            query_hash=query_hash,
            query_resonance=query_resonance,
        )

    return response_records


@router.get("/perf/stats")
async def get_memory_perf_stats(session: AsyncSession = Depends(get_session)):
    """Get memory service performance statistics."""
    cache_stats = embedding_cache.get_stats()
    perf_stats = perf_tracker.get_stats()
    semantic_stats = semantic_cache.get_stats()
    pgvector_stats = await pgvector_search.get_index_stats(session)
    
    return {
        "embedding_cache": cache_stats,
        "semantic_cache": semantic_stats,
        "pgvector": pgvector_stats,
        "performance": perf_stats,
    }


@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(
    payload: MemoryRetrieveRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Search memories with detailed response."""
    memories = await retrieve_memory(payload, request, session)
    return MemorySearchResponse(
        memories=memories,
        query=payload.query,
        total_found=len(memories),
    )


@router.post("/create-vector-index")
async def create_vector_index(
    session: AsyncSession = Depends(get_session),
    lists: int = 100,
):
    """
    Create pgvector IVFFlat index for faster similarity search.
    
    Requires pgvector extension to be installed in PostgreSQL:
    CREATE EXTENSION IF NOT EXISTS vector;
    
    Args:
        lists: Number of IVFFlat lists (higher = more accurate, slower build)
    """
    success = await pgvector_search.create_vector_index(session, lists=lists)
    
    if success:
        return {
            "status": "success",
            "message": f"Created IVFFlat index with {lists} lists",
            "pgvector_available": True,
        }
    else:
        return {
            "status": "failed",
            "message": "Failed to create index. Ensure pgvector extension is installed.",
            "pgvector_available": False,
            "hint": "Run: CREATE EXTENSION IF NOT EXISTS vector; in PostgreSQL",
        }


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a memory record and its embedding."""
    # Delete embedding
    stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
    result = await session.execute(stmt)
    embedding = result.scalar_one_or_none()
    if embedding:
        await session.delete(embedding)

    # Delete memory record
    stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()
    if record:
        await session.delete(record)
        await session.commit()
        return {"status": "deleted", "id": memory_id}

    return {"status": "not_found", "id": memory_id}


@router.get("/encryption/status")
async def encryption_status():
    """Get memory encryption service status."""
    return memory_encryption.get_status()


@router.get("/facts")
async def list_facts(
    user_id: Optional[str] = None,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    include_superseded: bool = False,
    limit: int = 100,
    request: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """List LLM-extracted atomic facts for a user (Tier 2).

    Returns active facts by default (contradiction-superseded ones excluded).
    Optionally filter by entity/attribute or include superseded history.
    """
    if request and not user_id:
        user_id = request.headers.get("x-user-id")

    user_uuid = None
    try:
        user_uuid = uuid.UUID(user_id) if user_id else None
    except Exception:
        user_uuid = None
    if not user_uuid:
        return {"facts": [], "total": 0}

    stmt = select(MemoryFact).where(MemoryFact.user_id == user_uuid)
    if not include_superseded:
        stmt = stmt.where(MemoryFact.status == "active")
    if entity:
        stmt = stmt.where(MemoryFact.entity == entity)
    if attribute:
        stmt = stmt.where(MemoryFact.attribute == attribute)
    stmt = stmt.order_by(MemoryFact.confidence.desc(), MemoryFact.created_at.desc()).limit(min(limit, 500))

    result = await session.execute(stmt)
    rows = result.scalars().all()
    return {
        "facts": [
            {
                "id": str(f.id),
                "fact": f.fact,
                "entity": f.entity,
                "attribute": f.attribute,
                "value": f.value,
                "confidence": f.confidence,
                "status": f.status,
                "superseded_by": str(f.superseded_by) if f.superseded_by else None,
                "memory_id": str(f.memory_id) if f.memory_id else None,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in rows
        ],
        "total": len(rows),
    }


@router.get("/stats")
async def memory_stats(
    user_id: Optional[str] = None,
    request: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """Get memory statistics including RAG documents and storage usage."""
    # Get user_id from header if not provided as param
    if not user_id and request:
        user_id = request.headers.get("x-user-id")

    import uuid

    user_uuid: Optional[uuid.UUID] = None
    try:
        if user_id:
            user_uuid = uuid.UUID(user_id)
    except Exception:
        user_uuid = None
    
    stmt = select(MemoryRecord)
    if user_uuid:
        stmt = stmt.where(MemoryRecord.user_id == user_uuid)

    result = await session.execute(stmt)
    records = result.scalars().all()

    sources = {}
    total_storage_bytes = 0
    rag_documents = 0
    anchors_count = 0
    cluster_ids = set()
    cluster_names = set()
    
    for r in records:
        sources[r.source] = sources.get(r.source, 0) + 1
        # Calculate storage
        if r.content:
            total_storage_bytes += len(r.content.encode('utf-8'))
        # Count RAG documents (source == "rag")
        if r.source == "rag":
            rag_documents += 1
        # Count anchors
        if r.source == "anchor":
            anchors_count += 1

        if r.cluster_id:
            cluster_ids.add(str(r.cluster_id))
        if r.cluster_name:
            cluster_names.add(r.cluster_name)

    embeddings_stmt = select(func.count(MemoryEmbedding.id))
    if user_uuid:
        embeddings_stmt = embeddings_stmt.where(MemoryEmbedding.user_id == user_uuid)
    embeddings_result = await session.execute(embeddings_stmt)
    total_embeddings = int(embeddings_result.scalar() or 0)

    total_clusters = max(len(cluster_ids), len(cluster_names))
    avg_cluster_size = round((len(records) / total_clusters), 2) if total_clusters > 0 else 0

    storage_mb = round(total_storage_bytes / (1024 * 1024), 2)

    return {
        "total_memories": len(records),
        "by_source": sources,
        "rag_documents": rag_documents,
        "anchors_count": anchors_count,
        "storage_bytes": total_storage_bytes,
        "storage_mb": storage_mb,
        # Frontend compatibility keys
        "total_anchors": anchors_count,
        "total_embeddings": total_embeddings,
        "storage_size_mb": storage_mb,
        "total_clusters": total_clusters,
        "avg_cluster_size": avg_cluster_size,
    }


@router.get("/health")
async def health():
    return {"service": "memory", "status": "ok"}


# ============================================
# PROJECT FILES ENDPOINTS
# ============================================

@router.get("/projects", response_model=ProjectsResponse)
async def list_projects(
    req: Request,
    user_id: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List projects for the current user.

    Projects are inferred from MemoryRecord.extra_metadata entries that include
    a stable `project_id`. This is used by gateway /code/projects.
    """
    effective_user_id = (req.headers.get("x-user-id") or "").strip() or (user_id or "").strip() or None
    effective_org_id = (req.headers.get("x-org-id") or "").strip() or None

    if not effective_user_id:
        return ProjectsResponse(projects=[], count=0)

    try:
        user_uuid = uuid.UUID(effective_user_id)
    except Exception:
        return ProjectsResponse(projects=[], count=0)

    org_uuid = None
    if effective_org_id:
        try:
            org_uuid = uuid.UUID(effective_org_id)
        except Exception:
            org_uuid = None

    stmt = select(MemoryRecord).where(MemoryRecord.user_id == user_uuid)
    if org_uuid is not None:
        stmt = stmt.where(MemoryRecord.org_id == org_uuid)

    # Keep this bounded; projects should be discoverable from recent writes.
    stmt = stmt.order_by(MemoryRecord.created_at.desc()).limit(5000)
    result = await session.execute(stmt)
    records = result.scalars().all()

    projects: Dict[str, Dict[str, Any]] = {}
    for record in records:
        metadata = record.extra_metadata
        if metadata is None:
            continue
        if isinstance(metadata, str):
            try:
                import json
                metadata = json.loads(metadata)
            except Exception:
                continue
        if not isinstance(metadata, dict):
            continue

        project_id = metadata.get("project_id")
        if not project_id:
            continue

        name = (
            metadata.get("project_name")
            or metadata.get("project")
            or metadata.get("name")
            or str(project_id)
        )

        entry = projects.get(project_id)
        created_at = record.created_at.isoformat() if record.created_at else None
        if not entry:
            projects[project_id] = {
                "project_id": str(project_id),
                "name": str(name),
                "file_count": 1 if metadata.get("file_path") else 0,
                "created_at": created_at,
                "updated_at": created_at,
            }
            continue

        if metadata.get("file_path"):
            entry["file_count"] = int(entry.get("file_count", 0) or 0) + 1

        if created_at:
            entry["updated_at"] = entry.get("updated_at") or created_at
            if entry["updated_at"] < created_at:
                entry["updated_at"] = created_at
            entry["created_at"] = entry.get("created_at") or created_at
            if entry["created_at"] > created_at:
                entry["created_at"] = created_at

    # Sort by updated_at desc
    items = list(projects.values())
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    if limit and limit > 0:
        items = items[: int(limit)]

    return ProjectsResponse(
        projects=[ProjectSummaryResponse(**p) for p in items],
        count=len(items),
    )

@router.get("/project/files", response_model=ProjectFilesResponse)
async def get_project_files(
    project_id: str,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get all files for a project from Hash Sphere memory.
    
    Excludes archived files - they exist in Hash Sphere but won't be returned.
    """
    from sqlalchemy import cast, String
    from sqlalchemy.dialects.postgresql import JSONB
    
    user_id = req.headers.get("x-user-id")
    
    # Find all memory records - fetch all and filter in Python
    # This is less efficient but more reliable for JSON filtering
    # Order newest first so de-dup picks latest version per file_path.
    stmt = select(MemoryRecord).order_by(MemoryRecord.created_at.desc())
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    # Filter and build file list
    files = []
    seen_paths = set()
    
    for record in records:
        # Get metadata - handle both dict and None
        metadata = record.extra_metadata
        if metadata is None:
            continue
        
        # Handle case where metadata might be a string (shouldn't happen but safety)
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                continue
        
        # Skip if not matching project_id
        record_project_id = metadata.get("project_id") if isinstance(metadata, dict) else None
        if record_project_id != project_id:
            continue
        
        # Skip archived files
        if metadata.get("is_archived", False):
            continue
        
        file_path = metadata.get("file_path")
        if not file_path or file_path in seen_paths:
            continue
        
        seen_paths.add(file_path)
        
        # Determine if file or directory
        file_type = metadata.get("type", "file")
        
        # Decrypt content before returning
        decrypted_content = decrypt_memory_content(record.content) if record.content else None
        
        files.append(ProjectFileResponse(
            path=file_path,
            type=file_type,
            size=len(decrypted_content) if decrypted_content else 0,
            content=decrypted_content if file_type == "file" else None,
            language=metadata.get("language"),
        ))
    
    # Sort by path for consistent ordering
    files.sort(key=lambda f: f.path)
    
    return ProjectFilesResponse(
        project_id=project_id,
        files=files,
        total=len(files),
    )

