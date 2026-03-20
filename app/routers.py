from typing import Any, Dict, List, Optional
import os
import time
import uuid
import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, Request, HTTPException, FastAPI
from pydantic import BaseModel
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

# Import crypto identity helper
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.crypto_identity import get_crypto_identity
    CRYPTO_IDENTITY_AVAILABLE = True
except ImportError:
    CRYPTO_IDENTITY_AVAILABLE = False

from .db import get_session
from .auth import get_current_user, get_optional_user
from .embeddings import embeddings_generator
from .models import MemoryRecord, MemoryEmbedding, MemoryChunk, MemoryAnchor
from .services import resonance_hasher, memory_anchor_service
from .services.resonance_hashing import ResonanceHasher, HashSphereCoordinates
from .services.memory_encryption import memory_encryption, encrypt_memory_content, decrypt_memory_content
from .services.memory_deduplication import memory_deduplication, check_for_duplicate
from .services.embedding_cache import embedding_cache
from .services.performance_logger import perf_tracker, TimingContext

BLOCKCHAIN_SERVICE_URL = os.getenv("BLOCKCHAIN_SERVICE_URL", "http://blockchain_service:8000")
from .services.semantic_cache import semantic_cache
from .services.pgvector_search import pgvector_search, VectorSearchResult
from .services.document_loaders import parse_document, chunk_text


BILLING_SERVICE_URL = os.getenv("BILLING_SERVICE_URL", "http://billing_service:8000")
PREMIUM_AGENT_GLOBAL_FEATURE = "hash_sphere_access"


router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryIngestRequest(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    generate_embedding: bool = True
    agent_hash: Optional[str] = None


class MemoryRecordResponse(BaseModel):
    id: str
    chat_id: Optional[str]
    user_id: Optional[str]
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None

    scope: Optional[str] = None
    tier: Optional[str] = None
    
    # ========== FULL HASH SPHERE COORDINATE SYSTEM ==========
    # Layer 2: Hash Generation
    hash: Optional[str] = None
    meaning_hash: Optional[str] = None
    energy_hash: Optional[str] = None
    spin_hash: Optional[str] = None
    
    # Layer 3: Universe ID
    universe_id: Optional[str] = None
    
    # Layer 5: Cartesian Coordinates (backward compatible)
    xyz: Optional[List[float]] = None
    xyz_x: Optional[float] = None
    xyz_y: Optional[float] = None
    xyz_z: Optional[float] = None
    
    # Hyperspherical Coordinates
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    
    # Layer 6: Resonance Scoring
    resonance_score: Optional[float] = None
    normalized_resonance: Optional[float] = None
    
    # Anchor Energy
    anchor_energy: Optional[float] = None
    
    # Spin Vector
    spin: Optional[Dict[str, Any]] = None
    
    # Semantic Components
    semantic: Optional[Dict[str, float]] = None
    
    # Cluster Assignment
    cluster: Optional[str] = None
    
    # Full coordinates as JSON
    hash_sphere_coords: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MemoryRetrieveRequest(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None
    team_id: Optional[str] = None
    query: str
    limit: int = 5
    use_vector_search: bool = True
    retrieval_mode: str = "embedding"


class MemorySearchResponse(BaseModel):
    memories: List[MemoryRecordResponse]
    query: str
    total_found: int


@router.post("/ingest", response_model=MemoryRecordResponse)
async def ingest_memory(
    payload: MemoryIngestRequest,
    session: AsyncSession = Depends(get_session),
):
    """Ingest a memory record with FULL Hash Sphere coordinate system."""
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

    # Generate FULL Hash Sphere coordinates using 9-Layer Architecture
    coords = ResonanceHasher.compute_full_coordinates(
        text=payload.content,
        embedding=None,  # Will be computed below if requested
        context=payload.metadata.get("context") if payload.metadata else None
    )
    
    # Encrypt content if encryption is enabled
    stored_content = encrypt_memory_content(payload.content)
    
    record = MemoryRecord(
        chat_id=chat_uuid,
        user_id=user_uuid,
        org_id=org_uuid,
        source=payload.source,
        content=stored_content,  # Store encrypted content
        extra_metadata=payload.metadata,
        agent_hash=payload.agent_hash,
        # ========== FULL HASH SPHERE COORDINATE SYSTEM ==========
        # Layer 2: Hash Generation
        hash=coords.hash,
        meaning_hash=coords.meaning_hash,
        energy_hash=coords.energy_hash,
        spin_hash=coords.spin_hash,
        # Layer 3: Universe ID
        universe_id=coords.universe_id,
        # Layer 5: Cartesian Coordinates
        xyz_x=coords.x,
        xyz_y=coords.y,
        xyz_z=coords.z,
        # Hyperspherical Coordinates
        sphere_r=coords.r,
        sphere_phi=coords.phi,
        sphere_theta=coords.theta,
        # Layer 6: Resonance Scoring
        resonance_score=coords.resonance_score,
        normalized_resonance=coords.normalized_resonance,
        # Anchor Energy
        anchor_energy=coords.energy,
        # Spin Vector
        spin_x=coords.spin_x,
        spin_y=coords.spin_y,
        spin_z=coords.spin_z,
        spin_magnitude=coords.spin_magnitude,
        # Semantic Components
        meaning_score=coords.meaning_score,
        intensity_score=coords.intensity_score,
        sentiment_score=coords.sentiment_score,
        # Full coordinates as JSON
        hash_sphere_coords=coords.to_dict(),
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)

    # Generate embedding if requested (use search_document task for storage)
    if payload.generate_embedding:
        embeddings = await embeddings_generator.generate([payload.content], task="search_document")
        if embeddings:
            embedding_record = MemoryEmbedding(
                memory_id=record.id,
                user_id=user_uuid,
                org_id=org_uuid,
                embedding=embeddings[0],
            )
            session.add(embedding_record)
            await session.commit()
    
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

            # Record memory anchors as blockchain transactions (fire-and-forget)
            try:
                async with httpx.AsyncClient(timeout=3.0) as http_client:
                    for keyword in anchor_keywords:
                        anchor_coords = ResonanceHasher.compute_full_coordinates(keyword)
                        anchor_hash = ResonanceHasher.hash_text_deterministic(keyword)
                        await http_client.post(
                            f"{BLOCKCHAIN_SERVICE_URL}/blockchain/transactions",
                            json={
                                "tx_type": "memory_anchor",
                                "payload": {
                                    "anchor_hash": anchor_hash,
                                    "anchor_text": keyword[:100],
                                    "xyz_x": round(anchor_coords.x, 6),
                                    "xyz_y": round(anchor_coords.y, 6),
                                    "xyz_z": round(anchor_coords.z, 6),
                                    "source": payload.source,
                                    "user_id": str(user_uuid) if user_uuid else None,
                                    "chat_id": str(chat_uuid) if chat_uuid else None,
                                },
                                "from_dsid": None,
                                "to_dsid": None,
                            },
                        )
                logger.info("Recorded %d memory anchors as blockchain transactions", len(anchor_keywords))
            except Exception as bc_err:
                logger.debug("Blockchain anchor recording skipped: %s", bc_err)

        except Exception as e:
            logger.warning(f"Anchor creation failed (non-critical): {e}")

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

            resonance_score = 0.0
            if r.hash:
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
                    query_coords = ResonanceHasher.compute_full_coordinates(payload.query)
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


# ============================================
# FULL HASH SPHERE MEMORY EXTRACTION ENDPOINT
# Production-ready multi-layer memory retrieval
# ============================================

class HashSphereExtractRequest(BaseModel):
    """Request for full Hash Sphere memory extraction."""
    query: str
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None  # For shared agent memory
    limit: int = 10
    # Enable/disable extraction methods
    use_anchors: bool = True      # Layer 4: Anchor-based lookup
    use_proximity: bool = True    # Layer 5: XYZ proximity search
    use_resonance: bool = True    # Layer 6: Hash resonance filtering
    use_clusters: bool = True     # Cluster-based retrieval
    use_rag_fallback: bool = True # RAG as last resort fallback
    # Advanced options
    include_coordinates: bool = True  # Include full Hash Sphere coords in response
    apply_magnetic_pull: bool = True  # Apply HS-MPS non-linear boost


class HashSphereMemory(BaseModel):
    """Memory with full Hash Sphere coordinates and scores."""
    id: str
    content: str
    type: str = "message"
    # Hash Sphere coordinates
    hash: Optional[str] = None
    xyz: Optional[List[float]] = None
    universe_id: Optional[str] = None
    # Hyperspherical coordinates
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    # Multi-method scores
    hybrid_score: float = 0.0
    rag_score: float = 0.0
    resonance_score: float = 0.0
    proximity_score: float = 0.0
    anchor_score: float = 0.0
    recency_score: float = 0.0
    # Layer 4 & 5 scores
    anchor_energy: float = 0.0
    resonance_function_score: float = 0.0
    # Magnetic pull
    magnetic_score: float = 0.0
    gravity_force: float = 0.0
    # Metadata
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HashSphereExtractResponse(BaseModel):
    """Response with full Hash Sphere extraction results."""
    memories: List[HashSphereMemory]
    query: str
    query_hash: str
    query_xyz: List[float]
    query_resonance: float
    total_found: int
    extraction_methods_used: List[str]
    # Performance metrics
    extraction_time_ms: float


@router.post("/hash-sphere/extract", response_model=HashSphereExtractResponse)
async def extract_hash_sphere_memories(
    request: HashSphereExtractRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    FULL Hash Sphere Memory Extraction - Production Ready
    
    This is the main endpoint for multi-layer memory retrieval using the
    complete 9-Layer Hash Sphere Architecture:
    
    - Layer 1: Input Processing (normalization)
    - Layer 2: Hash Generation (meaning + energy + spin)
    - Layer 3: Universe ID (SHA-256)
    - Layer 4: Anchor Energy calculation
    - Layer 5: Coordinate calculation (XYZ + hyperspherical)
    - Layer 6: Resonance scoring
    - Layer 7: Evidence aggregation
    - Layer 8: Multi-LLM routing (handled by chat service)
    - Layer 9: Output correction (handled by chat service)
    
    Extraction Methods (in priority order):
    1. Anchor-based lookup (fast keyword matching)
    2. Proximity search (3D XYZ distance)
    3. Resonance filtering (hash similarity)
    4. Cluster retrieval (context-based)
    5. RAG fallback (vector similarity - LAST RESORT)
    
    Scoring uses Hybrid Memory Ranker with weights:
    - RAG: 0.30 (fallback)
    - Resonance: 0.25
    - Resonance Function: 0.15
    - Anchor Energy: 0.10
    - Proximity: 0.10
    - Recency: 0.05
    - Anchor: 0.05
    """
    import time
    import uuid
    from sqlalchemy.orm import Session as SyncSession
    
    start_time = time.perf_counter()
    methods_used = []
    
    # Generate query hash and coordinates
    query_hash = ResonanceHasher.hash_text(request.query)
    query_coords = ResonanceHasher.compute_full_coordinates(request.query)
    query_xyz = (query_coords.x, query_coords.y, query_coords.z)
    query_resonance = query_coords.resonance_score
    
    # Collect all memories from different methods
    all_memories: Dict[str, Dict] = {}
    
    # Convert user_id/org_id to UUID if provided
    user_uuid = uuid.UUID(request.user_id) if request.user_id else None
    org_uuid = uuid.UUID(request.org_id) if request.org_id else None
    
    # ============================================
    # METHOD 1: Anchor-based lookup (PRIORITY 1)
    # ============================================
    if request.use_anchors:
        try:
            # Extract keywords from query
            keywords = resonance_hasher.extract_anchors(request.query)
            if keywords:
                # Search for anchors matching keywords
                stmt = select(MemoryRecord).where(
                    MemoryRecord.source == "anchor"
                )
                if user_uuid:
                    stmt = stmt.where(MemoryRecord.user_id == user_uuid)
                
                result = await session.execute(stmt)
                anchor_records = result.scalars().all()
                
                for record in anchor_records:
                    content_lower = record.content.lower()
                    for keyword in keywords:
                        if keyword.lower() in content_lower:
                            mem_id = str(record.id)
                            if mem_id not in all_memories:
                                importance = record.extra_metadata.get("importance_score", 0.5) if record.extra_metadata else 0.5
                                all_memories[mem_id] = {
                                    "id": mem_id,
                                    "content": decrypt_memory_content(record.content),
                                    "type": "anchor",
                                    "hash": record.hash,
                                    "xyz": [record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x else None,
                                    "anchor_score": importance,
                                    "timestamp": record.created_at.isoformat() if record.created_at else None,
                                }
                            break
                
                if all_memories:
                    methods_used.append("anchor")
        except Exception as e:
            pass  # Continue with other methods
    
    # ============================================
    # METHOD 2: Proximity search (PRIORITY 2)
    # ============================================
    if request.use_proximity:
        try:
            # Get memories with XYZ coordinates
            stmt = select(MemoryRecord).where(
                MemoryRecord.xyz_x.isnot(None)
            ).limit(200)
            if user_uuid:
                stmt = stmt.where(MemoryRecord.user_id == user_uuid)
            
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            for record in records:
                if record.xyz_x is not None:
                    mem_xyz = (record.xyz_x, record.xyz_y, record.xyz_z)
                    proximity = ResonanceHasher.calculate_proximity_score(query_xyz, mem_xyz)
                    
                    # Decrypt and quality-filter
                    content = decrypt_memory_content(record.content)
                    if not content or len(content) < 10 or content.startswith("ENC2:"):
                        continue
                    
                    mem_id = str(record.id)
                    if mem_id not in all_memories:
                        all_memories[mem_id] = {
                            "id": mem_id,
                            "content": content,
                            "type": record.source or "memory",
                            "hash": record.hash,
                            "xyz": [record.xyz_x, record.xyz_y, record.xyz_z],
                            "proximity_score": proximity,
                            "timestamp": record.created_at.isoformat() if record.created_at else None,
                        }
                    else:
                        all_memories[mem_id]["proximity_score"] = proximity
            
            if records:
                methods_used.append("proximity")
        except Exception as e:
            pass
    
    # ============================================
    # METHOD 3: Resonance filtering (PRIORITY 3)
    # ============================================
    if request.use_resonance:
        try:
            # Get memories with hashes
            stmt = select(MemoryRecord).where(
                MemoryRecord.hash.isnot(None)
            ).limit(200)
            if user_uuid:
                stmt = stmt.where(MemoryRecord.user_id == user_uuid)
            
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            for record in records:
                if record.hash:
                    resonance = resonance_hasher.calculate_resonance(query_hash, record.hash)
                    
                    # Decrypt and quality-filter
                    content = decrypt_memory_content(record.content)
                    if not content or len(content) < 10 or content.startswith("ENC2:"):
                        continue
                    
                    mem_id = str(record.id)
                    if mem_id not in all_memories:
                        all_memories[mem_id] = {
                            "id": mem_id,
                            "content": content,
                            "type": record.source or "memory",
                            "hash": record.hash,
                            "xyz": [record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x else None,
                            "resonance_score": resonance,
                            "timestamp": record.created_at.isoformat() if record.created_at else None,
                        }
                    else:
                        all_memories[mem_id]["resonance_score"] = resonance
            
            if records:
                methods_used.append("resonance")
        except Exception as e:
            pass
    
    # ============================================
    # METHOD 4: RAG Semantic Search (ALWAYS RUN)
    # RAG has the highest hybrid weight (0.30) — it MUST always run
    # to provide semantic similarity scores for proper ranking.
    # Previously gated behind quota which made it never execute.
    # ============================================
    if request.use_rag_fallback:
        try:
            # Generate query embedding
            query_embeddings = await embeddings_generator.generate([request.query], task="search_query")
            if query_embeddings:
                query_embedding = query_embeddings[0]
                
                # Use pgvector search — fetch more candidates for better ranking
                pgvector_results = await pgvector_search.search_similar(
                    session=session,
                    query_embedding=query_embedding,
                    user_id=request.user_id,
                    limit=request.limit * 2,
                )
                
                for r in pgvector_results:
                    # Quality filter on RAG results too
                    if not r.content or len(r.content) < 10:
                        continue
                    
                    mem_id = r.memory_id
                    if mem_id not in all_memories:
                        all_memories[mem_id] = {
                            "id": mem_id,
                            "content": r.content,
                            "type": "memory",
                            "hash": r.hash,
                            "xyz": list(r.xyz) if r.xyz else None,
                            "rag_score": r.similarity,
                            "resonance_score": r.resonance_score or 0.0,
                            "timestamp": None,
                        }
                    else:
                        # CRITICAL: Merge RAG score into existing memory
                        # This ensures memories found by other methods also
                        # get their semantic similarity score for hybrid ranking
                        all_memories[mem_id]["rag_score"] = r.similarity
                
                if pgvector_results:
                    methods_used.append("rag_semantic")
        except Exception as e:
            pass
    
    # ============================================
    # HYBRID RANKING with all scores
    # ============================================
    from .services.hybrid_memory_ranker import rank_memories
    import numpy as np
    from datetime import datetime
    
    # Prepare memories for hybrid ranking
    memories_list = list(all_memories.values())
    
    for mem in memories_list:
        # Ensure all score fields exist
        if "rag_score" not in mem:
            mem["rag_score"] = 0.0
        if "resonance_score" not in mem:
            mem["resonance_score"] = 0.0
        if "proximity_score" not in mem:
            # Calculate if we have XYZ
            if mem.get("xyz") and all(x is not None for x in mem["xyz"]):
                mem["proximity_score"] = ResonanceHasher.calculate_proximity_score(query_xyz, tuple(mem["xyz"]))
            else:
                mem["proximity_score"] = 0.0
        if "anchor_score" not in mem:
            mem["anchor_score"] = 0.0
        if "recency_score" not in mem:
            # Calculate recency
            if mem.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(mem["timestamp"].replace('Z', '+00:00'))
                    age_days = (datetime.now(ts.tzinfo) - ts).days
                    mem["recency_score"] = np.exp(-age_days / 30.0)
                except:
                    mem["recency_score"] = 0.5
            else:
                mem["recency_score"] = 0.5
        
        # Calculate Layer 5: Resonance Function score
        if mem.get("xyz") and all(x is not None for x in mem["xyz"]):
            mem_resonance = ResonanceHasher.calculate_resonance_function(tuple(mem["xyz"]))
            resonance_diff = abs(query_resonance - mem_resonance)
            mem["resonance_function_score"] = max(0.0, 1.0 - (resonance_diff / 6.0))
        else:
            mem["resonance_function_score"] = 0.0
        
        # Calculate Layer 4: Anchor Energy
        if mem.get("xyz") and all(x is not None for x in mem["xyz"]):
            mem["anchor_energy"] = ResonanceHasher.calculate_anchor_energy(
                np.array(query_xyz),
                np.array(mem["xyz"])
            )
        else:
            mem["anchor_energy"] = 0.0
        
        # Apply Magnetic Pull if enabled
        if request.apply_magnetic_pull:
            mem["magnetic_score"] = resonance_hasher.magnetic_pull(mem.get("resonance_score", 0.0))
        else:
            mem["magnetic_score"] = mem.get("resonance_score", 0.0)
    
    # Apply hybrid ranking
    ranked_memories = rank_memories(memories_list)
    
    # Take top results
    top_memories = ranked_memories[:request.limit]
    
    # Convert to response format
    response_memories = []
    for mem in top_memories:
        response_memories.append(HashSphereMemory(
            id=mem["id"],
            content=mem["content"],
            type=mem.get("type", "memory"),
            hash=mem.get("hash"),
            xyz=mem.get("xyz"),
            hybrid_score=mem.get("hybrid_score", 0.0),
            rag_score=mem.get("rag_score", 0.0),
            resonance_score=mem.get("resonance_score", 0.0),
            proximity_score=mem.get("proximity_score", 0.0),
            anchor_score=mem.get("anchor_score", 0.0),
            recency_score=mem.get("recency_score", 0.0),
            anchor_energy=mem.get("anchor_energy", 0.0),
            resonance_function_score=mem.get("resonance_function_score", 0.0),
            magnetic_score=mem.get("magnetic_score", 0.0),
            gravity_force=mem.get("gravity_force", 0.0),
            timestamp=mem.get("timestamp"),
        ))
    
    extraction_time = (time.perf_counter() - start_time) * 1000
    
    return HashSphereExtractResponse(
        memories=response_memories,
        query=request.query,
        query_hash=query_hash,
        query_xyz=list(query_xyz),
        query_resonance=query_resonance,
        total_found=len(response_memories),
        extraction_methods_used=methods_used,
        extraction_time_ms=extraction_time,
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

class ProjectSummaryResponse(BaseModel):
    project_id: str
    name: str
    file_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProjectsResponse(BaseModel):
    projects: List[ProjectSummaryResponse]
    count: int


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

class ProjectFileResponse(BaseModel):
    path: str
    type: str  # 'file' or 'directory'
    size: Optional[int] = None
    content: Optional[str] = None
    language: Optional[str] = None


class ProjectFilesResponse(BaseModel):
    project_id: str
    files: List[ProjectFileResponse]
    total: int


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


# ============================================
# HASH SPHERE ENDPOINTS
# ============================================

class HashRequest(BaseModel):
    text: str
    context: Optional[str] = None


class HashResponse(BaseModel):
    hash: str
    energy_score: float
    spin_score: float
    anchors: List[str]
    xyz: List[float]


class ResonanceRequest(BaseModel):
    hash1: str
    hash2: str


class ResonanceResponse(BaseModel):
    resonance_score: float
    boosted_score: float
    hash1: str
    hash2: str


class AnchorCreateRequest(BaseModel):
    anchor_text: str
    context: Optional[str] = None
    importance_score: float = 0.5


class AnchorResponse(BaseModel):
    id: str
    anchor_text: str
    anchor_hash: str
    context: str
    importance_score: float
    user_id: Optional[str] = None
    xyz_x: Optional[float] = None
    xyz_y: Optional[float] = None
    xyz_z: Optional[float] = None
    anchor_type: Optional[str] = None
    resonance_score: Optional[float] = None
    created_at: Optional[str] = None
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    normalized_resonance: Optional[float] = None
    anchor_energy: Optional[float] = None
    spin_x: Optional[float] = None
    spin_y: Optional[float] = None
    spin_z: Optional[float] = None
    spin_magnitude: Optional[float] = None
    meaning_score: Optional[float] = None
    intensity_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    meaning_hash: Optional[str] = None
    energy_hash: Optional[str] = None
    spin_hash: Optional[str] = None
    universe_id: Optional[str] = None
    cluster_name: Optional[str] = None


class AnchorSearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: int = 10


@router.post("/hash-sphere/hash", response_model=HashResponse)
async def hash_text(request: HashRequest):
    """Hash text using Hash Sphere."""
    hash_value = resonance_hasher.hash_text(request.text, request.context)
    energy = resonance_hasher._calculate_energy(request.text)
    spin = resonance_hasher._calculate_spin(request.text)
    anchors = resonance_hasher.extract_anchors(request.text)
    xyz = resonance_hasher.to_xyz(hash_value)
    
    return HashResponse(
        hash=hash_value,
        energy_score=energy,
        spin_score=spin,
        anchors=anchors,
        xyz=list(xyz)
    )


@router.post("/hash-sphere/resonance", response_model=ResonanceResponse)
async def calculate_resonance(request: ResonanceRequest):
    """Calculate resonance between two hashes."""
    resonance = resonance_hasher.calculate_resonance(request.hash1, request.hash2)
    boosted = resonance_hasher.magnetic_pull(resonance)
    
    return ResonanceResponse(
        resonance_score=resonance,
        boosted_score=boosted,
        hash1=request.hash1,
        hash2=request.hash2
    )


@router.post("/hash-sphere/anchors", response_model=AnchorResponse)
async def create_anchor(
    request: AnchorCreateRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a memory anchor."""
    import uuid

    raw_user_id = req.headers.get("x-user-id")
    raw_org_id = req.headers.get("x-org-id")

    user_uuid: Optional[uuid.UUID] = None
    org_uuid: Optional[uuid.UUID] = None
    try:
        if raw_user_id:
            user_uuid = uuid.UUID(raw_user_id)
    except Exception:
        user_uuid = None
    try:
        if raw_org_id:
            org_uuid = uuid.UUID(raw_org_id)
    except Exception:
        org_uuid = None
    
    anchor_hash = resonance_hasher.hash_text(request.anchor_text, request.context)
    xyz = resonance_hasher.to_xyz(anchor_hash)
    anchor_id = str(uuid.uuid4())
    
    # Store as memory record with anchor metadata
    record = MemoryRecord(
        id=uuid.UUID(anchor_id),
        user_id=user_uuid,
        org_id=org_uuid,
        source="anchor",
        content=request.anchor_text,
        hash=anchor_hash,
        xyz_x=xyz[0] if xyz else None,
        xyz_y=xyz[1] if xyz else None,
        xyz_z=xyz[2] if xyz else None,
        resonance_score=request.importance_score,
        extra_metadata={
            "type": "anchor",
            "hash": anchor_hash,
            "context": request.context or "",
            "importance_score": request.importance_score,
            "anchor_type": "chat",
        }
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    
    return AnchorResponse(
        id=anchor_id,
        anchor_text=request.anchor_text,
        anchor_hash=anchor_hash,
        context=request.context or "",
        importance_score=request.importance_score,
        user_id=str(record.user_id) if record.user_id else (raw_user_id or None),
        xyz_x=record.xyz_x,
        xyz_y=record.xyz_y,
        xyz_z=record.xyz_z,
        anchor_type=record.extra_metadata.get("anchor_type") if record.extra_metadata else None,
        resonance_score=record.resonance_score,
        created_at=record.created_at.isoformat() if record.created_at else None,
    )


@router.get("/hash-sphere/anchors", response_model=List[AnchorResponse])
async def list_anchors(
    user_id: Optional[str] = None,
    limit: int = 50,
    req: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """List memory anchors."""
    import uuid

    if not user_id and req:
        user_id = req.headers.get("x-user-id")

    user_uuid: Optional[uuid.UUID] = None
    try:
        if user_id:
            user_uuid = uuid.UUID(user_id)
    except Exception:
        user_uuid = None

    stmt = select(MemoryRecord).order_by(MemoryRecord.created_at.desc()).limit(limit)
    
    if user_uuid:
        stmt = stmt.where(MemoryRecord.user_id == user_uuid)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    return [
        AnchorResponse(
            id=str(r.id),
            anchor_text=(decrypt_memory_content(r.content) if r.content else "")[:100],
            anchor_hash=r.hash or (r.extra_metadata.get("hash", "") if r.extra_metadata else ""),
            context=decrypt_memory_content(r.content) if r.content else "",
            importance_score=r.meaning_score if r.meaning_score is not None else (r.extra_metadata.get("importance_score", 0.5) if r.extra_metadata else 0.5),
            user_id=str(r.user_id) if r.user_id else None,
            xyz_x=r.xyz_x,
            xyz_y=r.xyz_y,
            xyz_z=r.xyz_z,
            anchor_type=r.source or (r.extra_metadata.get("anchor_type") if r.extra_metadata else None),
            resonance_score=r.resonance_score,
            created_at=r.created_at.isoformat() if r.created_at else None,
            sphere_r=r.sphere_r,
            sphere_phi=r.sphere_phi,
            sphere_theta=r.sphere_theta,
            normalized_resonance=r.normalized_resonance,
            anchor_energy=r.anchor_energy,
            spin_x=r.spin_x,
            spin_y=r.spin_y,
            spin_z=r.spin_z,
            spin_magnitude=r.spin_magnitude,
            meaning_score=r.meaning_score,
            intensity_score=r.intensity_score,
            sentiment_score=r.sentiment_score,
            meaning_hash=r.meaning_hash,
            energy_hash=r.energy_hash,
            spin_hash=r.spin_hash,
            universe_id=r.universe_id,
            cluster_name=r.cluster_name,
        )
        for r in records
    ]


@router.post("/hash-sphere/search", response_model=List[AnchorResponse])
async def search_anchors(
    request: AnchorSearchRequest,
    req: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """Search anchors by resonance."""
    import uuid

    search_user_id = request.user_id
    if not search_user_id and req:
        search_user_id = req.headers.get("x-user-id")

    search_user_uuid: Optional[uuid.UUID] = None
    try:
        if search_user_id:
            search_user_uuid = uuid.UUID(search_user_id)
    except Exception:
        search_user_uuid = None

    # Get all anchors
    stmt = select(MemoryRecord).where(MemoryRecord.source == "anchor")
    if search_user_uuid:
        stmt = stmt.where(MemoryRecord.user_id == search_user_uuid)
    result = await session.execute(stmt)
    records = result.scalars().all()

    record_map: Dict[str, MemoryRecord] = {str(r.id): r for r in records}
    
    # Convert to anchor format
    anchors = [
        {
            "id": str(r.id),
            "text": r.content,
            "hash": r.hash or (r.extra_metadata.get("hash", "") if r.extra_metadata else ""),
            "context": r.extra_metadata.get("context", "") if r.extra_metadata else "",
            "importance_score": r.extra_metadata.get("importance_score", 0.5) if r.extra_metadata else 0.5
        }
        for r in records
    ]
    
    # Rank by resonance
    ranked = memory_anchor_service.rank_by_resonance(request.query, anchors, request.limit)
    
    responses: List[AnchorResponse] = []
    for a in ranked:
        r = record_map.get(a["id"])
        responses.append(
            AnchorResponse(
                id=a["id"],
                anchor_text=a["text"],
                anchor_hash=a["hash"],
                context=a["context"],
                importance_score=a["importance_score"],
                user_id=str(r.user_id) if r and r.user_id else None,
                xyz_x=r.xyz_x if r else None,
                xyz_y=r.xyz_y if r else None,
                xyz_z=r.xyz_z if r else None,
                anchor_type=r.extra_metadata.get("anchor_type") if (r and r.extra_metadata) else None,
                resonance_score=r.resonance_score if r else None,
                created_at=r.created_at.isoformat() if (r and r.created_at) else None,
            )
        )

    return responses


# ============================================
# ARCHIVE ENDPOINTS
# Hash Sphere is immutable - use archive to hide anchors instead of delete
# ============================================

class ArchiveRequest(BaseModel):
    """Request to archive/unarchive a file or anchor."""
    file_path: Optional[str] = None
    anchor_id: Optional[str] = None
    project_id: Optional[str] = None


class ArchiveResponse(BaseModel):
    """Response for archive operations."""
    success: bool
    archived_count: int
    file_path: Optional[str] = None
    message: str


@router.post("/archive/file", response_model=ArchiveResponse)
async def archive_file(
    request: ArchiveRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Archive a file - sets is_archived=True on all anchors with this file_path.
    
    Hash Sphere is immutable - data stays but won't be loaded when archived.
    """
    from datetime import datetime
    
    user_id = req.headers.get("x-user-id")
    file_path = request.file_path
    
    if not file_path:
        return ArchiveResponse(
            success=False,
            archived_count=0,
            file_path=file_path,
            message="file_path is required"
        )
    
    # Find all memory records with this file_path in metadata
    stmt = select(MemoryRecord).where(
        MemoryRecord.extra_metadata.contains({"file_path": file_path})
    )
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    archived_count = 0
    for record in records:
        if record.extra_metadata:
            record.extra_metadata["is_archived"] = True
            record.extra_metadata["archived_at"] = datetime.now().isoformat()
            archived_count += 1
    
    await session.commit()
    
    return ArchiveResponse(
        success=True,
        archived_count=archived_count,
        file_path=file_path,
        message=f"Archived {archived_count} anchors for file: {file_path}"
    )


@router.post("/unarchive/file", response_model=ArchiveResponse)
async def unarchive_file(
    request: ArchiveRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Unarchive a file - sets is_archived=False to restore visibility."""
    user_id = req.headers.get("x-user-id")
    file_path = request.file_path
    
    if not file_path:
        return ArchiveResponse(
            success=False,
            archived_count=0,
            file_path=file_path,
            message="file_path is required"
        )
    
    # Find all archived memory records with this file_path
    stmt = select(MemoryRecord).where(
        MemoryRecord.extra_metadata.contains({"file_path": file_path, "is_archived": True})
    )
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    unarchived_count = 0
    for record in records:
        if record.extra_metadata:
            record.extra_metadata["is_archived"] = False
            record.extra_metadata["archived_at"] = None
            unarchived_count += 1
    
    await session.commit()
    
    return ArchiveResponse(
        success=True,
        archived_count=unarchived_count,
        file_path=file_path,
        message=f"Unarchived {unarchived_count} anchors for file: {file_path}"
    )


@router.get("/archived/files")
async def list_archived_files(
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """List all archived files for the current user."""
    user_id = req.headers.get("x-user-id")
    
    stmt = select(MemoryRecord).where(
        MemoryRecord.extra_metadata.contains({"is_archived": True})
    )
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    # Extract unique file paths
    file_paths = set()
    for record in records:
        if record.extra_metadata and record.extra_metadata.get("file_path"):
            file_paths.add(record.extra_metadata["file_path"])
    
    return {
        "archived_files": list(file_paths),
        "count": len(file_paths)
    }


# ============================================
# RAG COMPATIBILITY ENDPOINTS
# Frontend expects /rag/... paths - these map to memory operations
# ============================================

rag_router = APIRouter(prefix="/rag", tags=["rag"])

from fastapi import UploadFile, File


class RAGFileUploadResponse(BaseModel):
    """Response for file upload."""
    id: str
    filename: str
    content_type: str
    size: int
    memories_created: int
    chunks: int


class RAGMemoryCreateRequest(BaseModel):
    """RAG memory creation request - matches old backend."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    is_shared: bool = False
    is_public: bool = False
    shared_with: Optional[List[str]] = None
    language: Optional[str] = None


class RAGMemoryResponse(BaseModel):
    """RAG memory response - matches old backend."""
    id: str
    content: str
    hash: Optional[str] = None
    xyz: Optional[List[float]] = None
    cluster: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: str
    is_shared: Optional[bool] = False
    shared_with: Optional[List[str]] = None
    is_public: Optional[bool] = False


class RAGMemoryUpdateRequest(BaseModel):
    """RAG memory update request."""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGAskRequest(BaseModel):
    """RAG ask request - matches old backend."""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    use_memory: bool = True
    provider: Optional[str] = None


class RAGAskResponse(BaseModel):
    """RAG ask response - matches old backend."""
    response: str
    sources: List[Dict[str, Any]]
    validity: float
    entropy: float
    evidence_graph: Dict[str, Any]
    context_used: bool
    conversation_id: str


class RAGConversationResponse(BaseModel):
    """RAG conversation response."""
    id: str
    role: str
    content: str
    provider: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    validity: Optional[float] = None
    created_at: str


from fastapi import Request, HTTPException
from uuid import UUID
import uuid as uuid_module


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user_id from request headers (set by gateway)."""
    return request.headers.get("x-user-id")


def _get_org_id(request: Request) -> Optional[str]:
    """Extract org_id from request headers (set by gateway)."""
    return request.headers.get("x-org-id")


@rag_router.post("/memories", response_model=RAGMemoryResponse, status_code=201)
async def create_rag_memory(
    payload: RAGMemoryCreateRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a memory/document for the current user - RAG compatibility."""
    user_id = _get_user_id(request)
    org_id = _get_org_id(request)
    
    # ============================================
    # CHECK RAG DOCUMENT LIMIT (GTM Critical)
    # ============================================
    if user_id:
        import httpx
        from sqlalchemy import func
        
        # Get user's plan — check forwarded headers first, fallback to billing service
        user_plan = "developer"
        unlimited_credits = request.headers.get("x-unlimited-credits", "").lower() in ("true", "1")
        header_plan = request.headers.get("x-user-plan", "").lower()
        header_role = request.headers.get("x-user-role", "").lower()
        
        if unlimited_credits or header_role in ("platform_owner", "owner", "admin"):
            user_plan = "unlimited"
        elif header_plan in ("enterprise", "plus", "professional", "unlimited"):
            user_plan = header_plan
        else:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://billing_service:8001/billing/subscription",
                        headers={"x-user-id": user_id},
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        user_plan = data.get("plan", "developer").lower()
                        if data.get("is_dev"):
                            user_plan = "unlimited"
            except Exception:
                pass  # Default to developer plan
        
        # Plan limits for RAG documents
        rag_limits = {
            "developer": 5, "free": 5,
            "plus": 100, "professional": 100,
            "enterprise": -1, "unlimited": -1,
        }
        max_docs = rag_limits.get(user_plan, 5)
        
        # Count existing RAG documents
        if max_docs > 0:
            count_result = await session.execute(
                select(func.count(MemoryRecord.id)).where(
                    MemoryRecord.user_id == user_id,
                    MemoryRecord.source == "rag"
                )
            )
            current_count = count_result.scalar() or 0
            
            if current_count >= max_docs:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rag_document_limit_exceeded",
                        "message": f"RAG document limit reached ({current_count}/{max_docs}). Upgrade to Plus for 100 documents.",
                        "used": current_count,
                        "limit": max_docs,
                        "upgrade_url": "/pricing"
                    }
                )
    
    # Generate hash and coordinates
    hash_value = resonance_hasher.hash_text(payload.content)
    xyz = resonance_hasher.to_xyz(hash_value)
    
    # Build metadata
    metadata = payload.metadata or {}
    metadata["is_shared"] = payload.is_shared
    metadata["is_public"] = payload.is_public
    if payload.shared_with:
        metadata["shared_with"] = payload.shared_with
    if payload.language:
        metadata["language"] = payload.language
    
    # Create memory record
    record = MemoryRecord(
        user_id=user_id,
        org_id=org_id,
        source="rag",
        content=payload.content,
        hash=hash_value,
        xyz_x=xyz[0] if xyz else None,
        xyz_y=xyz[1] if xyz else None,
        xyz_z=xyz[2] if xyz else None,
        extra_metadata=metadata,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    
    # Generate embedding
    embeddings = await embeddings_generator.generate([payload.content])
    if embeddings:
        embedding_record = MemoryEmbedding(
            memory_id=record.id,
            user_id=user_id,
            org_id=org_id,
            embedding=embeddings[0],
        )
        session.add(embedding_record)
        await session.commit()
    
    return RAGMemoryResponse(
        id=str(record.id),
        content=record.content,
        hash=record.hash,
        xyz=[record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x else None,
        cluster=None,
        metadata=record.extra_metadata or {},
        created_at=record.created_at.isoformat(),
        is_shared=metadata.get("is_shared", False),
        shared_with=metadata.get("shared_with"),
        is_public=metadata.get("is_public", False),
    )


@rag_router.get("/memories", response_model=List[RAGMemoryResponse])
async def list_rag_memories(
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List user's memories - RAG compatibility."""
    user_id = _get_user_id(request)
    
    stmt = select(MemoryRecord).where(
        MemoryRecord.source == "rag"
    ).order_by(MemoryRecord.created_at.desc()).limit(limit)
    
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    return [
        RAGMemoryResponse(
            id=str(r.id),
            content=r.content,
            hash=r.hash,
            xyz=[r.xyz_x, r.xyz_y, r.xyz_z] if r.xyz_x else None,
            cluster=None,
            metadata=r.extra_metadata or {},
            created_at=r.created_at.isoformat(),
            is_shared=r.extra_metadata.get("is_shared", False) if r.extra_metadata else False,
            shared_with=r.extra_metadata.get("shared_with") if r.extra_metadata else None,
            is_public=r.extra_metadata.get("is_public", False) if r.extra_metadata else False,
        )
        for r in records
    ]


@rag_router.get("/memories/{memory_id}", response_model=RAGMemoryResponse)
async def get_rag_memory(
    memory_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific memory - RAG compatibility."""
    user_id = _get_user_id(request)
    
    stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return RAGMemoryResponse(
        id=str(record.id),
        content=record.content,
        hash=record.hash,
        xyz=[record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x else None,
        cluster=None,
        metadata=record.extra_metadata or {},
        created_at=record.created_at.isoformat(),
        is_shared=record.extra_metadata.get("is_shared", False) if record.extra_metadata else False,
        shared_with=record.extra_metadata.get("shared_with") if record.extra_metadata else None,
        is_public=record.extra_metadata.get("is_public", False) if record.extra_metadata else False,
    )


@rag_router.put("/memories/{memory_id}", response_model=RAGMemoryResponse)
async def update_rag_memory(
    memory_id: str,
    payload: RAGMemoryUpdateRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Update a memory - RAG compatibility."""
    user_id = _get_user_id(request)
    
    stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if payload.content:
        record.content = payload.content
        # Regenerate hash and coordinates
        hash_value = resonance_hasher.hash_text(payload.content)
        xyz = resonance_hasher.to_xyz(hash_value)
        record.hash = hash_value
        record.xyz_x = xyz[0] if xyz else None
        record.xyz_y = xyz[1] if xyz else None
        record.xyz_z = xyz[2] if xyz else None
        
        # Regenerate embedding
        embeddings = await embeddings_generator.generate([payload.content])
        if embeddings:
            # Delete old embedding
            del_stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == record.id)
            del_result = await session.execute(del_stmt)
            old_emb = del_result.scalar_one_or_none()
            if old_emb:
                await session.delete(old_emb)
            
            # Create new embedding
            embedding_record = MemoryEmbedding(
                memory_id=record.id,
                user_id=user_id,
                embedding=embeddings[0],
            )
            session.add(embedding_record)
    
    if payload.metadata:
        record.extra_metadata = {**(record.extra_metadata or {}), **payload.metadata}
    
    await session.commit()
    await session.refresh(record)
    
    return RAGMemoryResponse(
        id=str(record.id),
        content=record.content,
        hash=record.hash,
        xyz=[record.xyz_x, record.xyz_y, record.xyz_z] if record.xyz_x else None,
        cluster=None,
        metadata=record.extra_metadata or {},
        created_at=record.created_at.isoformat(),
        is_shared=record.extra_metadata.get("is_shared", False) if record.extra_metadata else False,
        shared_with=record.extra_metadata.get("shared_with") if record.extra_metadata else None,
        is_public=record.extra_metadata.get("is_public", False) if record.extra_metadata else False,
    )


@rag_router.delete("/memories/{memory_id}")
async def delete_rag_memory(
    memory_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a RAG memory only. Hash Sphere memories (source != 'rag') are immutable
    and CANNOT be deleted — doing so would break the decentralized coordinate system."""
    user_id = _get_user_id(request)
    
    # Find the memory record first
    stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # CRITICAL: Only allow deletion of RAG source memories
    # Hash Sphere memories (chat, workflow, cognitive, etc.) are IMMUTABLE
    if record.source != "rag":
        raise HTTPException(
            status_code=403,
            detail="Hash Sphere memories are immutable and cannot be deleted. "
                   "Only user-created RAG memories can be removed."
        )
    
    # Delete embedding first
    emb_stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
    emb_result = await session.execute(emb_stmt)
    embedding = emb_result.scalar_one_or_none()
    if embedding:
        await session.delete(embedding)
    
    # Safe to delete — this is a user-created RAG memory
    await session.delete(record)
    await session.commit()
    
    return {"status": "deleted", "id": memory_id}


@rag_router.post("/ask", response_model=RAGAskResponse)
async def ask_with_rag(
    payload: RAGAskRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Ask a question with RAG retrieval - RAG compatibility."""
    import httpx
    from .config import settings
    
    user_id = _get_user_id(request)
    conversation_id = payload.conversation_id or str(uuid_module.uuid4())
    
    # Retrieve relevant memories using vector search
    sources = []
    if payload.use_memory:
        query_embeddings = await embeddings_generator.generate([payload.query])
        if query_embeddings:
            query_embedding = query_embeddings[0]
            
            # Get embeddings for user
            stmt = select(MemoryEmbedding)
            if user_id:
                stmt = stmt.where(MemoryEmbedding.user_id == user_id)
            
            result = await session.execute(stmt)
            embeddings_list = result.scalars().all()
            
            # Calculate similarities
            similarities = []
            for emb in embeddings_list:
                similarity = embeddings_generator.cosine_similarity(query_embedding, emb.embedding)
                similarities.append((emb.memory_id, similarity))
            
            # Sort and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memory_ids = [mid for mid, _ in similarities[:payload.top_k]]
            similarity_map = {mid: sim for mid, sim in similarities[:payload.top_k]}
            
            # Fetch memory records
            if top_memory_ids:
                mem_stmt = select(MemoryRecord).where(MemoryRecord.id.in_(top_memory_ids))
                mem_result = await session.execute(mem_stmt)
                records = mem_result.scalars().all()
                
                sources = [
                    {
                        "id": str(r.id),
                        "content": r.content[:500],  # Truncate for response
                        "score": similarity_map.get(r.id, 0.0),
                        "hash": r.hash,
                        "xyz": [r.xyz_x, r.xyz_y, r.xyz_z] if r.xyz_x else None,
                        "metadata": r.extra_metadata or {},
                    }
                    for r in records
                ]
    
    # Build context from sources
    context = "\n\n".join([s["content"] for s in sources]) if sources else ""
    
    # Call LLM service for response generation
    response_text = ""
    validity = 0.5
    
    try:
        # Build RAG prompt with context
        system_prompt = """You are a helpful AI assistant with access to the user's memory bank. 
Use the provided context from their memories to answer questions accurately and helpfully.
If the context doesn't contain relevant information, say so and provide a general response."""
        
        user_message = payload.query
        if context:
            user_message = f"""Context from user's memories:
---
{context[:4000]}
---

User question: {payload.query}

Please answer based on the context provided. If the context is relevant, cite it. If not, provide a helpful general response."""
        
        # Call LLM service
        async with httpx.AsyncClient(timeout=60.0) as client:
            llm_response = await client.post(
                f"{settings.LLM_SERVICE_URL}/llm/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "model": payload.model or "gpt-4-turbo-preview",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "user_id": user_id,
                },
                headers={"x-user-id": user_id} if user_id else {},
            )
            
            if llm_response.status_code == 200:
                llm_data = llm_response.json()
                if llm_data.get("choices"):
                    response_text = llm_data["choices"][0].get("message", {}).get("content", "")
                    validity = 0.9 if sources else 0.7
            else:
                # Fallback if LLM service fails
                response_text = f"Based on {len(sources)} relevant memories, here's what I found related to your query: '{payload.query}'"
                if context:
                    response_text = f"Context from your memories:\n{context[:1000]}...\n\nBased on this context, I can help answer your question about: {payload.query}"
    except Exception as e:
        # Fallback response if LLM service is unavailable
        response_text = f"Based on {len(sources)} relevant memories, here's what I found related to your query: '{payload.query}'"
        if context:
            response_text = f"Context from your memories:\n{context[:1000]}...\n\nBased on this context, I can help answer your question about: {payload.query}"
        validity = 0.6 if sources else 0.4
    
    return RAGAskResponse(
        response=response_text,
        sources=sources,
        validity=validity,
        entropy=0.2,
        evidence_graph={"nodes": [], "edges": []},
        context_used=bool(sources),
        conversation_id=conversation_id,
    )


@rag_router.get("/conversations")
async def list_rag_conversations(
    request: Request,
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """List user's conversations - RAG compatibility."""
    user_id = _get_user_id(request)
    
    # Get unique conversation IDs from memory records
    stmt = select(MemoryRecord.extra_metadata).where(
        MemoryRecord.source == "rag_conversation"
    ).order_by(MemoryRecord.created_at.desc()).limit(limit)
    
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    # Extract unique conversation IDs
    conversation_ids = list(set(
        r.get("conversation_id") for r in records if r and r.get("conversation_id")
    ))
    
    return conversation_ids[:limit]


@rag_router.get("/conversations/{conversation_id}", response_model=List[RAGConversationResponse])
async def get_rag_conversation(
    conversation_id: str,
    request: Request,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
):
    """Get conversation history - RAG compatibility."""
    user_id = _get_user_id(request)
    
    # For now, return empty list - conversations are stored in chat_service
    # This is a stub for frontend compatibility
    return []


@rag_router.delete("/conversations/{conversation_id}")
async def delete_rag_conversation(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a conversation - RAG compatibility."""
    # Stub for frontend compatibility
    return {"status": "deleted", "conversation_id": conversation_id}


@rag_router.put("/conversations/{conversation_id}")
async def update_rag_conversation(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Update conversation title - RAG compatibility."""
    # Stub for frontend compatibility
    return {"status": "updated", "conversation_id": conversation_id}


@rag_router.post("/files/upload", response_model=RAGFileUploadResponse)
async def upload_rag_file(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a file and create memories from its content.
    
    Supports: .txt, .md, .pdf, .docx, .csv, .json
    Files are chunked and each chunk becomes a memory with embedding.
    """
    import uuid as uuid_module
    from datetime import datetime
    
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Use document loaders for proper parsing (PDF, DOCX, CSV, HTML, etc.)
    try:
        text_content = parse_document(content, file.filename or "file.txt", file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to parse file: {e}")
    
    # Chunk with overlap for better RAG retrieval
    chunks = chunk_text(text_content, max_chars=1000, overlap=100)
    
    # Create memories for each chunk
    file_id = str(uuid_module.uuid4())
    memories_created = 0
    
    for i, chunk in enumerate(chunks):
        if not chunk or len(chunk) < 10:
            continue
            
        memory_id = str(uuid_module.uuid4())
        record = MemoryRecord(
            id=memory_id,
            user_id=user_id,
            source="file_upload",
            content=chunk,
            extra_metadata={
                "file_id": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        session.add(record)
        
        # Generate embedding
        embeddings = await embeddings_generator.generate([chunk])
        if embeddings:
            embedding_record = MemoryEmbedding(
                memory_id=memory_id,
                user_id=user_id,
                embedding=embeddings[0],
            )
            session.add(embedding_record)
        
        memories_created += 1
    
    await session.commit()
    
    return RAGFileUploadResponse(
        id=file_id,
        filename=file.filename or "unknown",
        content_type=file.content_type or "text/plain",
        size=file_size,
        memories_created=memories_created,
        chunks=len(chunks),
    )


# ============================================
# PUBLIC HASH SPHERE ENDPOINTS
# No authentication required - for public pages
# ============================================

public_router = APIRouter(prefix="/public", tags=["public"])


class HashSphereTokenRequest(BaseModel):
    """Request for Hash Sphere token."""
    is_owner: bool = False


class HashSphereTokenResponse(BaseModel):
    """Hash Sphere token response."""
    token: str
    expires_at: str
    expires_in_hours: int


@public_router.post("/hash-sphere/token", response_model=HashSphereTokenResponse)
async def get_hash_sphere_token(
    payload: HashSphereTokenRequest,
):
    """Get a Hash Sphere access token for public pages.
    
    - Owner token: unlimited memory, 30-day expiration
    - Guest token: limited memory, 1-hour expiration
    """
    import secrets
    from datetime import datetime, timedelta
    
    if payload.is_owner:
        expires_in_hours = 24 * 30  # 30 days
    else:
        expires_in_hours = 1  # 1 hour
    
    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    token = f"hs_{secrets.token_urlsafe(32)}"
    
    return HashSphereTokenResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        expires_in_hours=expires_in_hours,
    )


# ============================================
# AGENT MEMORY ENDPOINTS
# Advanced memory operations for Agent OS
# ============================================

agent_memory_router = APIRouter(prefix="/memory", tags=["agent-memory"])


class AgentMemoryCreateRequest(BaseModel):
    agent_id: str
    type: str  # short-term, episodic, semantic, strategic
    content: str
    importance: float = 0.5
    tags: Optional[List[str]] = None
    source: Optional[str] = None


class AgentMemoryResponse(BaseModel):
    id: str
    agent_id: str
    type: str
    content: str
    importance: float
    timestamp: str
    tokens: Optional[int] = None
    tags: Optional[List[str]] = None


class SemanticSearchRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5
    threshold: float = 0.75
    memory_types: Optional[List[str]] = None


class SemanticSearchResult(BaseModel):
    memory: AgentMemoryResponse
    score: float
    relevance: str  # high, medium, low


class SemanticSearchResponse(BaseModel):
    results: List[SemanticSearchResult]
    query: str
    total_found: int


class ConsolidateRequest(BaseModel):
    agent_id: str
    threshold: int = 100


class MemorySettingsRequest(BaseModel):
    agent_id: str
    settings: Dict[str, Any]


class MemoryClusterResponse(BaseModel):
    id: str
    name: str
    memories: List[str]
    coherence: float


class MemoryAnalyticsResponse(BaseModel):
    total_memories: int
    by_type: Dict[str, int]
    avg_importance: float
    storage_used: float
    retrieval_latency: float
    consolidation_rate: float


@agent_memory_router.post("/", response_model=AgentMemoryResponse)
async def create_agent_memory(
    payload: AgentMemoryCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a new agent memory with embedding."""
    import uuid
    from datetime import datetime
    
    memory_id = str(uuid.uuid4())
    
    # Store in database
    record = MemoryRecord(
        id=memory_id,
        user_id=payload.agent_id,
        source=payload.type,
        content=payload.content,
        extra_metadata={
            "importance": payload.importance,
            "tags": payload.tags or [],
            "agent_source": payload.source,
        },
    )
    session.add(record)
    await session.commit()
    
    # Generate embedding
    embeddings = await embeddings_generator.generate([payload.content])
    if embeddings:
        embedding_record = MemoryEmbedding(
            memory_id=memory_id,
            user_id=payload.agent_id,
            embedding=embeddings[0],
        )
        session.add(embedding_record)
        await session.commit()
    
    return AgentMemoryResponse(
        id=memory_id,
        agent_id=payload.agent_id,
        type=payload.type,
        content=payload.content,
        importance=payload.importance,
        timestamp=datetime.utcnow().isoformat(),
        tokens=len(payload.content.split()),
        tags=payload.tags,
    )


@agent_memory_router.delete("/{memory_id}")
async def delete_agent_memory(
    memory_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete an agent memory."""
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


@agent_memory_router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Perform semantic search across agent memories."""
    # Generate query embedding
    query_embeddings = await embeddings_generator.generate([payload.query])
    results = []
    
    if query_embeddings:
        query_embedding = query_embeddings[0]
        
        # Get embeddings for agent
        stmt = select(MemoryEmbedding).where(MemoryEmbedding.user_id == payload.agent_id)
        result = await session.execute(stmt)
        embeddings = result.scalars().all()
        
        # Calculate similarities
        similarities = []
        for emb in embeddings:
            similarity = embeddings_generator.cosine_similarity(query_embedding, emb.embedding)
            if similarity >= payload.threshold:
                similarities.append((emb.memory_id, similarity))
        
        # Sort and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:payload.top_k]
        
        # Fetch memory records
        for memory_id, score in top_results:
            stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            
            if record:
                # Filter by memory type if specified
                if payload.memory_types and record.source not in payload.memory_types:
                    continue
                
                relevance = "high" if score >= 0.85 else "medium" if score >= 0.7 else "low"
                
                results.append(SemanticSearchResult(
                    memory=AgentMemoryResponse(
                        id=str(record.id),
                        agent_id=payload.agent_id,
                        type=record.source,
                        content=record.content,
                        importance=record.extra_metadata.get("importance", 0.5) if record.extra_metadata else 0.5,
                        timestamp=record.created_at.isoformat() if record.created_at else "",
                        tokens=len(record.content.split()),
                        tags=record.extra_metadata.get("tags", []) if record.extra_metadata else [],
                    ),
                    score=score,
                    relevance=relevance,
                ))
    
    return SemanticSearchResponse(
        results=results,
        query=payload.query,
        total_found=len(results),
    )


@agent_memory_router.post("/consolidate")
async def consolidate_memories(
    payload: ConsolidateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Consolidate short-term memories into long-term storage."""
    # Get short-term memories for agent
    stmt = select(MemoryRecord).where(
        MemoryRecord.user_id == payload.agent_id,
        MemoryRecord.source == "short-term"
    ).order_by(MemoryRecord.created_at.asc())
    
    result = await session.execute(stmt)
    short_term_memories = result.scalars().all()
    
    consolidated_count = 0
    
    # If we have more than threshold, consolidate oldest ones
    if len(short_term_memories) > payload.threshold:
        to_consolidate = short_term_memories[:len(short_term_memories) - payload.threshold]
        
        for memory in to_consolidate:
            # Move to episodic memory
            memory.source = "episodic"
            if memory.extra_metadata:
                memory.extra_metadata["consolidated_from"] = "short-term"
            consolidated_count += 1
        
        await session.commit()
    
    return {
        "status": "completed",
        "agent_id": payload.agent_id,
        "consolidated_count": consolidated_count,
        "remaining_short_term": len(short_term_memories) - consolidated_count,
    }


@agent_memory_router.get("/export/{agent_id}")
async def export_memories(
    agent_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Export all memories for an agent."""
    stmt = select(MemoryRecord).where(MemoryRecord.user_id == agent_id)
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    return {
        "agent_id": agent_id,
        "memories": [
            {
                "id": str(r.id),
                "type": r.source,
                "content": r.content,
                "metadata": r.extra_metadata,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
        "total": len(records),
    }


@agent_memory_router.post("/import")
async def import_memories(
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
):
    """Import memories for an agent."""
    import uuid
    
    agent_id = payload.get("agent_id")
    memories = payload.get("memories", [])
    imported_count = 0
    
    for mem in memories:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=agent_id,
            source=mem.get("type", "imported"),
            content=mem.get("content", ""),
            extra_metadata=mem.get("metadata", {}),
        )
        session.add(record)
        
        # Generate embedding
        embeddings = await embeddings_generator.generate([mem.get("content", "")])
        if embeddings:
            embedding_record = MemoryEmbedding(
                memory_id=record.id,
                user_id=agent_id,
                embedding=embeddings[0],
            )
            session.add(embedding_record)
        
        imported_count += 1
    
    await session.commit()
    
    return {
        "status": "completed",
        "agent_id": agent_id,
        "imported_count": imported_count,
    }


@agent_memory_router.put("/settings")
async def update_memory_settings(
    payload: MemorySettingsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Update memory settings for an agent."""
    # Store settings in metadata or dedicated table
    # For now, return success
    return {
        "status": "updated",
        "agent_id": payload.agent_id,
        "settings": payload.settings,
    }


@agent_memory_router.get("/clusters/{agent_id}", response_model=List[MemoryClusterResponse])
async def get_memory_clusters(
    agent_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get memory clusters for an agent using embedding similarity."""
    stmt = select(MemoryRecord).where(MemoryRecord.user_id == agent_id)
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    # Simple clustering by memory type
    clusters = {}
    for r in records:
        cluster_name = f"{r.source.replace('-', ' ').title()} Memories"
        if cluster_name not in clusters:
            clusters[cluster_name] = {
                "id": f"cl-{r.source}",
                "name": cluster_name,
                "memories": [],
                "coherence": 0.85,
            }
        clusters[cluster_name]["memories"].append(str(r.id))
    
    return [
        MemoryClusterResponse(**cluster)
        for cluster in clusters.values()
    ]


@agent_memory_router.get("/analytics/{agent_id}", response_model=MemoryAnalyticsResponse)
async def get_memory_analytics(
    agent_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get memory analytics for an agent."""
    stmt = select(MemoryRecord).where(MemoryRecord.user_id == agent_id)
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    by_type = {}
    total_importance = 0
    
    for r in records:
        by_type[r.source] = by_type.get(r.source, 0) + 1
        if r.extra_metadata and "importance" in r.extra_metadata:
            total_importance += r.extra_metadata["importance"]
        else:
            total_importance += 0.5
    
    avg_importance = total_importance / len(records) if records else 0
    
    return MemoryAnalyticsResponse(
        total_memories=len(records),
        by_type=by_type,
        avg_importance=avg_importance,
        storage_used=len(records) * 0.5,  # Estimate 0.5KB per memory
        retrieval_latency=23,  # Simulated
        consolidation_rate=0.85,  # Simulated
    )


# ============================================================================
# MEMORY PANEL - 3D VISUALIZATION ENDPOINTS
# ============================================================================

class MemoryNodeResponse(BaseModel):
    """Memory node for 3D visualization."""
    id: str
    content: str
    title: Optional[str] = None
    x: float
    y: float
    z: float
    layer: str = "active"
    cluster: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    tags: List[str] = []
    created_at: str


class MemoryEdgeResponse(BaseModel):
    """Edge between memories based on similarity."""
    source: str
    target: str
    weight: float


class MemoryUniverseResponse(BaseModel):
    """Full memory universe for 3D visualization."""
    nodes: List[MemoryNodeResponse]
    edges: List[MemoryEdgeResponse]
    clusters: List[Dict[str, Any]]
    stats: Dict[str, Any]


@rag_router.get("/universe")
async def get_memory_universe(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Get all user memories with XYZ coordinates for 3D visualization.
    Uses real ResonanceHasher coordinates from embeddings.
    """
    user_id = _get_user_id(request)
    org_id = _get_org_id(request)
    
    # Get all memories for user
    stmt = select(MemoryRecord).where(
        MemoryRecord.user_id == user_id,
        MemoryRecord.org_id == org_id
    ).order_by(MemoryRecord.created_at.desc()).limit(500)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    nodes = []
    clusters_map = {}
    
    for r in records:
        # Get XYZ coordinates (already computed by ResonanceHasher)
        x = float(r.xyz_x) if r.xyz_x is not None else 0.0
        y = float(r.xyz_y) if r.xyz_y is not None else 0.0
        z = float(r.xyz_z) if r.xyz_z is not None else 0.0
        
        # Scale coordinates for visualization (0-1 range to -200 to 200)
        x = (x - 0.5) * 400
        y = (y - 0.5) * 400
        z = (z - 0.5) * 400
        
        metadata = r.extra_metadata or {}
        layer = metadata.get("layer", "active")
        cluster = metadata.get("cluster", "default")
        importance = metadata.get("importance", 0.5)
        tags = metadata.get("tags", [])
        access_count = metadata.get("access_count", 0)
        title = metadata.get("title", r.content[:50] if r.content else "Untitled")
        
        # Track clusters
        if cluster not in clusters_map:
            clusters_map[cluster] = {"name": cluster, "count": 0, "center": [0, 0, 0]}
        clusters_map[cluster]["count"] += 1
        clusters_map[cluster]["center"][0] += x
        clusters_map[cluster]["center"][1] += y
        clusters_map[cluster]["center"][2] += z
        
        nodes.append(MemoryNodeResponse(
            id=str(r.id),
            content=r.content[:500] if r.content else "",
            title=title,
            x=x,
            y=y,
            z=z,
            layer=layer,
            cluster=cluster,
            importance=importance,
            access_count=access_count,
            tags=tags if isinstance(tags, list) else [],
            created_at=r.created_at.isoformat() if r.created_at else "",
        ))
    
    # Calculate cluster centers
    clusters = []
    for name, data in clusters_map.items():
        if data["count"] > 0:
            data["center"] = [c / data["count"] for c in data["center"]]
        clusters.append(data)
    
    # Generate edges based on proximity (memories close in 3D space)
    edges = []
    for i, n1 in enumerate(nodes[:100]):  # Limit for performance
        for n2 in nodes[i+1:100]:
            dist = ((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2) ** 0.5
            if dist < 100:  # Only connect nearby memories
                weight = 1.0 - (dist / 100)
                edges.append(MemoryEdgeResponse(
                    source=n1.id,
                    target=n2.id,
                    weight=weight
                ))
    
    stats = {
        "total_memories": len(nodes),
        "total_edges": len(edges),
        "total_clusters": len(clusters),
        "storage_mb": len(nodes) * 0.001,  # Rough estimate
    }
    
    return MemoryUniverseResponse(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        stats=stats
    )
