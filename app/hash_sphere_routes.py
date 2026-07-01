import os
import re as _re
import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .embeddings import embeddings_generator
from .models import MemoryRecord, MemoryEmbedding
from .services import resonance_hasher, memory_anchor_service
from .services.resonance_hashing import ResonanceHasher
from .services.memory_encryption import decrypt_memory_content
from .services.pgvector_search import pgvector_search
from .schemas import (
    HashSphereExtractRequest,
    HashSphereExtractResponse,
    HashSphereMemory,
    HashRequest,
    HashResponse,
    ResonanceRequest,
    ResonanceResponse,
    AnchorCreateRequest,
    AnchorResponse,
    AnchorSearchRequest,
    ArchiveRequest,
    ArchiveResponse,
    HashSphereTokenRequest,
    HashSphereTokenResponse,
)

logger = logging.getLogger(__name__)

hash_sphere_router = APIRouter(prefix="/memory", tags=["hash-sphere"])
public_router = APIRouter(prefix="/public", tags=["public"])


@hash_sphere_router.post("/hash-sphere/extract", response_model=HashSphereExtractResponse)
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
    start_time = time.perf_counter()
    methods_used = []
    
    # Generate query embedding FIRST so coordinates are semantic
    query_embedding = None
    try:
        query_embeddings = await embeddings_generator.generate([request.query], task="search_query")
        if query_embeddings:
            query_embedding = query_embeddings[0]
    except Exception:
        pass
    
    # Generate query hash and coordinates (embedding-driven XYZ)
    query_hash = ResonanceHasher.hash_text(request.query)
    query_coords = ResonanceHasher.compute_full_coordinates(
        request.query,
        embedding=query_embedding,
    )
    query_xyz = (query_coords.x, query_coords.y, query_coords.z)
    query_resonance = query_coords.resonance_score
    
    # Collect all memories from different methods
    all_memories: Dict[str, Dict] = {}
    
    # Convert user_id/org_id to UUID if provided
    user_uuid = uuid.UUID(request.user_id) if request.user_id else None
    org_uuid = uuid.UUID(request.org_id) if request.org_id else None

    # PATCH B: session scoping — narrow candidates to this conversation when provided
    session_uuid = None
    if getattr(request, "session_id", None):
        try:
            session_uuid = uuid.UUID(request.session_id)
        except (ValueError, TypeError):
            session_uuid = None

    # Relevance floor (env override) — discards low-relevance noise.
    # NOTE: this gates on SEMANTIC similarity (rag/resonance cosine, 0-1 range),
    # NOT the RRF hybrid_score. RRF fusion scores max out around ~0.05, so the old
    # 0.35 hybrid floor silently rejected 100% of results. RRF is used only for
    # ordering; relevance is decided by actual cosine similarity.
    min_score = request.min_score
    if min_score is None:
        try:
            min_score = float(os.getenv("MIN_SEMANTIC_SCORE", "0.25"))
        except (ValueError, TypeError):
            min_score = 0.25
    
    # ============================================
    # METHOD 1: Anchor-based lookup (PRIORITY 1)
    # ============================================
    if request.use_anchors:
        try:
            keywords = resonance_hasher.extract_anchors(request.query)
            if keywords:
                stmt = select(MemoryRecord).where(
                    MemoryRecord.source == "anchor"
                )
                if user_uuid:
                    stmt = stmt.where(MemoryRecord.user_id == user_uuid)
                if session_uuid:
                    stmt = stmt.where(MemoryRecord.chat_id == session_uuid)
                
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
        except Exception:
            pass
    
    # ============================================
    # METHOD 2: Proximity search (PRIORITY 2)
    # ============================================
    if request.use_proximity:
        try:
            stmt = select(MemoryRecord).where(
                MemoryRecord.xyz_x.isnot(None)
            ).limit(200)
            if user_uuid:
                stmt = stmt.where(MemoryRecord.user_id == user_uuid)
            if session_uuid:
                stmt = stmt.where(MemoryRecord.chat_id == session_uuid)
            
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            for record in records:
                if record.xyz_x is not None:
                    mem_xyz = (record.xyz_x, record.xyz_y, record.xyz_z)
                    proximity = ResonanceHasher.calculate_proximity_score(query_xyz, mem_xyz)
                    
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
        except Exception:
            pass
    
    # ============================================
    # METHOD 3: Resonance filtering (PRIORITY 3)
    # Uses EMBEDDING cosine similarity (semantic) instead of hash Hamming (noise)
    # ============================================
    if request.use_resonance:
        try:
            if query_embedding:
                emb_stmt = select(MemoryEmbedding).limit(200)
                if user_uuid:
                    emb_stmt = emb_stmt.where(MemoryEmbedding.user_id == user_uuid)
                if session_uuid and hasattr(MemoryEmbedding, "chat_id"):
                    emb_stmt = emb_stmt.where(MemoryEmbedding.chat_id == session_uuid)
                
                emb_result = await session.execute(emb_stmt)
                emb_records = emb_result.scalars().all()
                
                mem_embedding_map = {}
                for emb_rec in emb_records:
                    mem_embedding_map[str(emb_rec.memory_id)] = emb_rec.embedding
                
                if mem_embedding_map:
                    mem_ids_to_load = [
                        uuid.UUID(mid) for mid in mem_embedding_map.keys()
                        if mid not in all_memories
                    ]
                    if mem_ids_to_load:
                        rec_stmt = select(MemoryRecord).where(
                            MemoryRecord.id.in_(mem_ids_to_load)
                        )
                        if session_uuid:
                            rec_stmt = rec_stmt.where(MemoryRecord.chat_id == session_uuid)
                        rec_result = await session.execute(rec_stmt)
                        rec_records = rec_result.scalars().all()
                        
                        for record in rec_records:
                            content = decrypt_memory_content(record.content)
                            if not content or len(content) < 10 or content.startswith("ENC2:"):
                                continue
                            
                            mem_id = str(record.id)
                            mem_emb = mem_embedding_map.get(mem_id)
                            resonance = ResonanceHasher.calculate_resonance_from_embeddings(
                                query_embedding, mem_emb
                            ) if mem_emb else 0.0
                            
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
                    
                    for mem_id, mem_data in all_memories.items():
                        if mem_id in mem_embedding_map:
                            mem_emb = mem_embedding_map[mem_id]
                            mem_data["resonance_score"] = ResonanceHasher.calculate_resonance_from_embeddings(
                                query_embedding, mem_emb
                            )
                    
                    if emb_records:
                        methods_used.append("resonance_embedding")
            else:
                stmt = select(MemoryRecord).where(
                    MemoryRecord.hash.isnot(None)
                ).limit(200)
                if user_uuid:
                    stmt = stmt.where(MemoryRecord.user_id == user_uuid)
                if session_uuid:
                    stmt = stmt.where(MemoryRecord.chat_id == session_uuid)
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                for record in records:
                    if record.hash:
                        resonance = resonance_hasher.calculate_resonance(query_hash, record.hash)
                        
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
                    methods_used.append("resonance_hash_legacy")
        except Exception:
            pass
    
    # ============================================
    # METHOD 4: RAG Semantic Search (ALWAYS RUN)
    # RAG has the highest hybrid weight (0.30) — it MUST always run
    # to provide semantic similarity scores for proper ranking.
    # ============================================
    if request.use_rag_fallback:
        try:
            if query_embedding:
                pgvector_results = await pgvector_search.search_similar(
                    session=session,
                    query_embedding=query_embedding,
                    user_id=request.user_id,
                    limit=request.limit * 2,
                )
                
                for r in pgvector_results:
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
                        all_memories[mem_id]["rag_score"] = r.similarity
                
                if pgvector_results:
                    methods_used.append("rag_semantic")
        except Exception:
            pass
    
    # ============================================
    # METHOD 5: BM25 full-text search (keyword matching)
    # ============================================
    try:
        bm25_results = await pgvector_search.search_bm25(
            session=session,
            query=request.query,
            user_id=request.user_id,
            limit=request.limit * 2,
        )
        for r in bm25_results:
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
                    "bm25_score": r.similarity,
                    "resonance_score": r.resonance_score or 0.0,
                    "timestamp": None,
                }
            else:
                all_memories[mem_id]["bm25_score"] = r.similarity
        if bm25_results:
            methods_used.append("bm25_fulltext")
    except Exception:
        pass

    # ============================================
    # METHOD 6: Temporal search (time-based queries)
    # ============================================
    try:
        from .services.temporal_memory import extract_temporal_memories
        temporal_results = await extract_temporal_memories(
            session=session,
            user_id=request.user_id,
            query=request.query,
            limit=request.limit,
        )
        for mem in temporal_results:
            mem_id = mem.get("id")
            if mem_id and mem_id not in all_memories:
                all_memories[mem_id] = {
                    "id": mem_id,
                    "content": mem["content"],
                    "type": "temporal",
                    "hash": None,
                    "xyz": None,
                    "rag_score": 0.5,
                    "resonance_score": 0.0,
                    "timestamp": mem.get("timestamp"),
                }
        if temporal_results:
            methods_used.append("temporal")
    except Exception:
        pass

    # ============================================
    # HYBRID RANKING with all scores
    # ============================================
    from .services.hybrid_memory_ranker import rank_memories
    
    # Session scope filter: drop any memory not in the current chat session
    if session_uuid and all_memories:
        try:
            mem_uuids = []
            for mid in all_memories.keys():
                if not mid:
                    continue
                try:
                    mem_uuids.append(uuid.UUID(mid))
                except (ValueError, TypeError):
                    continue
            if mem_uuids:
                scope_stmt = select(MemoryRecord.id).where(
                    MemoryRecord.id.in_(mem_uuids),
                    MemoryRecord.chat_id == session_uuid,
                )
                scope_result = await session.execute(scope_stmt)
                allowed_ids = {str(row[0]) for row in scope_result.all()}
                all_memories = {
                    mid: m for mid, m in all_memories.items()
                    if mid in allowed_ids
                }
        except Exception:
            pass

    memories_list = list(all_memories.values())
    
    # Normalize BM25 scores to 0-1 range
    raw_bm25 = [m.get("bm25_score", 0.0) for m in memories_list]
    max_bm25 = max(raw_bm25) if raw_bm25 else 1.0
    max_bm25 = max(max_bm25, 0.001)
    
    for mem in memories_list:
        if "rag_score" not in mem:
            mem["rag_score"] = 0.0
        if "resonance_score" not in mem:
            mem["resonance_score"] = 0.0
        
        mem["bm25_score"] = mem.get("bm25_score", 0.0) / max_bm25
        
        if "recency_score" not in mem:
            if mem.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(mem["timestamp"].replace('Z', '+00:00'))
                    age_days = (datetime.now(ts.tzinfo) - ts).days
                    mem["recency_score"] = np.exp(-age_days / 30.0)
                except Exception:
                    mem["recency_score"] = 0.5
            else:
                mem["recency_score"] = 0.5
    
    ranked_memories = rank_memories(memories_list)

    # Drop low-relevance noise: keep a memory only if its RAG semantic similarity
    # (pgvector cosine, a clean 0-1 signal) clears the floor. This filters out
    # proximity-only hits (XYZ is hash-derived noise) and keyword-only BM25 hits
    # with no semantic overlap. RRF above still decides ordering among survivors.
    # NOTE: resonance_score is intentionally NOT used here — RAG/BM25 paths fill it
    # from the stored R(h)=sin+cos+tan resonance-function column (unbounded, ~1.7),
    # a different metric from the embedding cosine, so it can't share this floor.
    # Only apply the RAG floor when RAG semantic search actually ran; otherwise
    # (embeddings unavailable / use_rag_fallback disabled) rag_score is 0 for all
    # and filtering would wipe every result — keep the ranked list as-is instead.
    if min_score and min_score > 0 and "rag_semantic" in methods_used:
        ranked_memories = [
            m for m in ranked_memories
            if float(m.get("rag_score", 0.0) or 0.0) >= min_score
        ]

    top_memories = ranked_memories[:request.limit]
    
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


@hash_sphere_router.post("/hash-sphere/hash", response_model=HashResponse)
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


@hash_sphere_router.post("/hash-sphere/resonance", response_model=ResonanceResponse)
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


@hash_sphere_router.post("/hash-sphere/anchors", response_model=AnchorResponse)
async def create_anchor(
    request: AnchorCreateRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a memory anchor."""
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


@hash_sphere_router.get("/hash-sphere/anchors", response_model=List[AnchorResponse])
async def list_anchors(
    user_id: Optional[str] = None,
    limit: int = 2000,
    req: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """List memory anchors."""
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


@hash_sphere_router.post("/hash-sphere/search", response_model=List[AnchorResponse])
async def search_anchors(
    request: AnchorSearchRequest,
    req: Request = None,
    session: AsyncSession = Depends(get_session),
):
    """Search anchors by resonance."""
    search_user_id = request.user_id
    if not search_user_id and req:
        search_user_id = req.headers.get("x-user-id")

    search_user_uuid: Optional[uuid.UUID] = None
    try:
        if search_user_id:
            search_user_uuid = uuid.UUID(search_user_id)
    except Exception:
        search_user_uuid = None

    stmt = select(MemoryRecord).where(MemoryRecord.source == "anchor")
    if search_user_uuid:
        stmt = stmt.where(MemoryRecord.user_id == search_user_uuid)
    result = await session.execute(stmt)
    records = result.scalars().all()

    record_map: Dict[str, MemoryRecord] = {str(r.id): r for r in records}
    
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

@hash_sphere_router.post("/archive/file", response_model=ArchiveResponse)
async def archive_file(
    request: ArchiveRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Archive a file - sets is_archived=True on all anchors with this file_path.
    
    Hash Sphere is immutable - data stays but won't be loaded when archived.
    """
    user_id = req.headers.get("x-user-id")
    file_path = request.file_path
    
    if not file_path:
        return ArchiveResponse(
            success=False,
            archived_count=0,
            file_path=file_path,
            message="file_path is required"
        )
    
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


@hash_sphere_router.post("/unarchive/file", response_model=ArchiveResponse)
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


@hash_sphere_router.get("/archived/files")
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
    
    file_paths = set()
    for record in records:
        if record.extra_metadata and record.extra_metadata.get("file_path"):
            file_paths.add(record.extra_metadata["file_path"])
    
    return {
        "archived_files": list(file_paths),
        "count": len(file_paths)
    }


# ============================================
# PUBLIC HASH SPHERE ENDPOINTS
# No authentication required - for public pages
# ============================================

@public_router.post("/hash-sphere/token", response_model=HashSphereTokenResponse)
async def get_hash_sphere_token(
    payload: HashSphereTokenRequest,
):
    """Get a Hash Sphere access token for public pages.
    
    - Owner token: unlimited memory, 30-day expiration
    - Guest token: limited memory, 1-hour expiration
    """
    import secrets
    from datetime import timedelta
    
    if payload.is_owner:
        expires_in_hours = 24 * 30
    else:
        expires_in_hours = 1
    
    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    token = f"hs_{secrets.token_urlsafe(32)}"
    
    return HashSphereTokenResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        expires_in_hours=expires_in_hours,
    )
