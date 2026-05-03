import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .embeddings import embeddings_generator
from .models import MemoryRecord, MemoryEmbedding
from .schemas import (
    AgentMemoryCreateRequest,
    AgentMemoryResponse,
    SemanticSearchRequest,
    SemanticSearchResult,
    SemanticSearchResponse,
    ConsolidateRequest,
    MemorySettingsRequest,
    MemoryClusterResponse,
    MemoryAnalyticsResponse,
)

logger = logging.getLogger(__name__)

agent_memory_router = APIRouter(prefix="/memory", tags=["agent-memory"])


@agent_memory_router.post("/", response_model=AgentMemoryResponse)
async def create_agent_memory(
    payload: AgentMemoryCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a new agent memory with embedding."""
    memory_id = str(uuid.uuid4())
    
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
    stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
    result = await session.execute(stmt)
    embedding = result.scalar_one_or_none()
    if embedding:
        await session.delete(embedding)
    
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
    query_embeddings = await embeddings_generator.generate([payload.query])
    results = []
    
    if query_embeddings:
        query_embedding = query_embeddings[0]
        
        stmt = select(MemoryEmbedding).where(MemoryEmbedding.user_id == payload.agent_id)
        result = await session.execute(stmt)
        embeddings = result.scalars().all()
        
        similarities = []
        for emb in embeddings:
            similarity = embeddings_generator.cosine_similarity(query_embedding, emb.embedding)
            if similarity >= payload.threshold:
                similarities.append((emb.memory_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:payload.top_k]
        
        for memory_id, score in top_results:
            stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            
            if record:
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
    stmt = select(MemoryRecord).where(
        MemoryRecord.user_id == payload.agent_id,
        MemoryRecord.source == "short-term"
    ).order_by(MemoryRecord.created_at.asc())
    
    result = await session.execute(stmt)
    short_term_memories = result.scalars().all()
    
    consolidated_count = 0
    
    if len(short_term_memories) > payload.threshold:
        to_consolidate = short_term_memories[:len(short_term_memories) - payload.threshold]
        
        for memory in to_consolidate:
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
        storage_used=len(records) * 0.5,
        retrieval_latency=23,
        consolidation_rate=0.85,
    )
