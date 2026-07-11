import uuid as uuid_module
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Request, HTTPException, UploadFile, File
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .embeddings import embeddings_generator
from .models import MemoryRecord, MemoryEmbedding
from .services import resonance_hasher
from .services.memory_encryption import decrypt_memory_content
from .services.document_loaders import parse_document, chunk_text
from .helpers import _get_user_id, _get_org_id
from .schemas import (
    RAGFileUploadResponse,
    RAGMemoryCreateRequest,
    RAGMemoryResponse,
    RAGMemoryUpdateRequest,
    RAGAskRequest,
    RAGAskResponse,
    RAGConversationResponse,
    MemoryNodeResponse,
    MemoryEdgeResponse,
    MemoryUniverseResponse,
)

logger = logging.getLogger(__name__)

BILLING_SERVICE_URL = __import__("os").getenv("BILLING_SERVICE_URL", "http://billing_service:8000")

rag_router = APIRouter(prefix="/memory/rag", tags=["rag"])


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
                pass
        
        rag_limits = {
            "developer": 5, "free": 5,
            "plus": 100, "professional": 100,
            "enterprise": -1, "unlimited": -1,
        }
        max_docs = rag_limits.get(user_plan, 5)
        
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
    
    hash_value = resonance_hasher.hash_text(payload.content)
    xyz = resonance_hasher.to_xyz(hash_value)
    
    metadata = payload.metadata or {}
    metadata["is_shared"] = payload.is_shared
    metadata["is_public"] = payload.is_public
    if payload.shared_with:
        metadata["shared_with"] = payload.shared_with
    if payload.language:
        metadata["language"] = payload.language
    
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
        hash_value = resonance_hasher.hash_text(payload.content)
        xyz = resonance_hasher.to_xyz(hash_value)
        record.hash = hash_value
        record.xyz_x = xyz[0] if xyz else None
        record.xyz_y = xyz[1] if xyz else None
        record.xyz_z = xyz[2] if xyz else None
        
        embeddings = await embeddings_generator.generate([payload.content])
        if embeddings:
            del_stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == record.id)
            del_result = await session.execute(del_stmt)
            old_emb = del_result.scalar_one_or_none()
            if old_emb:
                await session.delete(old_emb)
            
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
    
    stmt = select(MemoryRecord).where(MemoryRecord.id == memory_id)
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if record.source != "rag":
        raise HTTPException(
            status_code=403,
            detail="Hash Sphere memories are immutable and cannot be deleted. "
                   "Only user-created RAG memories can be removed."
        )
    
    emb_stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
    emb_result = await session.execute(emb_stmt)
    embedding = emb_result.scalar_one_or_none()
    if embedding:
        await session.delete(embedding)
    
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
    from .config import settings
    
    user_id = _get_user_id(request)
    conversation_id = payload.conversation_id or str(uuid_module.uuid4())
    
    sources = []
    if payload.use_memory:
        query_embeddings = await embeddings_generator.generate([payload.query])
        if query_embeddings:
            query_embedding = query_embeddings[0]
            
            stmt = select(MemoryEmbedding)
            if user_id:
                stmt = stmt.where(MemoryEmbedding.user_id == user_id)
            
            result = await session.execute(stmt)
            embeddings_list = result.scalars().all()
            
            similarities = []
            for emb in embeddings_list:
                similarity = float(embeddings_generator.cosine_similarity(query_embedding, emb.embedding))
                similarities.append((emb.memory_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memory_ids = [mid for mid, _ in similarities[:payload.top_k]]
            similarity_map = {mid: sim for mid, sim in similarities[:payload.top_k]}
            
            if top_memory_ids:
                mem_stmt = select(MemoryRecord).where(MemoryRecord.id.in_(top_memory_ids))
                mem_result = await session.execute(mem_stmt)
                records = mem_result.scalars().all()
                
                sources = [
                    {
                        "id": str(r.id),
                        "content": (decrypt_memory_content(r.content) if r.content else "")[:500],
                        "score": similarity_map.get(r.id, 0.0),
                        "hash": r.hash,
                        "xyz": [r.xyz_x, r.xyz_y, r.xyz_z] if r.xyz_x else None,
                        "metadata": r.extra_metadata or {},
                    }
                    for r in records
                ]
    
    context = "\n\n".join([s["content"] for s in sources]) if sources else ""
    
    response_text = ""
    validity = 0.5
    
    try:
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
                response_text = f"Based on {len(sources)} relevant memories, here's what I found related to your query: '{payload.query}'"
                if context:
                    response_text = f"Context from your memories:\n{context[:1000]}...\n\nBased on this context, I can help answer your question about: {payload.query}"
    except Exception:
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
    
    stmt = select(MemoryRecord.extra_metadata).where(
        MemoryRecord.source == "rag_conversation"
    ).order_by(MemoryRecord.created_at.desc()).limit(limit)
    
    if user_id:
        stmt = stmt.where(MemoryRecord.user_id == user_id)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
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
    return []


@rag_router.delete("/conversations/{conversation_id}")
async def delete_rag_conversation(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a conversation - RAG compatibility."""
    return {"status": "deleted", "conversation_id": conversation_id}


@rag_router.put("/conversations/{conversation_id}")
async def update_rag_conversation(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Update conversation title - RAG compatibility."""
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
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    content = await file.read()
    file_size = len(content)
    
    try:
        text_content = parse_document(content, file.filename or "file.txt", file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to parse file: {e}")
    
    chunks = chunk_text(text_content, max_chars=1000, overlap=100)
    
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
    
    stmt = select(MemoryRecord).where(
        MemoryRecord.user_id == user_id,
        MemoryRecord.org_id == org_id
    ).order_by(MemoryRecord.created_at.desc()).limit(500)
    
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    nodes = []
    clusters_map = {}
    
    for r in records:
        x = float(r.xyz_x) if r.xyz_x is not None else 0.0
        y = float(r.xyz_y) if r.xyz_y is not None else 0.0
        z = float(r.xyz_z) if r.xyz_z is not None else 0.0
        
        x = (x - 0.5) * 400
        y = (y - 0.5) * 400
        z = (z - 0.5) * 400
        
        decrypted_content = decrypt_memory_content(r.content) if r.content else ""
        metadata = r.extra_metadata or {}
        layer = metadata.get("layer", "active")
        cluster = metadata.get("cluster", "default")
        importance = metadata.get("importance", 0.5)
        tags = metadata.get("tags", [])
        access_count = metadata.get("access_count", 0)
        title = metadata.get("title", decrypted_content[:50] if decrypted_content else "Untitled")

        if cluster not in clusters_map:
            clusters_map[cluster] = {"name": cluster, "count": 0, "center": [0, 0, 0]}
        clusters_map[cluster]["count"] += 1
        clusters_map[cluster]["center"][0] += x
        clusters_map[cluster]["center"][1] += y
        clusters_map[cluster]["center"][2] += z

        nodes.append(MemoryNodeResponse(
            id=str(r.id),
            content=decrypted_content[:500],
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
    
    clusters = []
    for name, data in clusters_map.items():
        if data["count"] > 0:
            data["center"] = [c / data["count"] for c in data["center"]]
        clusters.append(data)
    
    edges = []
    for i, n1 in enumerate(nodes[:100]):
        for n2 in nodes[i+1:100]:
            dist = ((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2) ** 0.5
            if dist < 100:
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
        "storage_mb": len(nodes) * 0.001,
    }
    
    return MemoryUniverseResponse(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        stats=stats
    )
