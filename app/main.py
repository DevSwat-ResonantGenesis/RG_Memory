import os
import sys
import logging
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add shared modules to path
SHARED_PATH = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_PATH))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Deterministic sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Credit costs from pricing.yaml
CREDIT_COSTS = {
    "embed": 100,
    "retrieve": 50,
    "store": 20,
    "delete": 5,
    "per_mb": 1,
    "per_gb": 1000,
    "memory_write": 2,
    "memory_read": 0,
    "rag_upload": 10,
}

BILLING_SERVICE_URL = os.getenv("BILLING_SERVICE_URL", "http://billing_service:8000")

async def deduct_credits(user_id: str, amount: int, reference_type: str, description: str) -> dict:
    """Deduct credits from user's balance via billing service."""
    if amount <= 0:
        return {"status": "skipped", "reason": "no credits to deduct"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BILLING_SERVICE_URL}/billing/credits/deduct",
                json={
                    "amount": amount,
                    "reference_type": reference_type,
                    "description": description,
                },
                headers={"X-User-Id": user_id},
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Credit deduction failed: {e}")
        return {"error": str(e)}

# Single service entrypoint
app = FastAPI(
    title="Memory_Service Service",
    description="Service for Genesis2026",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .embeddings import embeddings_generator
from .visualizer_routes import router as visualizer_router

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory_service"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": f"Memory_Service Service is running"}

# Service-specific endpoint
@app.get("/api/v1/status")
async def status():
    return {"service": "memory_service", "status": "active", "version": "1.0.0"}


# ============================================
# RAG ENDPOINTS - Stub implementations for frontend compatibility
# ============================================

@app.get("/memory/rag/conversations")
async def rag_conversations(limit: int = 1000, include_details: bool = True):
    """Get RAG conversations - stub for frontend compatibility."""
    return []


@app.get("/memory/rag/memories")
async def rag_memories(limit: int = 100, request: Request = None):
    """Get RAG memories - proxy to the full RAG router."""
    from .routers import rag_router
    # Forward to the actual RAG memories endpoint
    user_id = request.headers.get("x-user-id") if request else None
    if not user_id:
        return []
    # Import and call the actual endpoint
    from .routers import list_rag_memories
    from .db import get_session
    async for session in get_session():
        try:
            return await list_rag_memories(limit=limit, request=request, session=session)
        except Exception as e:
            logger.error(f"Failed to list RAG memories: {e}")
            return []


@app.post("/memory/rag/memories")
async def create_rag_memory_proxy(request: Request):
    """Create RAG memory - proxy to the full RAG router."""
    from .routers import create_rag_memory, RAGMemoryCreateRequest
    from .db import get_session
    
    user_id = request.headers.get("x-user-id")
    if not user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Parse request body
    body = await request.json()
    payload = RAGMemoryCreateRequest(
        content=body.get("content", ""),
        metadata=body.get("metadata"),
        is_shared=body.get("is_shared", False),
        is_public=body.get("is_public", False),
        shared_with=body.get("shared_with"),
        language=body.get("language"),
    )
    
    async for session in get_session():
        try:
            return await create_rag_memory(payload=payload, request=request, session=session)
        except Exception as e:
            logger.error(f"Failed to create RAG memory: {e}")
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/rag/search")
async def rag_search(query: str = "", limit: int = 10, request: Request = None):
    """Search RAG memories - stub for frontend compatibility."""
    # NOTE: No credit deduction on stub endpoint - returns empty results
    # Credits will be deducted when actual RAG functionality is implemented
    return {"results": [], "query": query}


class MemoryIngestRequest(BaseModel):
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    chat_id: Optional[str] = None
    source: str = "resonant-chat"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    generate_embedding: bool = True
    agent_hash: Optional[str] = None


class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    content: Optional[str] = None
    task: str = "search_query"

@app.post("/memory/ingest")
async def ingest_memory(request: MemoryIngestRequest, req: Request):
    """Ingest a memory into the system with credit deduction."""
    from .db import get_session
    from .routers import ingest_memory as ingest_memory_endpoint
    from .routers import MemoryIngestRequest as RouterMemoryIngestRequest

    payload = RouterMemoryIngestRequest(
        chat_id=request.chat_id,
        user_id=request.user_id,
        org_id=request.org_id,
        source=request.source,
        content=request.content,
        metadata=request.metadata,
        generate_embedding=request.generate_embedding,
        agent_hash=request.agent_hash,
    )

    # Deduct credits AFTER successful storage to avoid charging for failed operations
    result = None
    async for session in get_session():
        try:
            result = await ingest_memory_endpoint(payload=payload, session=session)
            # Only deduct credits if storage succeeded
            user_id = req.headers.get("x-user-id") or request.user_id
            if user_id and result and getattr(result, "id", None):
                await deduct_credits(user_id, CREDIT_COSTS["store"], "memory_store", f"Memory ingest from {request.source}")
                logger.info(f"💳 Deducted {CREDIT_COSTS['store']} credits for memory ingest")
            return result
        except Exception as e:
            logger.error(f"Memory ingest failed: {e}")
            return {"status": "failed", "error": str(e)}

    return {"status": "failed", "error": "no_db_session"}

@app.post("/memory/embed")
async def embed_content(payload: EmbedRequest, request: Request = None):
    """Embed content into vector space with credit deduction."""
    user_id = request.headers.get("x-user-id") if request else None
    if user_id:
        await deduct_credits(user_id, CREDIT_COSTS["embed"], "memory_embed", "Content embedding")
        logger.info(f"💳 Deducted {CREDIT_COSTS['embed']} credits for embedding")

    texts: List[str] = []
    if payload.texts:
        texts = payload.texts
    elif payload.content:
        texts = [payload.content]

    if not texts:
        return {"status": "embedded", "dimensions": 0, "embeddings": []}

    try:
        embeddings = await embeddings_generator.generate(texts, task=payload.task)
        dims = len(embeddings[0]) if embeddings else 0
        return {"status": "embedded", "dimensions": dims, "embeddings": embeddings}
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return {"status": "embedded", "dimensions": 0, "embeddings": []}

@app.get("/memory/rag/stats")
async def rag_stats():
    """Get RAG statistics from real database."""
    from .db import get_session
    from sqlalchemy import text
    try:
        async for session in get_session():
            r1 = await session.execute(text("SELECT COUNT(*) FROM memory_records"))
            total_memories = r1.scalar() or 0
            r2 = await session.execute(text("SELECT COUNT(DISTINCT chat_id) FROM memory_records WHERE chat_id IS NOT NULL"))
            total_conversations = r2.scalar() or 0
            r3 = await session.execute(text("SELECT COUNT(DISTINCT cluster_name) FROM memory_records WHERE cluster_name IS NOT NULL"))
            total_clusters = r3.scalar() or 0
            r4 = await session.execute(text("SELECT pg_total_relation_size('memory_records')"))
            total_size_bytes = r4.scalar() or 0
            return {
                "total_memories": total_memories,
                "total_conversations": total_conversations,
                "total_clusters": total_clusters,
                "total_size_bytes": total_size_bytes,
                "storage_bytes": total_size_bytes,
                "storage_mb": round(total_size_bytes / (1024 * 1024), 2),
            }
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        return {
            "total_memories": 0,
            "total_conversations": 0,
            "total_clusters": 0,
            "total_size_bytes": 0,
            "storage_bytes": 0,
            "storage_mb": 0,
        }


# Include the full routers with Hash Sphere extraction endpoint
from .routers import router as memory_router, rag_router
app.include_router(visualizer_router)
app.include_router(memory_router)
app.include_router(rag_router)


# ============================================
# HASH SPHERE ENDPOINTS - Stub implementations
# ============================================

@app.get("/memory/hash-sphere/anchors")
async def hash_sphere_anchors(user_id: str = None, limit: int = 10000):
    """Get Hash Sphere anchors - returns real memory records with coordinates."""
    from .db import get_session
    from sqlalchemy import text
    try:
        logger.info(f"Hash sphere anchors request: user_id={user_id}, limit={limit}")
        session_gen = get_session()
        session = await session_gen.__anext__()
        try:
            query = "SELECT id, content, hash, source, xyz_x, xyz_y, xyz_z, "
            query += "sphere_r, sphere_phi, sphere_theta, resonance_score, normalized_resonance, "
            query += "anchor_energy, spin_x, spin_y, spin_z, spin_magnitude, "
            query += "meaning_score, intensity_score, sentiment_score, "
            query += "meaning_hash, energy_hash, spin_hash, universe_id, "
            query += "cluster_name, cluster_distance, created_at "
            query += "FROM memory_records"
            params = {}
            if user_id:
                query += " WHERE user_id = :user_id"
                params["user_id"] = user_id
            query += " ORDER BY created_at DESC LIMIT :limit"
            params["limit"] = limit
            result = await session.execute(text(query), params)
            rows = result.fetchall()
            columns = list(result.keys())
            logger.info(f"Hash sphere anchors: found {len(rows)} records")
            anchors = []
            for row in rows:
                record = dict(zip(columns, row))
                anchors.append({
                    "id": str(record.get("id", "")),
                    "anchor_text": (record.get("content") or "")[:100],
                    "anchor_hash": record.get("hash", ""),
                    "context": record.get("content", ""),
                    "anchor_type": record.get("source", "chat"),
                    "xyz_x": record.get("xyz_x"),
                    "xyz_y": record.get("xyz_y"),
                    "xyz_z": record.get("xyz_z"),
                    "sphere_r": record.get("sphere_r"),
                    "sphere_phi": record.get("sphere_phi"),
                    "sphere_theta": record.get("sphere_theta"),
                    "resonance_score": record.get("resonance_score"),
                    "normalized_resonance": record.get("normalized_resonance"),
                    "anchor_energy": record.get("anchor_energy"),
                    "spin_x": record.get("spin_x"),
                    "spin_y": record.get("spin_y"),
                    "spin_z": record.get("spin_z"),
                    "spin_magnitude": record.get("spin_magnitude"),
                    "meaning_score": record.get("meaning_score"),
                    "intensity_score": record.get("intensity_score"),
                    "sentiment_score": record.get("sentiment_score"),
                    "importance_score": record.get("meaning_score"),
                    "meaning_hash": record.get("meaning_hash"),
                    "energy_hash": record.get("energy_hash"),
                    "spin_hash": record.get("spin_hash"),
                    "universe_id": record.get("universe_id"),
                    "cluster_name": record.get("cluster_name"),
                    "created_at": str(record.get("created_at", "")),
                })
            return anchors
        finally:
            await session.close()
    except Exception as e:
        logger.error(f"Failed to get hash sphere anchors: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


@app.get("/memory/hash-sphere/health_stub")
async def hash_sphere_health():
    """Hash Sphere health check."""
    return {"status": "ok", "service": "hash-sphere"}


@app.post("/memory/clusters/compute")
async def compute_clusters(batch_size: int = 500):
    """Retroactively compute cluster assignments for memories missing cluster_name.
    Uses SemanticEncoder to assign one of 6 clusters (Alpha-Zeta) based on content."""
    from .db import get_session
    from .services.semantic_encoder import get_semantic_encoder, SemanticCluster
    from sqlalchemy import text

    encoder = get_semantic_encoder()
    cluster_names = {
        SemanticCluster.ALPHA: "Alpha-Living",
        SemanticCluster.BETA: "Beta-Inanimate",
        SemanticCluster.GAMMA: "Gamma-Abstract",
        SemanticCluster.DELTA: "Delta-Actions",
        SemanticCluster.EPSILON: "Epsilon-Qualities",
        SemanticCluster.ZETA: "Zeta-Relations",
    }
    updated = 0
    errors = 0
    try:
        async for session in get_session():
            rows = await session.execute(
                text("SELECT id, content FROM memory_records WHERE cluster_name IS NULL LIMIT :lim"),
                {"lim": batch_size},
            )
            records = rows.fetchall()
            for row in records:
                rid, content = row
                if not content:
                    continue
                try:
                    result = encoder.encode(content[:2000])
                    cname = cluster_names.get(result.dominant_cluster, "Unknown")
                    await session.execute(
                        text("UPDATE memory_records SET cluster_name = :cn WHERE id = :rid"),
                        {"cn": cname, "rid": rid},
                    )
                    updated += 1
                except Exception:
                    errors += 1
            await session.commit()
            remaining = await session.execute(
                text("SELECT COUNT(*) FROM memory_records WHERE cluster_name IS NULL")
            )
            rem = remaining.scalar() or 0
            return {
                "updated": updated,
                "errors": errors,
                "remaining": rem,
                "batch_size": batch_size,
            }
    except Exception as e:
        logger.error(f"Cluster computation failed: {e}")
        return {"updated": updated, "errors": errors, "error": str(e)[:200]}


@app.get("/memory/clusters/stats")
async def cluster_stats():
    """Get cluster distribution stats."""
    from .db import get_session
    from sqlalchemy import text
    try:
        async for session in get_session():
            rows = await session.execute(text(
                "SELECT cluster_name, COUNT(*) as cnt FROM memory_records "
                "WHERE cluster_name IS NOT NULL GROUP BY cluster_name ORDER BY cnt DESC"
            ))
            clusters = [{"name": r[0], "count": r[1]} for r in rows.fetchall()]
            total_with = sum(c["count"] for c in clusters)
            total_without_r = await session.execute(
                text("SELECT COUNT(*) FROM memory_records WHERE cluster_name IS NULL")
            )
            total_without = total_without_r.scalar() or 0
            return {
                "clusters": clusters,
                "total_clustered": total_with,
                "total_unclustered": total_without,
                "total_clusters": len(clusters),
            }
    except Exception as e:
        logger.error(f"Cluster stats failed: {e}")
        return {"clusters": [], "total_clustered": 0, "total_unclustered": 0, "error": str(e)[:200]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
