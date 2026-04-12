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
# RAG ENDPOINTS - Stubs removed; rag_router now handles /memory/rag/* paths
# Only /memory/rag/stats is kept here as it queries raw SQL.
# ============================================


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

logger.info("Routers mounted: memory_router (prefix=/memory), rag_router (prefix=/memory/rag), visualizer_router")


# ============================================
# HASH SPHERE ENDPOINTS — duplicate /memory/hash-sphere/anchors REMOVED (security fix)
# The router.py version at /memory/hash-sphere/anchors now handles this
# with proper user_id filtering from x-user-id header.
# ============================================


@app.get("/memory/hash-sphere/health_stub")
async def hash_sphere_health():
    """Hash Sphere health check."""
    return {"status": "ok", "service": "hash-sphere"}


@app.post("/memory/hash-sphere/retrain")
async def trigger_retrain():
    """Manually trigger retraining of Hash Sphere ML models.
    
    Retrains:
    - Semantic encoder (cluster/temperature/polarity from embeddings)
    - Sphere projection (512→3D triplet-loss neural network)
    
    Uses production MemoryEmbedding data as training source.
    """
    try:
        from .services.retraining_loop import get_retrainer
        retrainer = get_retrainer()
        result = await retrainer.trigger_manual_retrain()
        return result
    except Exception as e:
        logger.error(f"Manual retrain failed: {e}")
        return {"status": "failed", "error": str(e)}


@app.on_event("startup")
async def start_retraining_loop():
    """Start the autonomous retraining loop on app startup."""
    try:
        from .services.retraining_loop import get_retrainer
        retrainer = get_retrainer()
        await retrainer.start()
        logger.info("Hash Sphere autonomous retraining loop started")
    except Exception as e:
        logger.warning(f"Failed to start retraining loop: {e}")


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
