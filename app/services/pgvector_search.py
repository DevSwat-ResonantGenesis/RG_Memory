"""
pgvector Integration Service
Fast vector similarity search using PostgreSQL pgvector extension.

Provides 10x faster similarity search compared to linear scan.
Requires: PostgreSQL with pgvector extension installed.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    memory_id: str
    content: str
    similarity: float
    hash: Optional[str] = None
    xyz: Optional[Tuple[float, float, float]] = None
    resonance_score: Optional[float] = None
    metadata: Optional[Dict] = None


class PgVectorSearch:
    """
    pgvector-based similarity search.
    
    Uses IVFFlat index for approximate nearest neighbor search.
    Falls back to exact search if pgvector is not available.
    """
    
    def __init__(self):
        self._pgvector_available: Optional[bool] = None
    
    async def check_pgvector_available(self, session: AsyncSession) -> bool:
        """Check if pgvector extension is installed."""
        if self._pgvector_available is not None:
            return self._pgvector_available
        
        try:
            result = await session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            self._pgvector_available = result.scalar() is not None
            
            if self._pgvector_available:
                logger.info("✅ pgvector extension detected")
            else:
                logger.warning("⚠️ pgvector extension not installed, using fallback")
            
            return self._pgvector_available
        except Exception as e:
            logger.warning(f"⚠️ Could not check pgvector: {e}")
            self._pgvector_available = False
            return False
    
    async def search_similar(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        agent_hash: Optional[str] = None,
        require_agent_hash_null: bool = False,
        require_user_id_null: bool = False,
        limit: int = 20,
        min_similarity: float = 0.0
    ) -> List[VectorSearchResult]:
        """
        Search for similar memories using vector similarity.
        
        Uses pgvector if available, otherwise falls back to application-level search.
        
        Args:
            session: Database session
            query_embedding: Query embedding vector
            user_id: Optional user ID filter
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of VectorSearchResult sorted by similarity (descending)
        """
        is_pgvector = await self.check_pgvector_available(session)
        
        if is_pgvector:
            return await self._search_pgvector(
                session,
                query_embedding,
                user_id,
                org_id,
                agent_hash,
                require_agent_hash_null,
                require_user_id_null,
                limit,
                min_similarity,
            )
        else:
            return await self._search_fallback(
                session,
                query_embedding,
                user_id,
                org_id,
                agent_hash,
                require_agent_hash_null,
                require_user_id_null,
                limit,
                min_similarity,
            )
    
    async def _search_pgvector(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        user_id: Optional[str],
        org_id: Optional[str],
        agent_hash: Optional[str],
        require_agent_hash_null: bool,
        require_user_id_null: bool,
        limit: int,
        min_similarity: float
    ) -> List[VectorSearchResult]:
        """Search using pgvector's native vector operations."""
        try:
            # Convert embedding to pgvector format
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Build query with optional scope filters
            user_filter = "AND me.user_id = :user_id" if user_id else ""
            org_filter = "AND me.org_id = :org_id AND mr.org_id = :org_id" if org_id else ""
            agent_filter = "AND mr.agent_hash = :agent_hash" if agent_hash else ""
            agent_hash_null_filter = "AND mr.agent_hash IS NULL" if require_agent_hash_null else ""
            user_id_null_filter = "AND mr.user_id IS NULL" if require_user_id_null else ""
            
            query = text(f"""
                SELECT 
                    me.memory_id::text as memory_id,
                    mr.content,
                    mr.hash,
                    mr.user_id::text as record_user_id,
                    mr.agent_hash as record_agent_hash,
                    mr.xyz_x,
                    mr.xyz_y,
                    mr.xyz_z,
                    mr.resonance_score,
                    mr.extra_metadata,
                    1 - (me.embedding <=> :query_embedding::vector) as similarity
                FROM memory_embeddings me
                JOIN memory_records mr ON me.memory_id = mr.id
                WHERE 1=1 {user_filter} {org_filter} {agent_filter} {agent_hash_null_filter} {user_id_null_filter}
                    AND (mr.extra_metadata IS NULL 
                         OR mr.extra_metadata->>'is_archived' IS NULL 
                         OR mr.extra_metadata->>'is_archived' = 'false')
                ORDER BY me.embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            
            params = {
                "query_embedding": embedding_str,
                "limit": limit,
            }
            if user_id:
                params["user_id"] = user_id
            if org_id:
                params["org_id"] = org_id
            if agent_hash:
                params["agent_hash"] = agent_hash
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                similarity = float(row.similarity) if row.similarity else 0.0
                if similarity >= min_similarity:
                    results.append(VectorSearchResult(
                        memory_id=row.memory_id,
                        content=row.content,
                        similarity=similarity,
                        hash=row.hash,
                        xyz=(row.xyz_x, row.xyz_y, row.xyz_z) if row.xyz_x is not None else None,
                        resonance_score=row.resonance_score,
                        metadata={
                            **(row.extra_metadata or {}),
                            "record_user_id": row.record_user_id,
                            "record_agent_hash": row.record_agent_hash,
                        },
                    ))
            
            logger.debug(f"pgvector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"pgvector search failed: {e}, falling back")
            return await self._search_fallback(
                session,
                query_embedding,
                user_id,
                org_id,
                agent_hash,
                require_agent_hash_null,
                require_user_id_null,
                limit,
                min_similarity,
            )
    
    async def _search_fallback(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        user_id: Optional[str],
        org_id: Optional[str],
        agent_hash: Optional[str],
        require_agent_hash_null: bool,
        require_user_id_null: bool,
        limit: int,
        min_similarity: float
    ) -> List[VectorSearchResult]:
        """Fallback search using application-level cosine similarity."""
        from ..embeddings import embeddings_generator
        from ..models import MemoryEmbedding, MemoryRecord
        from sqlalchemy import select
        
        # Get all embeddings for user
        stmt = select(MemoryEmbedding)
        if user_id:
            stmt = stmt.where(MemoryEmbedding.user_id == user_id)
        if org_id:
            stmt = stmt.where(MemoryEmbedding.org_id == org_id)
        
        result = await session.execute(stmt)
        embeddings = result.scalars().all()
        
        # Calculate similarities
        similarities = []
        for emb in embeddings:
            similarity = embeddings_generator.cosine_similarity(
                query_embedding, emb.embedding
            )
            if similarity >= min_similarity:
                similarities.append((emb.memory_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:limit]
        
        if not top_results:
            return []
        
        # Fetch memory records
        memory_ids = [mid for mid, _ in top_results]
        similarity_map = {mid: sim for mid, sim in top_results}
        
        stmt = select(MemoryRecord).where(MemoryRecord.id.in_(memory_ids))
        if org_id:
            stmt = stmt.where(MemoryRecord.org_id == org_id)
        if agent_hash:
            stmt = stmt.where(MemoryRecord.agent_hash == agent_hash)
        if require_agent_hash_null:
            stmt = stmt.where(MemoryRecord.agent_hash.is_(None))
        if require_user_id_null:
            stmt = stmt.where(MemoryRecord.user_id.is_(None))
        result = await session.execute(stmt)
        records = result.scalars().all()
        
        # Filter archived and build results
        from .memory_encryption import decrypt_memory_content
        
        results = []
        for record in records:
            if record.extra_metadata and record.extra_metadata.get("is_archived"):
                continue
            
            results.append(VectorSearchResult(
                memory_id=str(record.id),
                content=decrypt_memory_content(record.content),
                similarity=similarity_map.get(record.id, 0.0),
                hash=record.hash,
                xyz=(record.xyz_x, record.xyz_y, record.xyz_z) if record.xyz_x is not None else None,
                resonance_score=record.resonance_score,
                metadata={
                    **(record.extra_metadata or {}),
                    "record_user_id": str(record.user_id) if record.user_id else None,
                    "record_agent_hash": record.agent_hash,
                },
            ))
        
        # Sort by similarity (records may be out of order from DB)
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        logger.debug(f"Fallback search returned {len(results)} results")
        return results
    
    async def search_multi_scope(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        agent_hash: Optional[str] = None,
        limit: int = 20,
        min_similarity: float = 0.0,
    ) -> Dict[str, List[VectorSearchResult]]:
        """
        Combined 3-scope pgvector search in a single UNION ALL query.
        
        Scopes:
          - user_overlay: user_id + agent_hash (personal agent memories)
          - user_global: user_id + agent_hash IS NULL (personal non-agent memories)
          - agent_global: org_id + agent_hash + user_id IS NULL (shared agent memories)
        
        Returns dict keyed by scope name → list of VectorSearchResult.
        """
        is_pgvector = await self.check_pgvector_available(session)
        if not is_pgvector:
            # Fall back to 3 separate calls
            result: Dict[str, List[VectorSearchResult]] = {}
            if user_id and agent_hash:
                result["user_overlay"] = await self._search_fallback(
                    session, query_embedding, user_id, org_id, agent_hash, False, False, limit, min_similarity)
            if user_id:
                result["user_global"] = await self._search_fallback(
                    session, query_embedding, user_id, org_id, None, True, False, limit, min_similarity)
            if agent_hash and org_id:
                result["agent_global"] = await self._search_fallback(
                    session, query_embedding, None, org_id, agent_hash, False, True, limit, min_similarity)
            return result

        try:
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            archive_filter = (
                "AND (mr.extra_metadata IS NULL "
                "OR mr.extra_metadata->>'is_archived' IS NULL "
                "OR mr.extra_metadata->>'is_archived' = 'false')"
            )

            scopes = []
            params: Dict = {"query_embedding": embedding_str, "lim": limit}

            if user_id and agent_hash:
                scopes.append(f"""
                    SELECT 'user_overlay' AS scope, me.memory_id::text, mr.content, mr.hash,
                           mr.user_id::text AS record_user_id, mr.agent_hash AS record_agent_hash,
                           mr.xyz_x, mr.xyz_y, mr.xyz_z, mr.resonance_score, mr.extra_metadata,
                           1 - (me.embedding <=> :query_embedding::vector) AS similarity
                    FROM memory_embeddings me
                    JOIN memory_records mr ON me.memory_id = mr.id
                    WHERE me.user_id = :uid AND mr.agent_hash = :ahash {archive_filter}
                    ORDER BY me.embedding <=> :query_embedding::vector LIMIT :lim
                """)
                params["uid"] = user_id
                params["ahash"] = agent_hash

            if user_id:
                scopes.append(f"""
                    SELECT 'user_global' AS scope, me.memory_id::text, mr.content, mr.hash,
                           mr.user_id::text AS record_user_id, mr.agent_hash AS record_agent_hash,
                           mr.xyz_x, mr.xyz_y, mr.xyz_z, mr.resonance_score, mr.extra_metadata,
                           1 - (me.embedding <=> :query_embedding::vector) AS similarity
                    FROM memory_embeddings me
                    JOIN memory_records mr ON me.memory_id = mr.id
                    WHERE me.user_id = :uid2 AND mr.agent_hash IS NULL {archive_filter}
                    ORDER BY me.embedding <=> :query_embedding::vector LIMIT :lim
                """)
                params["uid2"] = user_id

            if agent_hash and org_id:
                scopes.append(f"""
                    SELECT 'agent_global' AS scope, me.memory_id::text, mr.content, mr.hash,
                           mr.user_id::text AS record_user_id, mr.agent_hash AS record_agent_hash,
                           mr.xyz_x, mr.xyz_y, mr.xyz_z, mr.resonance_score, mr.extra_metadata,
                           1 - (me.embedding <=> :query_embedding::vector) AS similarity
                    FROM memory_embeddings me
                    JOIN memory_records mr ON me.memory_id = mr.id
                    WHERE me.org_id = :oid AND mr.agent_hash = :ahash_g AND mr.user_id IS NULL {archive_filter}
                    ORDER BY me.embedding <=> :query_embedding::vector LIMIT :lim
                """)
                params["oid"] = org_id
                params["ahash_g"] = agent_hash

            if not scopes:
                return {}

            union_query = " UNION ALL ".join(scopes)
            result = await session.execute(text(union_query), params)
            rows = result.fetchall()

            grouped: Dict[str, List[VectorSearchResult]] = {}
            for row in rows:
                scope = row.scope
                similarity = float(row.similarity) if row.similarity else 0.0
                if similarity < min_similarity:
                    continue
                vsr = VectorSearchResult(
                    memory_id=row.memory_id,
                    content=row.content,
                    similarity=similarity,
                    hash=row.hash,
                    xyz=(row.xyz_x, row.xyz_y, row.xyz_z) if row.xyz_x is not None else None,
                    resonance_score=row.resonance_score,
                    metadata={
                        **(row.extra_metadata or {}),
                        "record_user_id": row.record_user_id,
                        "record_agent_hash": row.record_agent_hash,
                    },
                )
                grouped.setdefault(scope, []).append(vsr)

            logger.debug(f"Multi-scope search: {', '.join(f'{k}={len(v)}' for k, v in grouped.items())}")
            return grouped

        except Exception as e:
            logger.error(f"Multi-scope pgvector search failed: {e}")
            return {}

    async def create_vector_index(
        self,
        session: AsyncSession,
        lists: int = 100
    ) -> bool:
        """
        Create IVFFlat index for faster approximate search.
        
        Args:
            session: Database session
            lists: Number of lists for IVFFlat (higher = more accurate, slower build)
            
        Returns:
            True if index created successfully
        """
        try:
            # Check if pgvector is available
            if not await self.check_pgvector_available(session):
                logger.warning("Cannot create vector index: pgvector not available")
                return False
            
            # Create IVFFlat index
            await session.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_memory_embedding_ivfflat
                ON memory_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """))
            
            await session.commit()
            logger.info(f"✅ Created IVFFlat index with {lists} lists")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False
    
    async def get_index_stats(self, session: AsyncSession) -> Dict:
        """Get statistics about vector indexes."""
        try:
            result = await session.execute(text("""
                SELECT 
                    indexrelname as indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size,
                    idx_scan as scans,
                    idx_tup_read as tuples_read
                FROM pg_stat_user_indexes
                WHERE relname = 'memory_embeddings'
            """))
            
            indexes = []
            for row in result.fetchall():
                indexes.append({
                    "name": row.indexname,
                    "size": row.size,
                    "scans": row.scans,
                    "tuples_read": row.tuples_read,
                })
            
            return {
                "pgvector_available": self._pgvector_available,
                "indexes": indexes,
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}


# Global singleton
pgvector_search = PgVectorSearch()
