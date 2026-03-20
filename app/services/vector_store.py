"""
Vector Store Service with pgvector Integration
===============================================

Provides high-performance vector similarity search using PostgreSQL pgvector extension.
This enables fast semantic search for memory retrieval.
"""
from __future__ import annotations

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector Store using pgvector for efficient similarity search.
    
    Features:
    - Store embeddings with metadata
    - Cosine similarity search
    - L2 distance search
    - Inner product search
    - Batch operations
    - Index management
    """
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (default 1536 for OpenAI ada-002)
        """
        self.dimension = dimension
        self._initialized = False
    
    async def initialize(self, session: AsyncSession) -> bool:
        """
        Initialize pgvector extension and create tables if needed.
        
        Returns:
            True if initialization successful
        """
        try:
            # Enable pgvector extension
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create embeddings table if not exists
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS memory_vectors (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                org_id UUID,
                chat_id UUID,
                message_id UUID,
                content TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                embedding vector({self.dimension}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
            await session.execute(text(create_table_sql))
            
            # Create indexes for fast similarity search
            # IVFFlat index for approximate nearest neighbor search
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS memory_vectors_embedding_idx 
            ON memory_vectors 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
            try:
                await session.execute(text(index_sql))
            except Exception as e:
                # Index creation may fail if not enough rows, that's ok
                logger.warning(f"Index creation skipped: {e}")
            
            # Create indexes for filtering
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS memory_vectors_user_idx ON memory_vectors(user_id)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS memory_vectors_chat_idx ON memory_vectors(chat_id)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS memory_vectors_hash_idx ON memory_vectors(content_hash)"
            ))
            
            await session.commit()
            self._initialized = True
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False
    
    def _content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    async def upsert(
        self,
        session: AsyncSession,
        user_id: str,
        content: str,
        embedding: List[float],
        org_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Insert or update a vector embedding.
        
        Args:
            session: Database session
            user_id: User ID
            content: Text content
            embedding: Vector embedding
            org_id: Organization ID (optional)
            chat_id: Chat ID (optional)
            message_id: Message ID (optional)
            metadata: Additional metadata (optional)
        
        Returns:
            ID of the inserted/updated record, or None on failure
        """
        try:
            content_hash = self._content_hash(content)
            normalized_embedding = self._normalize_embedding(embedding)
            
            # Check if content already exists
            check_sql = text("""
                SELECT id FROM memory_vectors 
                WHERE user_id = :user_id AND content_hash = :content_hash
                LIMIT 1
            """)
            result = await session.execute(check_sql, {
                "user_id": user_id,
                "content_hash": content_hash,
            })
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing record
                update_sql = text("""
                    UPDATE memory_vectors 
                    SET embedding = :embedding::vector,
                        metadata = :metadata::jsonb,
                        updated_at = NOW()
                    WHERE id = :id
                    RETURNING id
                """)
                result = await session.execute(update_sql, {
                    "id": str(existing),
                    "embedding": str(normalized_embedding),
                    "metadata": metadata or {},
                })
                record_id = str(existing)
            else:
                # Insert new record
                insert_sql = text("""
                    INSERT INTO memory_vectors 
                    (user_id, org_id, chat_id, message_id, content, content_hash, embedding, metadata)
                    VALUES (:user_id, :org_id, :chat_id, :message_id, :content, :content_hash, :embedding::vector, :metadata::jsonb)
                    RETURNING id
                """)
                result = await session.execute(insert_sql, {
                    "user_id": user_id,
                    "org_id": org_id,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                    "content_hash": content_hash,
                    "embedding": str(normalized_embedding),
                    "metadata": metadata or {},
                })
                record_id = str(result.scalar_one())
            
            await session.commit()
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            await session.rollback()
            return None
    
    async def batch_upsert(
        self,
        session: AsyncSession,
        user_id: str,
        items: List[Dict[str, Any]],
    ) -> int:
        """
        Batch insert/update multiple vectors.
        
        Args:
            session: Database session
            user_id: User ID
            items: List of dicts with 'content', 'embedding', and optional metadata
        
        Returns:
            Number of successfully processed items
        """
        success_count = 0
        for item in items:
            result = await self.upsert(
                session=session,
                user_id=user_id,
                content=item.get("content", ""),
                embedding=item.get("embedding", []),
                org_id=item.get("org_id"),
                chat_id=item.get("chat_id"),
                message_id=item.get("message_id"),
                metadata=item.get("metadata"),
            )
            if result:
                success_count += 1
        return success_count
    
    async def search(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.0,
        search_type: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            session: Database session
            query_embedding: Query vector
            user_id: Filter by user ID (optional)
            org_id: Filter by org ID (optional)
            chat_id: Filter by chat ID (optional)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1 for cosine)
            search_type: 'cosine', 'l2', or 'inner_product'
        
        Returns:
            List of matching records with similarity scores
        """
        try:
            normalized_query = self._normalize_embedding(query_embedding)
            
            # Build distance operator based on search type
            if search_type == "l2":
                distance_op = "<->"
                order_direction = "ASC"
            elif search_type == "inner_product":
                distance_op = "<#>"
                order_direction = "ASC"
            else:  # cosine (default)
                distance_op = "<=>"
                order_direction = "ASC"
            
            # Build WHERE clause
            where_conditions = []
            params = {"query_embedding": str(normalized_query)}
            
            if user_id:
                where_conditions.append("user_id = :user_id")
                params["user_id"] = user_id
            if org_id:
                where_conditions.append("org_id = :org_id")
                params["org_id"] = org_id
            if chat_id:
                where_conditions.append("chat_id = :chat_id")
                params["chat_id"] = chat_id
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Build query
            search_sql = text(f"""
                SELECT 
                    id,
                    user_id,
                    org_id,
                    chat_id,
                    message_id,
                    content,
                    metadata,
                    created_at,
                    1 - (embedding {distance_op} :query_embedding::vector) as similarity
                FROM memory_vectors
                WHERE {where_clause}
                ORDER BY embedding {distance_op} :query_embedding::vector {order_direction}
                LIMIT :top_k
            """)
            params["top_k"] = top_k
            
            result = await session.execute(search_sql, params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                similarity = float(row.similarity) if row.similarity else 0.0
                
                # Apply threshold filter
                if similarity < threshold:
                    continue
                
                results.append({
                    "id": str(row.id),
                    "user_id": str(row.user_id) if row.user_id else None,
                    "org_id": str(row.org_id) if row.org_id else None,
                    "chat_id": str(row.chat_id) if row.chat_id else None,
                    "message_id": str(row.message_id) if row.message_id else None,
                    "content": row.content,
                    "metadata": row.metadata or {},
                    "similarity": similarity,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete(
        self,
        session: AsyncSession,
        record_id: str,
    ) -> bool:
        """Delete a vector by ID."""
        try:
            delete_sql = text("DELETE FROM memory_vectors WHERE id = :id")
            await session.execute(delete_sql, {"id": record_id})
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            await session.rollback()
            return False
    
    async def delete_by_user(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> int:
        """Delete all vectors for a user."""
        try:
            delete_sql = text("DELETE FROM memory_vectors WHERE user_id = :user_id")
            result = await session.execute(delete_sql, {"user_id": user_id})
            await session.commit()
            return result.rowcount
        except Exception as e:
            logger.error(f"Failed to delete user vectors: {e}")
            await session.rollback()
            return 0
    
    async def get_stats(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if user_id:
                count_sql = text(
                    "SELECT COUNT(*) FROM memory_vectors WHERE user_id = :user_id"
                )
                result = await session.execute(count_sql, {"user_id": user_id})
            else:
                count_sql = text("SELECT COUNT(*) FROM memory_vectors")
                result = await session.execute(count_sql)
            
            total_count = result.scalar_one()
            
            return {
                "total_vectors": total_count,
                "dimension": self.dimension,
                "initialized": self._initialized,
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "initialized": self._initialized,
                "error": str(e),
            }


# Global instance
vector_store = VectorStore(dimension=1536)
