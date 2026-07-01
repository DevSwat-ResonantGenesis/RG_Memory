-- Migration: Convert memory_embeddings.embedding to native pgvector(384)
-- Date: 2026-07-01
-- Purpose: Enable pgvector <=> cosine operator. The column was double precision[]
--          with mixed dimensions (384 MiniLM / 512 Nomic / 1536 OpenAI), so the
--          <=> operator failed and all vector search fell back to linear scan.
-- Strategy: DROP the incompatible non-384 embeddings (source memory_records are
--           retained and can be re-embedded later), then convert to vector(384).

-- 0. Safety: ensure pgvector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Delete embeddings that are not 384-dim (old Nomic 512 + OpenAI 1536 models).
--    Their memory_records stay intact; only the stale embedding vectors are removed.
DELETE FROM memory_embeddings
WHERE array_length(embedding, 1) IS DISTINCT FROM 384;

-- 2. Convert the column type double precision[] -> vector(384)
ALTER TABLE memory_embeddings
    ALTER COLUMN embedding TYPE vector(384)
    USING embedding::vector(384);

-- 3. Backfill model/dimensions metadata for the surviving rows
UPDATE memory_embeddings
SET model = 'all-MiniLM-L6-v2', dimensions = 384
WHERE dimensions IS DISTINCT FROM 384;

-- 4. HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS ix_memory_embeddings_embedding_hnsw
    ON memory_embeddings USING hnsw (embedding vector_cosine_ops);
