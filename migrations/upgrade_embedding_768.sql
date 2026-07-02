-- Migration: base-embedder upgrade MiniLM-384 → BGE-base-768
-- Date: 2026-07-02
-- The old 384-dim vectors are from a different model and cannot be converted;
-- they are cleared and every memory is RE-EMBEDDED with the new model (via the
-- /memory/reembed-all endpoint after deploy). memory_records (content) are
-- untouched — only the derived embedding index changes.

-- 1. Drop the HNSW index (dimension changes)
DROP INDEX IF EXISTS ix_memory_embeddings_embedding_hnsw;

-- 2. Clear old-model embeddings (records keep their content, re-embedded next)
TRUNCATE TABLE memory_embeddings;

-- 3. Convert the column to the new dimension
ALTER TABLE memory_embeddings
    ALTER COLUMN embedding TYPE vector(768);

-- 4. (Re-embedding repopulates rows; HNSW index recreated afterwards by the app
--    or a follow-up create-index call.)
