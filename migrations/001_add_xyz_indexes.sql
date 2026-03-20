-- Migration: Add XYZ coordinate indexes for faster proximity queries
-- Run this against your PostgreSQL database
-- Date: January 2026

-- Composite index for XYZ proximity queries (most important)
CREATE INDEX IF NOT EXISTS idx_memory_xyz_composite 
ON memory_records(xyz_x, xyz_y, xyz_z);

-- Individual indexes for range queries
CREATE INDEX IF NOT EXISTS idx_memory_xyz_x ON memory_records(xyz_x);
CREATE INDEX IF NOT EXISTS idx_memory_xyz_y ON memory_records(xyz_y);
CREATE INDEX IF NOT EXISTS idx_memory_xyz_z ON memory_records(xyz_z);

-- Index for resonance score sorting
CREATE INDEX IF NOT EXISTS idx_memory_resonance ON memory_records(resonance_score DESC);

-- Index for user_id + created_at (common query pattern)
CREATE INDEX IF NOT EXISTS idx_memory_user_created 
ON memory_records(user_id, created_at DESC);

-- Index for hash lookups
CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory_records(hash);

-- Partial index for active (non-archived) records
-- This speeds up queries that filter out archived records
CREATE INDEX IF NOT EXISTS idx_memory_active 
ON memory_records(user_id, created_at DESC) 
WHERE (extra_metadata IS NULL OR extra_metadata->>'is_archived' IS NULL OR extra_metadata->>'is_archived' = 'false');

-- Embedding table indexes
CREATE INDEX IF NOT EXISTS idx_embedding_user ON memory_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_embedding_memory ON memory_embeddings(memory_id);

-- Analyze tables to update statistics for query planner
ANALYZE memory_records;
ANALYZE memory_embeddings;

-- Verify indexes were created
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename IN ('memory_records', 'memory_embeddings')
ORDER BY tablename, indexname;
