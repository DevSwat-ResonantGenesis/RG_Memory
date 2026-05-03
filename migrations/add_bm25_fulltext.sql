-- Migration: Add BM25 full-text search + content_hash dedup to memory_records
-- Date: 2026-05-02
-- Purpose: Enable keyword search alongside pgvector semantic search + dedup

-- 0. Add content_hash for deduplication
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
CREATE INDEX IF NOT EXISTS ix_memory_records_content_hash ON memory_records(content_hash);

-- 1. Add tsvector column
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS search_tsv tsvector;

-- 2. Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS ix_memory_records_search_tsv
ON memory_records USING gin(search_tsv);

-- 3. Backfill existing records
UPDATE memory_records
SET search_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE search_tsv IS NULL;

-- 4. Create trigger to auto-populate on INSERT/UPDATE
CREATE OR REPLACE FUNCTION memory_records_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memory_records_tsv ON memory_records;
CREATE TRIGGER trg_memory_records_tsv
BEFORE INSERT OR UPDATE OF content ON memory_records
FOR EACH ROW EXECUTE FUNCTION memory_records_tsv_trigger();
