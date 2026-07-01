-- Migration: RFC-0002 Wave 1 clean-slate
-- Date: 2026-07-01
-- Purpose: Legacy memory data was stored with the wrong model (hash-trig XYZ used
--          for retrieval, mixed-dim embeddings, no 12-D core). Per owner decision
--          there is no real data to preserve. Wipe so every NEW memory is written
--          with a proper 12-D hash-sphere core. Immutability applies GOING FORWARD.

TRUNCATE TABLE
    memory_facts,
    memory_embeddings,
    memory_anchors,
    memory_chunks,
    memory_records
RESTART IDENTITY;

-- resonance_clusters are derived; clear them too so they recompute from fresh data.
TRUNCATE TABLE resonance_clusters RESTART IDENTITY;
