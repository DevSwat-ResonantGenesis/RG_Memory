-- Migration: RFC-0002 Wave 3c — self-organizing mesh edges
-- Date: 2026-07-01
CREATE TABLE IF NOT EXISTS memory_edges (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID,
    org_id            UUID,
    agent_hash        VARCHAR(64),
    src_id            UUID NOT NULL,
    dst_id            UUID NOT NULL,
    weight            DOUBLE PRECISION DEFAULT 0.1,
    coretrieval_count INTEGER DEFAULT 1,
    last_reinforced   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_memory_edges_user_id ON memory_edges(user_id);
CREATE INDEX IF NOT EXISTS ix_memory_edges_src_id ON memory_edges(src_id);
CREATE INDEX IF NOT EXISTS ix_memory_edges_dst_id ON memory_edges(dst_id);
CREATE INDEX IF NOT EXISTS ix_memory_edges_agent_hash ON memory_edges(agent_hash);
CREATE UNIQUE INDEX IF NOT EXISTS ix_memory_edges_pair ON memory_edges(user_id, src_id, dst_id);
