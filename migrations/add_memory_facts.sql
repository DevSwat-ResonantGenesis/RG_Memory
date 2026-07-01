-- Migration: Tier 2 — atomic fact extraction (memory_facts table)
-- Date: 2026-07-01
-- Purpose: Store LLM-extracted subject-attribute-value facts with confidence
--          and a supersede lifecycle for contradiction detection.

CREATE TABLE IF NOT EXISTS memory_facts (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id     UUID,
    user_id       UUID,
    org_id        UUID,
    agent_hash    VARCHAR(64),
    fact          TEXT NOT NULL,
    entity        VARCHAR(255),
    attribute     VARCHAR(255),
    value         TEXT,
    confidence    DOUBLE PRECISION DEFAULT 0.5,
    status        VARCHAR(16) DEFAULT 'active',
    superseded_by UUID,
    fact_hash     VARCHAR(64),
    extra_metadata JSON,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_memory_facts_memory_id ON memory_facts(memory_id);
CREATE INDEX IF NOT EXISTS ix_memory_facts_user_id ON memory_facts(user_id);
CREATE INDEX IF NOT EXISTS ix_memory_facts_org_id ON memory_facts(org_id);
CREATE INDEX IF NOT EXISTS ix_memory_facts_agent_hash ON memory_facts(agent_hash);
CREATE INDEX IF NOT EXISTS ix_memory_facts_entity ON memory_facts(entity);
CREATE INDEX IF NOT EXISTS ix_memory_facts_attribute ON memory_facts(attribute);
CREATE INDEX IF NOT EXISTS ix_memory_facts_status ON memory_facts(status);
CREATE INDEX IF NOT EXISTS ix_memory_facts_fact_hash ON memory_facts(fact_hash);
CREATE INDEX IF NOT EXISTS ix_memory_facts_user_entity_attr ON memory_facts(user_id, entity, attribute);
