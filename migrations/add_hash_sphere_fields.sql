-- Migration: Add Full Hash Sphere Coordinate System Fields
-- Date: 2026-01-09
-- Description: Adds all Hash Sphere coordinate fields to memory_records table

-- Layer 2: Hash Generation
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS meaning_hash VARCHAR(64);
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS energy_hash VARCHAR(64);
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS spin_hash VARCHAR(64);

-- Layer 3: Universe ID
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS universe_id VARCHAR(64);
CREATE INDEX IF NOT EXISTS idx_memory_records_universe_id ON memory_records(universe_id);

-- Hyperspherical Coordinates
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS sphere_r FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS sphere_phi FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS sphere_theta FLOAT;

-- Layer 6: Resonance Scoring (normalized)
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS normalized_resonance FLOAT;

-- Anchor Energy
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS anchor_energy FLOAT;

-- Spin Vector (semantic rotation)
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS spin_x FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS spin_y FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS spin_z FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS spin_magnitude FLOAT;

-- Semantic Components
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS meaning_score FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS intensity_score FLOAT;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS sentiment_score FLOAT;

-- Cluster Assignment
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS cluster_id UUID;
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS cluster_name VARCHAR(255);
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS cluster_distance FLOAT;
CREATE INDEX IF NOT EXISTS idx_memory_records_cluster_id ON memory_records(cluster_id);

-- Full coordinates as JSON (for backward compatibility)
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS hash_sphere_coords JSONB;

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memory_records_normalized_resonance ON memory_records(normalized_resonance);
CREATE INDEX IF NOT EXISTS idx_memory_records_anchor_energy ON memory_records(anchor_energy);

-- Update existing records with default values for new fields
-- This will set normalized_resonance based on existing resonance_score
UPDATE memory_records 
SET normalized_resonance = 1.0 / (1.0 + EXP(-COALESCE(resonance_score, 0)))
WHERE normalized_resonance IS NULL AND resonance_score IS NOT NULL;

-- Set default sentiment_score to neutral (0.5)
UPDATE memory_records 
SET sentiment_score = 0.5
WHERE sentiment_score IS NULL;

-- Set default meaning_score based on content length
UPDATE memory_records 
SET meaning_score = LEAST(1.0, LN(1 + LENGTH(content)) / 10)
WHERE meaning_score IS NULL;

COMMENT ON COLUMN memory_records.meaning_hash IS 'Semantic meaning hash (20 chars)';
COMMENT ON COLUMN memory_records.energy_hash IS 'Emotional intensity hash (8 chars)';
COMMENT ON COLUMN memory_records.spin_hash IS 'Direction/intent hash (8 chars)';
COMMENT ON COLUMN memory_records.universe_id IS 'SHA-256 universe ID (64 chars)';
COMMENT ON COLUMN memory_records.sphere_r IS 'Hyperspherical radius (should be ~1 for unit sphere)';
COMMENT ON COLUMN memory_records.sphere_phi IS 'Hyperspherical latitude in radians';
COMMENT ON COLUMN memory_records.sphere_theta IS 'Hyperspherical longitude in radians';
COMMENT ON COLUMN memory_records.normalized_resonance IS 'Resonance score normalized to 0-1 using sigmoid';
COMMENT ON COLUMN memory_records.anchor_energy IS 'Anchor attraction energy E_j(s) = exp(-β·||s - A_j||²)';
COMMENT ON COLUMN memory_records.spin_x IS 'Spin vector X component (topic direction)';
COMMENT ON COLUMN memory_records.spin_y IS 'Spin vector Y component (emotional valence)';
COMMENT ON COLUMN memory_records.spin_z IS 'Spin vector Z component (complexity)';
COMMENT ON COLUMN memory_records.spin_magnitude IS 'Magnitude of spin vector';
COMMENT ON COLUMN memory_records.meaning_score IS 'Content richness score (0-1)';
COMMENT ON COLUMN memory_records.intensity_score IS 'Emotional intensity (0-1)';
COMMENT ON COLUMN memory_records.sentiment_score IS 'Sentiment: 0=negative, 0.5=neutral, 1=positive';
COMMENT ON COLUMN memory_records.cluster_id IS 'Assigned cluster UUID';
COMMENT ON COLUMN memory_records.cluster_name IS 'Assigned cluster name';
COMMENT ON COLUMN memory_records.cluster_distance IS 'Distance to cluster centroid';
COMMENT ON COLUMN memory_records.hash_sphere_coords IS 'Full HashSphereCoordinates as JSON';
