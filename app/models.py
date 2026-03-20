"""Memory Service Models - Full old backend compatibility.

Includes:
- MemoryRecord: Basic memory storage
- MemoryEmbedding: Vector embeddings for similarity search
- MemoryChunk: Chunked memory for long documents
- MemoryAnchor: Key memory points with Hash Sphere coordinates
- ResonanceCluster: Grouped memories by resonance
"""
from datetime import datetime
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func

from .db import Base


class MemoryRecord(Base):
    """Basic memory storage with full Hash Sphere coordinate system."""
    __tablename__ = "memory_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=True)  # Multi-tenant
    source = Column(String(64), nullable=False)  # chat, workflow, cognitive, etc.
    content = Column(Text, nullable=False)
    
    # ========== HASH SPHERE COORDINATE SYSTEM ==========
    
    # Layer 2: Hash Generation
    hash = Column(String(255), index=True, nullable=True)  # Unique resonance hash
    meaning_hash = Column(String(64), nullable=True)  # Semantic meaning hash
    energy_hash = Column(String(64), nullable=True)  # Emotional intensity hash
    spin_hash = Column(String(64), nullable=True)  # Direction/intent hash
    
    # Layer 3: Universe ID
    universe_id = Column(String(64), index=True, nullable=True)  # SHA-256 universe ID
    
    # Layer 5: Cartesian Coordinates (3D semantic space)
    xyz_x = Column(Float, nullable=True)  # X coordinate
    xyz_y = Column(Float, nullable=True)  # Y coordinate
    xyz_z = Column(Float, nullable=True)  # Z coordinate
    
    # Hyperspherical Coordinates
    sphere_r = Column(Float, nullable=True)  # Radius (should be ~1 for unit sphere)
    sphere_phi = Column(Float, nullable=True)  # Latitude in radians
    sphere_theta = Column(Float, nullable=True)  # Longitude in radians
    
    # Layer 6: Resonance Scoring
    resonance_score = Column(Float, nullable=True)  # R(h) = sin(a·x) + cos(b·y) + tan(c·z)
    normalized_resonance = Column(Float, nullable=True)  # Normalized to 0-1
    
    # Anchor Energy
    anchor_energy = Column(Float, nullable=True)  # E_j(s) = exp(-β·||s - A_j||²)
    
    # Spin Vector (semantic rotation)
    spin_x = Column(Float, nullable=True)
    spin_y = Column(Float, nullable=True)
    spin_z = Column(Float, nullable=True)
    spin_magnitude = Column(Float, nullable=True)
    
    # Semantic Components
    meaning_score = Column(Float, nullable=True)  # Content richness score
    intensity_score = Column(Float, nullable=True)  # Emotional intensity
    sentiment_score = Column(Float, nullable=True)  # Positive/negative sentiment
    
    # Cluster Assignment
    cluster_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    cluster_name = Column(String(255), nullable=True)
    cluster_distance = Column(Float, nullable=True)
    
    # ========== END HASH SPHERE FIELDS ==========
    
    # Shared agent support
    agent_hash = Column(String(64), index=True, nullable=True)
    
    # Full coordinates as JSON (for backward compatibility)
    hash_sphere_coords = Column(JSON, nullable=True)  # Full HashSphereCoordinates dict
    
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class MemoryEmbedding(Base):
    """Store embeddings for vector similarity search."""
    __tablename__ = "memory_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    embedding = Column(ARRAY(Float), nullable=False)  # Vector embedding
    model = Column(String(64), default="text-embedding-3-small")
    dimensions = Column(Integer, default=1536)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MemoryChunk(Base):
    """Chunked memory for long documents."""
    __tablename__ = "memory_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MemoryAnchor(Base):
    """Memory anchor - key memory points extracted from conversations and code.
    
    Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/models/governance/resonant_chat.py
    
    Note: Hash Sphere is immutable - data cannot be deleted once hashed.
    Use is_archived to hide anchors instead of deleting them.
    """
    __tablename__ = "memory_anchors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    message_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    
    # Anchor content
    anchor_text = Column(Text, nullable=False)  # The anchor text/key phrase
    anchor_hash = Column(String(255), index=True, nullable=False)  # Hash of anchor
    context = Column(Text, default="")  # Context around the anchor
    importance_score = Column(Float, default=0.5)  # Importance (0-1)
    
    # Hash Sphere 3D coordinates
    xyz_x = Column(Float, nullable=True)
    xyz_y = Column(Float, nullable=True)
    xyz_z = Column(Float, nullable=True)
    
    # Type discrimination
    anchor_type = Column(String(50), index=True, default="chat")  # chat, code, function, pattern
    
    # Archive flag - Hash Sphere is immutable, use this to hide anchors
    is_archived = Column(Boolean, default=False, index=True)  # Archived anchors won't be loaded
    archived_at = Column(DateTime(timezone=True), nullable=True)  # When it was archived
    
    # Code-specific fields (optional, NULL for chat anchors)
    file_path = Column(String(500), index=True, nullable=True)
    function_name = Column(String(255), index=True, nullable=True)
    language = Column(String(50), nullable=True)  # python, typescript, etc.
    line_range = Column(JSON, nullable=True)  # {start: int, end: int}
    code_snippet = Column(Text, nullable=True)  # Full code snippet
    
    # Deterministic Anchor Universes (Phase 2)
    universe_id = Column(String(32), index=True, nullable=True)
    deterministic = Column(Boolean, default=False)
    seed_hash = Column(String(64), nullable=True)  # Hash of seed (not the seed itself)
    
    # Shared agent support
    agent_hash = Column(String(64), index=True, nullable=True)
    
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ResonanceCluster(Base):
    """Resonance cluster - grouped memories by resonance.
    
    Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/models/governance/resonant_chat.py
    """
    __tablename__ = "resonance_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    
    cluster_name = Column(String(255), nullable=False)
    cluster_hash = Column(String(255), index=True, nullable=False)
    anchor_ids = Column(JSON, default=list)  # List of anchor IDs
    resonance_score = Column(Float, default=0.0)  # Overall cluster resonance
    personality_traits = Column(JSON, default=dict)  # Personality traits
    
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
