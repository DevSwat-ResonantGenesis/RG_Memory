from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ============================================
# CORE MEMORY SCHEMAS
# ============================================

class MemoryIngestRequest(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    generate_embedding: bool = True
    agent_hash: Optional[str] = None
    # Benchmark/bulk mode: skip the per-memory LLM fact-extraction + on-chain
    # anchoring background tasks (they're costly at bulk-ingest scale).
    skip_enrichment: bool = False


class MemoryRecordResponse(BaseModel):
    id: str
    chat_id: Optional[str]
    user_id: Optional[str]
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None

    scope: Optional[str] = None
    tier: Optional[str] = None
    
    # ========== FULL HASH SPHERE COORDINATE SYSTEM ==========
    # Layer 2: Hash Generation
    hash: Optional[str] = None
    meaning_hash: Optional[str] = None
    energy_hash: Optional[str] = None
    spin_hash: Optional[str] = None
    
    # Layer 3: Universe ID
    universe_id: Optional[str] = None
    
    # Layer 5: Cartesian Coordinates (backward compatible)
    xyz: Optional[List[float]] = None
    xyz_x: Optional[float] = None
    xyz_y: Optional[float] = None
    xyz_z: Optional[float] = None
    
    # Hyperspherical Coordinates
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    
    # Layer 6: Resonance Scoring
    resonance_score: Optional[float] = None
    normalized_resonance: Optional[float] = None
    
    # Anchor Energy
    anchor_energy: Optional[float] = None
    
    # Spin Vector
    spin: Optional[Dict[str, Any]] = None
    
    # Semantic Components
    semantic: Optional[Dict[str, float]] = None
    
    # Cluster Assignment
    cluster: Optional[str] = None
    
    # Full coordinates as JSON
    hash_sphere_coords: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MemoryRetrieveRequest(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None
    team_id: Optional[str] = None
    query: str
    limit: int = 5
    use_vector_search: bool = True
    retrieval_mode: str = "embedding"


class MemorySearchResponse(BaseModel):
    memories: List[MemoryRecordResponse]
    query: str
    total_found: int


# ============================================
# HASH SPHERE SCHEMAS
# ============================================

class HashSphereExtractRequest(BaseModel):
    """Request for full Hash Sphere memory extraction."""
    query: str
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    agent_hash: Optional[str] = None
    session_id: Optional[str] = None  # Conversation scope → MemoryRecord.chat_id
    min_score: Optional[float] = None  # Hybrid-score floor (defaults to env MIN_HYBRID_SCORE)
    limit: int = 10
    use_anchors: bool = True
    use_proximity: bool = True
    use_resonance: bool = True
    use_clusters: bool = True
    use_rag_fallback: bool = True
    include_coordinates: bool = True
    apply_magnetic_pull: bool = True


class HashSphereMemory(BaseModel):
    """Memory with full Hash Sphere coordinates and scores."""
    id: str
    content: str
    type: str = "message"
    hash: Optional[str] = None
    xyz: Optional[List[float]] = None
    universe_id: Optional[str] = None
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    hybrid_score: float = 0.0
    rag_score: float = 0.0
    resonance_score: float = 0.0
    proximity_score: float = 0.0
    anchor_score: float = 0.0
    recency_score: float = 0.0
    anchor_energy: float = 0.0
    resonance_function_score: float = 0.0
    magnetic_score: float = 0.0
    gravity_force: float = 0.0
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HashSphereExtractResponse(BaseModel):
    """Response with full Hash Sphere extraction results."""
    memories: List[HashSphereMemory]
    query: str
    query_hash: str
    query_xyz: List[float]
    query_resonance: float
    total_found: int
    extraction_methods_used: List[str]
    extraction_time_ms: float
    # Confidence gate (RFC-0002 Wave 3): when confidence is high the caller can
    # answer directly from memory WITHOUT invoking an LLM ("no-LLM recall").
    confidence: float = 0.0
    answer_from_memory: bool = False
    # Evidence ledger (RFC-0002 Wave 4b): cryptographic provenance hash for a
    # confident recall — which memories+weights justified the answer, anchored
    # on-chain. "Here is why I recalled this."
    evidence_hash: Optional[str] = None


class HashRequest(BaseModel):
    text: str
    context: Optional[str] = None


class HashResponse(BaseModel):
    hash: str
    energy_score: float
    spin_score: float
    anchors: List[str]
    xyz: List[float]


class ResonanceRequest(BaseModel):
    hash1: str
    hash2: str


class ResonanceResponse(BaseModel):
    resonance_score: float
    boosted_score: float
    hash1: str
    hash2: str


class AnchorCreateRequest(BaseModel):
    anchor_text: str
    context: Optional[str] = None
    importance_score: float = 0.5


class AnchorResponse(BaseModel):
    id: str
    anchor_text: str
    anchor_hash: str
    context: str
    importance_score: float
    user_id: Optional[str] = None
    xyz_x: Optional[float] = None
    xyz_y: Optional[float] = None
    xyz_z: Optional[float] = None
    anchor_type: Optional[str] = None
    resonance_score: Optional[float] = None
    created_at: Optional[str] = None
    sphere_r: Optional[float] = None
    sphere_phi: Optional[float] = None
    sphere_theta: Optional[float] = None
    normalized_resonance: Optional[float] = None
    anchor_energy: Optional[float] = None
    spin_x: Optional[float] = None
    spin_y: Optional[float] = None
    spin_z: Optional[float] = None
    spin_magnitude: Optional[float] = None
    meaning_score: Optional[float] = None
    intensity_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    meaning_hash: Optional[str] = None
    energy_hash: Optional[str] = None
    spin_hash: Optional[str] = None
    universe_id: Optional[str] = None
    cluster_name: Optional[str] = None


class AnchorSearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: int = 10


class ArchiveRequest(BaseModel):
    """Request to archive/unarchive a file or anchor."""
    file_path: Optional[str] = None
    anchor_id: Optional[str] = None
    project_id: Optional[str] = None


class ArchiveResponse(BaseModel):
    """Response for archive operations."""
    success: bool
    archived_count: int
    file_path: Optional[str] = None
    message: str


# ============================================
# PUBLIC HASH SPHERE SCHEMAS
# ============================================

class HashSphereTokenRequest(BaseModel):
    """Request for Hash Sphere token."""
    is_owner: bool = False


class HashSphereTokenResponse(BaseModel):
    """Hash Sphere token response."""
    token: str
    expires_at: str
    expires_in_hours: int


# ============================================
# PROJECT FILES SCHEMAS
# ============================================

class ProjectSummaryResponse(BaseModel):
    project_id: str
    name: str
    file_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProjectsResponse(BaseModel):
    projects: List[ProjectSummaryResponse]
    count: int


class ProjectFileResponse(BaseModel):
    path: str
    type: str
    size: Optional[int] = None
    content: Optional[str] = None
    language: Optional[str] = None


class ProjectFilesResponse(BaseModel):
    project_id: str
    files: List[ProjectFileResponse]
    total: int


# ============================================
# RAG SCHEMAS
# ============================================

class RAGFileUploadResponse(BaseModel):
    """Response for file upload."""
    id: str
    filename: str
    content_type: str
    size: int
    memories_created: int
    chunks: int


class RAGMemoryCreateRequest(BaseModel):
    """RAG memory creation request - matches old backend."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    is_shared: bool = False
    is_public: bool = False
    shared_with: Optional[List[str]] = None
    language: Optional[str] = None


class RAGMemoryResponse(BaseModel):
    """RAG memory response - matches old backend."""
    id: str
    content: str
    hash: Optional[str] = None
    xyz: Optional[List[float]] = None
    cluster: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: str
    is_shared: Optional[bool] = False
    shared_with: Optional[List[str]] = None
    is_public: Optional[bool] = False


class RAGMemoryUpdateRequest(BaseModel):
    """RAG memory update request."""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGAskRequest(BaseModel):
    """RAG ask request - matches old backend."""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    use_memory: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None


class RAGAskResponse(BaseModel):
    """RAG ask response - matches old backend."""
    response: str
    sources: List[Dict[str, Any]]
    validity: float
    entropy: float
    evidence_graph: Dict[str, Any]
    context_used: bool
    conversation_id: str


class RAGConversationResponse(BaseModel):
    """RAG conversation response."""
    id: str
    role: str
    content: str
    provider: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    validity: Optional[float] = None
    created_at: str


# ============================================
# AGENT MEMORY SCHEMAS
# ============================================

class AgentMemoryCreateRequest(BaseModel):
    agent_id: str
    type: str
    content: str
    importance: float = 0.5
    tags: Optional[List[str]] = None
    source: Optional[str] = None


class AgentMemoryResponse(BaseModel):
    id: str
    agent_id: str
    type: str
    content: str
    importance: float
    timestamp: str
    tokens: Optional[int] = None
    tags: Optional[List[str]] = None


class SemanticSearchRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5
    threshold: float = 0.75
    memory_types: Optional[List[str]] = None


class SemanticSearchResult(BaseModel):
    memory: AgentMemoryResponse
    score: float
    relevance: str


class SemanticSearchResponse(BaseModel):
    results: List[SemanticSearchResult]
    query: str
    total_found: int


class ConsolidateRequest(BaseModel):
    agent_id: str
    threshold: int = 100


class MemorySettingsRequest(BaseModel):
    agent_id: str
    settings: Dict[str, Any]


class MemoryClusterResponse(BaseModel):
    id: str
    name: str
    memories: List[str]
    coherence: float


class MemoryAnalyticsResponse(BaseModel):
    total_memories: int
    by_type: Dict[str, int]
    avg_importance: float
    storage_used: float
    retrieval_latency: float
    consolidation_rate: float


# ============================================
# 3D VISUALIZATION SCHEMAS
# ============================================

class MemoryNodeResponse(BaseModel):
    """Memory node for 3D visualization."""
    id: str
    content: str
    title: Optional[str] = None
    x: float
    y: float
    z: float
    layer: str = "active"
    cluster: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    tags: List[str] = []
    created_at: str


class MemoryEdgeResponse(BaseModel):
    """Edge between memories based on similarity."""
    source: str
    target: str
    weight: float


class MemoryUniverseResponse(BaseModel):
    """Full memory universe for 3D visualization."""
    nodes: List[MemoryNodeResponse]
    edges: List[MemoryEdgeResponse]
    clusters: List[Dict[str, Any]]
    stats: Dict[str, Any]
