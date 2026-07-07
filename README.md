# RG Memory

> **Part of the [DevSwat](https://resonant.dev-swat.com) platform** — Core memory engine with Hash Sphere coordinate system, RAG, vector search, and ML-powered embeddings.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

---

## What This Service Does

The Memory service is the **"what you know"** layer of the platform. Every piece of knowledge — chat messages, code context, agent observations, uploaded documents — flows through it.

| Service | Owns | Example |
|---|---|---|
| **RG_Memory** (this) | Knowledge & recall | "What do you remember?" — embeddings, vector search, Hash Sphere, RAG |
| **RG_Auth** | Identity & security | "Who are you?" — login, JWT, MFA, API keys |
| **RG_Chat** | Conversations | "What are you saying?" — chat messages, AI responses, skill routing |

---

## GitHub Repository

```
git@github-devswat:DevSwat-ResonantGenesis/RG_Memory.git
```

**Server path**: `/home/deploy/RG_Memory`
**Local path**: `/Users/louie/CascadeProjects/RG/RG_Memory`

---

## Responsibilities

### 1. Memory Ingestion & Storage
- **Ingest** memories from chat, workflows, IDE uploads, agents, and external sources
- **Hash Sphere coordinate system** — every memory gets 3D semantic coordinates (xyz), hashes, resonance scores, spin vectors, cluster assignment
- **Multi-tenant** — memories scoped by `user_id` and `org_id`
- **Agent-scoped** — memories can be tied to a specific agent via `agent_hash`
- **Credit deduction** — billing integration charges per ingest/embed/retrieve operation

### 2. Embeddings (ML-Powered)
- **MiniLM-L6-v2** (default) — 384-dim, local, free, optimized for LOCOMO benchmark (0.597 score)
- **Nomic Embed v1.5** (optional) — 512-dim Matryoshka, local, free
- **OpenAI text-embedding-3-small** (cloud fallback) — 1536-dim, paid
- **Hash-based fallback** (development) — deterministic, no ML required
- Lazy-loaded models — no startup delay unless embeddings are requested

### 3. Hash Sphere Coordinate System
Every memory is projected into a 3D semantic space with:
- **Layer 2: Hash Generation** — `meaning_hash`, `energy_hash`, `spin_hash` (semantic, emotional, directional)
- **Layer 3: Universe ID** — SHA-256 deterministic universe per user
- **Layer 5: Cartesian Coordinates** — `xyz_x`, `xyz_y`, `xyz_z` (PCA-reduced from embedding)
- **Layer 5: Hyperspherical** — `sphere_r`, `sphere_phi`, `sphere_theta`
- **Layer 6: Resonance Scoring** — `R(h) = sin(a·x) + cos(b·y) + tan(c·z)`, normalized 0-1
- **Anchor Energy** — `E_j(s) = exp(-β·||s - A_j||²)` proximity to memory anchors
- **Spin Vector** — 3D semantic rotation (spin_x, spin_y, spin_z, magnitude)
- **Semantic Components** — meaning_score, intensity_score, sentiment_score

### 4. RAG (Retrieval-Augmented Generation)
- **CRUD memories** — create, list, get, update, delete with user-scoped auth
- **Semantic search** (`/memory/rag/ask`) — vector similarity + keyword search, returns relevant memories as context
- **Conversations** — list, get, rename, delete conversation threads
- **File upload** — PDF, DOCX, TXT ingested via document loaders with chunking
- **Universe view** — get user's complete memory universe stats

### 5. Memory Anchors
- **Anchor creation** — key memory points extracted from conversations or code
- **Anchor search** — find anchors by text similarity, type, or proximity
- **Anchor types** — `chat`, `code`, `function`, `pattern`
- **Code-specific fields** — `file_path`, `function_name`, `language`, `line_range`, `code_snippet`
- **Immutability** — Hash Sphere is append-only; use `is_archived` to hide anchors
- **Deterministic Anchor Universes** — derived from BIP-39 seed hash

### 6. Memory Intelligence
- **Deduplication** — SimHash + cosine similarity to detect near-duplicate memories
- **Semantic clustering** — 6 clusters (Alpha-Living, Beta-Inanimate, Gamma-Abstract, Delta-Actions, Epsilon-Qualities, Zeta-Relations)
- **Semantic cache** — cache frequent queries to avoid re-embedding
- **Embedding cache** — LRU cache for repeated content
- **Memory encryption** — AES encryption at rest for sensitive memories
- **Temporal memory** — time-decay relevance scoring
- **Short-term memory** — volatile session-scoped memory
- **Performance tracking** — per-operation latency and throughput metrics

### 7. ML Retraining Loop
- **Autonomous retraining** — background loop retrains Hash Sphere ML models on new data
- **Semantic encoder** — classifies memories into clusters + extracts temperature/polarity
- **Sphere projection** — 512→3D neural network (triplet-loss) for coordinate projection
- **Manual trigger** — `POST /memory/hash-sphere/retrain` to force immediate retraining
- **Training data** — uses production `MemoryEmbedding` records as training source
- **Pre-trained models** — shipped in `app/data/models/` (semantic_encoder_model.pkl, tinyu_model.json)

### 8. Visualizers (HTML)
- **Semantic Space Visualizer** — 3D interactive visualization of memory coordinates
- **Memory Manager** — browse, search, manage memories via web UI
- **Hash Sphere Visualizer Pro** — advanced 3D Hash Sphere rendering (iframe-embeddable in frontend)

---

## Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                      ORG_Frontend (React)                        │
│                                                                  │
│  Memory page      ──→ GET  /memory/rag/memories                  │
│  RAG ask          ──→ POST /memory/rag/ask                       │
│  File upload      ──→ POST /memory/rag/files/upload              │
│  Conversations    ──→ GET  /memory/rag/conversations             │
│  Hash Sphere viz  ──→ GET  /memory/visualizer/hash-sphere        │
│  Anchor list      ──→ GET  /memory/hash-sphere/anchors           │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTPS (resonant.dev-swat.com)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RG_Gateway (FastAPI proxy)                     │
│                                                                  │
│  /memory/*         ──proxy──→  memory_service:8000/memory/*      │
│  /hash-sphere/*    ──proxy──→  memory_service:8000/memory/*      │
│                                                                  │
│  Gateway injects x-user-id header from JWT on every request      │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Docker internal network
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   RG_Memory (this service)                        │
│                      Port 8000                                   │
│                                                                  │
│  3 routers:                                                      │
│    routers.py (memory_router)  ── /memory/*  (20 endpoints)      │
│    routers.py (rag_router)     ── /memory/rag/*  (12 endpoints)  │
│    visualizer_routes.py        ── /memory/visualizer/* (4 endpts)│
│  + main.py direct endpoints    ── /memory/* (7 endpoints)        │
│                                                                  │
│  ML models loaded at startup:                                    │
│    ├── MiniLM-L6-v2 (sentence-transformers, 384-dim)             │
│    ├── Semantic encoder (cluster classification)                 │
│    └── Sphere projection (3D coordinate mapping)                 │
│                                                                  │
│  Outbound calls:                                                 │
│    ├── RG_Billing  (billing_service:8000)                        │
│    │   → Credit deduction on ingest/embed/retrieve               │
│    ├── RG_DSID_Blockchain (blockchain_service:8000)              │
│    │   → DSID creation, anchor blockchain proofs                 │
│    └── DigitalOcean Spaces (S3-compatible)                       │
│        → File storage for uploaded documents                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│           PostgreSQL (DigitalOcean Managed Database)              │
│           resonant-db / defaultdb                                │
│                                                                  │
│  Tables (5):                                                     │
│    memory_records     — core memories with Hash Sphere coords    │
│    memory_embeddings  — vector embeddings (ARRAY Float)          │
│    memory_chunks      — chunked long documents                   │
│    memory_anchors     — key memory points with 3D coordinates    │
│    resonance_clusters — grouped memories by resonance score      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dependencies & Connections

### Upstream (who calls this service)

| Caller | How | What it calls |
|---|---|---|
| **ORG_Frontend** | Via Gateway HTTPS proxy | RAG endpoints, anchors, visualizers, memory CRUD |
| **RG_Gateway** | HTTP proxy on Docker network | All `/memory/*` routes |
| **RG_Chat** | Direct HTTP `http://memory_service:8000` | `/memory/rag/memories`, `/memory/rag/ask`, `/memory/ingest` — chat context injection |
| **RG_Agent_Engine** | Direct HTTP `http://memory_service:8000` | Memory read/write during agent execution |
| **RG_agent_architect** | Direct HTTP `http://memory_service:8000` | Memory context for agent planning |
| **RG_AST_analysis** | Direct HTTP `http://memory_service:8000` | Code memory storage |
| **RG_OpenClaw** | Via Gateway HTTPS (federated) | Local agents writing to platform memory (opt-in) |

### Downstream (what this service calls)

| Service | URL (Docker internal) | Why |
|---|---|---|
| **PostgreSQL** | `MEMORY_DATABASE_URL` env var | All persistent data (memories, embeddings, anchors, clusters) |
| **RG_Billing** | `http://billing_service:8000` | Credit deduction on ingest/embed/retrieve |
| **RG_DSID_Blockchain** | `http://blockchain_service:8000` | DSID creation, anchor blockchain proofs |
| **DigitalOcean Spaces** | `sfo3.digitaloceanspaces.com` (S3) | File storage for uploaded documents (MinIO client) |

### No dependency on

| Service | Why |
|---|---|
| **RG_Auth** | Memory uses `x-user-id` header from Gateway — no direct auth calls |
| **RG_LLM_Service** | Embeddings are generated locally (MiniLM), not via LLM service |
| **Redis** | No cache layer — uses in-process semantic cache and embedding cache |

---

## Database Schema

### `memory_records` (core memory table)
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | Unique memory ID |
| `chat_id` | UUID | Conversation reference |
| `user_id` | UUID | Memory owner (indexed) |
| `org_id` | UUID | Multi-tenant org (indexed) |
| `agent_hash` | String(64) | Agent-scoped memory (indexed) |
| `source` | String(64) | Origin: `chat`, `workflow`, `ide_upload`, etc. |
| `content` | Text | Memory content |
| `hash` | String(255) | Unique resonance hash (indexed) |
| `meaning_hash` | String(64) | Semantic meaning hash |
| `energy_hash` | String(64) | Emotional intensity hash |
| `spin_hash` | String(64) | Direction/intent hash |
| `universe_id` | String(64) | User's Memory Universe ID (indexed) |
| `xyz_x/y/z` | Float | 3D Cartesian coordinates |
| `sphere_r/phi/theta` | Float | Hyperspherical coordinates |
| `resonance_score` | Float | `R(h) = sin(a·x) + cos(b·y) + tan(c·z)` |
| `normalized_resonance` | Float | Resonance normalized to 0-1 |
| `anchor_energy` | Float | Proximity to nearest anchor |
| `spin_x/y/z` | Float | Semantic rotation vector |
| `spin_magnitude` | Float | Rotation magnitude |
| `meaning_score` | Float | Content richness |
| `intensity_score` | Float | Emotional intensity |
| `sentiment_score` | Float | Positive/negative sentiment |
| `cluster_id` | UUID | Cluster assignment (indexed) |
| `cluster_name` | String(255) | Alpha-Living, Beta-Inanimate, etc. |
| `hash_sphere_coords` | JSON | Full coordinates dict (backward compat) |
| `extra_metadata` | JSON | Extensible metadata |
| `created_at` / `updated_at` | Timestamp | Auto |

### `memory_embeddings`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `memory_id` | UUID | References `memory_records.id` (indexed) |
| `user_id` | UUID | Embedding owner (indexed) |
| `embedding` | Float[] (ARRAY) | Vector embedding (384-dim MiniLM or 1536-dim OpenAI) |
| `model` | String(64) | Model name (`all-MiniLM-L6-v2`, `text-embedding-3-small`) |
| `dimensions` | Integer | Vector dimensions |

### `memory_chunks`
| Column | Type | Notes |
|---|---|---|
| `memory_id` | UUID | Parent memory (indexed) |
| `chunk_index` | Integer | Chunk position |
| `content` | Text | Chunk content |
| `token_count` | Integer | Token count |

### `memory_anchors`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID (PK) | |
| `user_id` | UUID | Anchor owner (indexed) |
| `org_id` | UUID | Org scope (indexed) |
| `chat_id` / `message_id` | UUID | Source reference |
| `anchor_text` | Text | Key phrase |
| `anchor_hash` | String(255) | Hash of anchor (indexed) |
| `context` | Text | Surrounding context |
| `importance_score` | Float | 0-1 importance |
| `xyz_x/y/z` | Float | 3D Hash Sphere coordinates |
| `anchor_type` | String(50) | `chat`, `code`, `function`, `pattern` (indexed) |
| `is_archived` | Boolean | Soft-delete (Hash Sphere is immutable) |
| `file_path` | String(500) | Code file path (code anchors only) |
| `function_name` | String(255) | Function name (code anchors only) |
| `language` | String(50) | Programming language |
| `code_snippet` | Text | Full code snippet |
| `universe_id` | String(32) | Deterministic universe |
| `agent_hash` | String(64) | Agent-scoped anchor |

### `resonance_clusters`
| Column | Type | Notes |
|---|---|---|
| `cluster_name` | String(255) | Cluster display name |
| `cluster_hash` | String(255) | Cluster identity hash |
| `anchor_ids` | JSON | List of anchor UUIDs in this cluster |
| `resonance_score` | Float | Overall cluster resonance |
| `personality_traits` | JSON | Personality traits extracted |

---

## API Routes (43 endpoints)

### Core Memory (`/memory/*` — memory_router)
| Method | Path | Description |
|---|---|---|
| POST | `/memory/ingest` | Ingest memory with Hash Sphere coordinates + credit deduction |
| POST | `/memory/retrieve` | Retrieve memories by similarity search |
| POST | `/memory/search` | Full-text + vector hybrid search |
| DELETE | `/memory/{memory_id}` | Delete a memory record |
| GET | `/memory/stats` | Memory statistics for current user |
| GET | `/memory/health` | Router health check |
| GET | `/memory/perf/stats` | Performance tracking stats |
| POST | `/memory/create-vector-index` | Create pgvector index for similarity search |
| GET | `/memory/encryption/status` | Memory encryption status |
| GET | `/memory/projects` | List user's code projects in memory |
| GET | `/memory/project/files` | List files in a project |

### Hash Sphere (`/memory/hash-sphere/*`)
| Method | Path | Description |
|---|---|---|
| POST | `/memory/hash-sphere/extract` | Extract Hash Sphere coordinates from content |
| POST | `/memory/hash-sphere/hash` | Generate resonance hash for content |
| POST | `/memory/hash-sphere/resonance` | Calculate resonance between two content pieces |
| POST | `/memory/hash-sphere/anchors` | Create a new memory anchor |
| GET | `/memory/hash-sphere/anchors` | List user's memory anchors (auth-filtered) |
| POST | `/memory/hash-sphere/search` | Search anchors by text similarity |
| POST | `/memory/hash-sphere/retrain` | Trigger ML model retraining |
| GET | `/memory/hash-sphere/health_stub` | Hash Sphere health check |

### Archive (`/memory/archive/*`)
| Method | Path | Description |
|---|---|---|
| POST | `/memory/archive/file` | Archive a code file's anchors |
| POST | `/memory/unarchive/file` | Unarchive a code file's anchors |
| GET | `/memory/archived/files` | List archived files |

### RAG (`/memory/rag/*` — rag_router)
| Method | Path | Description |
|---|---|---|
| POST | `/memory/rag/memories` | Create a RAG memory |
| GET | `/memory/rag/memories` | List user's RAG memories |
| GET | `/memory/rag/memories/{id}` | Get a specific memory |
| PUT | `/memory/rag/memories/{id}` | Update a memory |
| DELETE | `/memory/rag/memories/{id}` | Delete a memory |
| POST | `/memory/rag/ask` | Semantic search — returns relevant memories as context |
| GET | `/memory/rag/conversations` | List conversation threads |
| GET | `/memory/rag/conversations/{id}` | Get conversation messages |
| PUT | `/memory/rag/conversations/{id}` | Rename a conversation |
| DELETE | `/memory/rag/conversations/{id}` | Delete a conversation |
| POST | `/memory/rag/files/upload` | Upload file (PDF, DOCX, TXT) for RAG |
| GET | `/memory/rag/universe` | Get user's memory universe stats |
| GET | `/memory/rag/stats` | Database-level RAG statistics |

### Clusters (`/memory/clusters/*`)
| Method | Path | Description |
|---|---|---|
| POST | `/memory/clusters/compute` | Retroactively assign clusters to unclustered memories |
| GET | `/memory/clusters/stats` | Cluster distribution stats |

### Embeddings
| Method | Path | Description |
|---|---|---|
| POST | `/memory/embed` | Generate embeddings for text (with credit deduction) |

### Visualizers (`/memory/visualizer/*`)
| Method | Path | Description |
|---|---|---|
| GET | `/memory/visualizer/semantic-space` | 3D semantic space visualizer (HTML) |
| GET | `/memory/visualizer/memory-manager` | Memory manager UI (HTML) |
| GET | `/memory/visualizer/hash-sphere` | Hash Sphere visualizer pro (HTML, iframe-embeddable) |
| GET | `/memory/visualizer/health` | Visualizer health check |

---

## ML Models & Embeddings

### MiniLM-L6-v2 (Default — Local)
- **Dimensions**: 384
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LOCOMO benchmark**: 0.597 (vs Nomic's 0.472)
- **Deduplication threshold**: 0.85 cosine similarity
- **Loaded**: Lazy on first embedding request

### Nomic Embed v1.5 (Optional — Local)
- **Dimensions**: 512 (Matryoshka — can truncate to 256, 128, 64)
- **Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Task prefixes**: `search_document:`, `search_query:`, `clustering:`, `classification:`

### Semantic Encoder (Cluster Classification)
- **Input**: Memory content (text, max 2000 chars)
- **Output**: Cluster assignment (Alpha through Zeta) + temperature + polarity
- **Model**: Pre-trained sklearn model at `app/data/models/semantic_encoder_model.pkl`
- **Retrained**: Autonomously on new data via retraining loop

### Sphere Projection (3D Coordinates)
- **Input**: 512-dim embedding
- **Output**: 3D (xyz) coordinates via triplet-loss neural network
- **Model**: Pre-trained at `app/data/models/tinyu_model.json`
- **Retrained**: Autonomously alongside semantic encoder

---

## Security Features

- **User isolation**: All queries filtered by `x-user-id` header (injected by Gateway from JWT)
- **Memory encryption**: AES encryption at rest for sensitive memories (`memory_encryption.py`)
- **Immutable Hash Sphere**: Anchors cannot be deleted — only archived (`is_archived` flag)
- **Credit gating**: Billing integration prevents unlimited usage
- **No direct auth dependency**: Uses Gateway-injected headers, not direct JWT validation

---

## File Structure

```
RG_Memory/
├── Dockerfile                        # Python 3.11-slim, uvicorn
├── LICENSE.txt
├── README.md                         # This file
├── requirements.txt                  # 25 dependencies (ML-heavy)
├── migrations/
│   ├── 001_add_xyz_indexes.sql       # Performance indexes for Hash Sphere coords
│   └── add_hash_sphere_fields.sql    # Add all Hash Sphere columns to memory_records
├── tests/
│   └── ...                           # Test suite
└── app/
    ├── main.py                       # FastAPI app, credit deduction, router mounting
    ├── config.py                     # Settings (Pydantic), DB URL, S3, embedding config
    ├── db.py                         # Async SQLAlchemy engine + session (NullPool option)
    ├── models.py                     # 5 tables (MemoryRecord, Embedding, Chunk, Anchor, Cluster)
    ├── auth.py                       # JWT validation (x-user-id header + Bearer token)
    │
    │   # ── Routers ──
    ├── routers.py                    # memory_router + rag_router (32 endpoints, 3319 lines)
    ├── visualizer_routes.py          # HTML visualizer serving (4 endpoints)
    │
    │   # ── Embeddings ──
    ├── embeddings.py                 # Multi-provider embeddings (MiniLM, Nomic, OpenAI, hash)
    ├── embeddings_minilm.py          # MiniLM-L6-v2 wrapper (384-dim, LOCOMO-optimized)
    │
    │   # ── Services ──
    ├── services/
    │   ├── __init__.py               # Exports resonance_hasher, memory_anchor_service
    │   ├── resonance_hashing.py      # ResonanceHasher — full Hash Sphere pipeline (1.3K lines)
    │   ├── hash_sphere.py            # High-level Hash Sphere + MemoryAnchorService
    │   ├── sphere_projection.py      # 512→3D neural network projection (triplet-loss)
    │   ├── semantic_encoder.py       # Cluster classification (Alpha-Zeta)
    │   ├── trained_semantic_encoder.py  # ML-trained version of semantic encoder
    │   ├── pgvector_search.py        # PostgreSQL vector similarity search
    │   ├── vector_store.py           # Vector storage abstraction
    │   ├── semantic_cache.py         # Query result caching
    │   ├── embedding_cache.py        # LRU cache for repeated embeddings
    │   ├── memory_deduplication.py   # SimHash + cosine near-duplicate detection
    │   ├── memory_encryption.py      # AES encryption at rest
    │   ├── memory_extraction.py      # Extract structured info from raw content
    │   ├── memory_summarization.py   # Summarize long memories
    │   ├── document_loaders.py       # PDF, DOCX, HTML document parsing + chunking
    │   ├── hybrid_memory_ranker.py   # Combined vector + keyword ranking
    │   ├── dual_memory_engine.py     # Short-term + long-term memory routing
    │   ├── short_term_memory.py      # Volatile session-scoped memory
    │   ├── temporal_memory.py        # Time-decay relevance scoring
    │   ├── simhash.py                # SimHash fingerprinting for dedup
    │   ├── retraining_loop.py        # Autonomous ML retraining (background loop)
    │   └── performance_logger.py     # Per-operation latency tracking
    │
    │   # ── Static HTML Visualizers ──
    ├── static/
    │   ├── semantic_space_visualizer.html   # 3D semantic space (Three.js)
    │   ├── memory_manager.html              # Memory management UI
    │   ├── memory_visualizer_pro.html       # Hash Sphere pro visualizer (102KB)
    │   └── hash_sphere_visualizer.html      # Basic Hash Sphere visualizer
    │
    │   # ── Pre-trained ML Models ──
    └── data/
        ├── semantic_training_data.json      # Training samples for semantic encoder
        └── models/
            ├── semantic_encoder_model.pkl   # Pre-trained cluster classifier
            ├── tinyu_model.json             # Pre-trained sphere projection (2.3MB)
            └── train_semantic_model.py      # Script to retrain semantic model
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DATABASE_URL` | (constructed from PG host/port/db) | PostgreSQL connection string |
| `MEMORY_POSTGRES_HOST` | `memory_db` | PostgreSQL host |
| `MEMORY_POSTGRES_PORT` | `5432` | PostgreSQL port |
| `MEMORY_POSTGRES_USER` | `memory_user` | PostgreSQL user |
| `MEMORY_POSTGRES_PASSWORD` | `memory_pass` | PostgreSQL password |
| `MEMORY_POSTGRES_DB` | `memory_db` | PostgreSQL database name |
| `MEMORY_DB_POOL_CLASS` | `queue` | Set to `null` for NullPool (production) |
| `MEMORY_DB_POOL_SIZE` | `10` | Connection pool size |
| `ML_DATABASE_URL` | *(empty)* | Separate DB for ML retraining state |
| `MEMORY_OPENAI_API_KEY` | *(empty)* | OpenAI API key (optional, for cloud embeddings) |
| `MEMORY_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `MEMORY_USE_NOMIC_EMBED` | `true` | Use Nomic Embed (local) |
| `MEMORY_NOMIC_MATRYOSHKA_DIM` | `512` | Nomic embedding dimensions |
| `MEMORY_CHUNK_SIZE` | `500` | Document chunk size (tokens) |
| `MEMORY_CHUNK_OVERLAP` | `50` | Chunk overlap (tokens) |
| `MEMORY_LLM_SERVICE_URL` | `http://llm_service:8000` | LLM service URL |
| `BILLING_SERVICE_URL` | `http://billing_service:8000` | Billing service for credit deduction |
| `BLOCKCHAIN_SERVICE_URL` | `http://blockchain_service:8000` | Blockchain for DSID/anchor proofs |
| `MEMORY_STORAGE_ENDPOINT` | `sfo3.digitaloceanspaces.com` | S3-compatible storage endpoint |
| `DO_SPACES_ACCESS_KEY` | *(required for file upload)* | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET_KEY` | *(required for file upload)* | DigitalOcean Spaces secret key |
| `MEMORY_STORAGE_BUCKET` | `genesis2026-memory` | S3 bucket name |
| `AUTH_JWT_SECRET_KEY` | *(auto-generated)* | JWT secret for token validation |

---

## Credit Costs

| Operation | Credits | Description |
|---|---|---|
| `embed` | 100 | Generate embedding for content |
| `retrieve` | 50 | Retrieve memories by similarity |
| `store` | 20 | Ingest a new memory |
| `delete` | 5 | Delete a memory |
| `memory_write` | 2 | Write via RAG endpoint |
| `memory_read` | 0 | Read is free |
| `rag_upload` | 10 | Upload file for RAG |
| `per_mb` | 1 | Per MB stored |
| `per_gb` | 1000 | Per GB stored |

---

## Deployment

- **Container name**: `memory_service`
- **Port**: 8000
- **Server path**: `/home/deploy/RG_Memory`
- **Docker network**: `genesis2026_production_backend_app-network`
- **Database**: DigitalOcean Managed PostgreSQL (`resonant-db`)
- **Health check**: `GET /health` every 30s
- **Restart policy**: `unless-stopped`
- **ML models**: ~100MB total (downloaded on first run, cached in container)

### Docker Compose entry (in `RG_core/docker-compose.unified.yml`)
```yaml
memory_service:
  build:
    context: /home/deploy/RG_Memory
    dockerfile: Dockerfile
  container_name: memory_service
  env_file:
    - ./.env.production
  environment:
    DATABASE_URL: ${MEMORY_DATABASE_URL}
    REDIS_URL: redis://shared_redis:6379/0
  networks:
    - app-network
  restart: unless-stopped
```

---

## Quick Start (local development)

```bash
cd RG_Memory
pip install -r requirements.txt

export MEMORY_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/memory_db"
export BILLING_SERVICE_URL="http://localhost:9999"  # stub or skip billing
export AUTH_JWT_SECRET_KEY="dev-secret-change-me"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Note**: First embedding request will download MiniLM-L6-v2 (~90MB). Subsequent requests use cached model.

---

**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [resonant.dev-swat.com](https://resonant.dev-swat.com)
