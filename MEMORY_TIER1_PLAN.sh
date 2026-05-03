#!/bin/bash
# =============================================================================
# RG_Memory Tier 1 Upgrade Plan
# Date: 2026-05-02
# Status: TIER 1 COMPLETE — ready for production deploy + migration
# =============================================================================

# ── COMPLETED ──────────────────────────────────────────────────────────────────

# STEP 1: DEAD CODE CLEANUP ✅
# Deleted 9 dead service files + 1 dead training script (~3,700 lines removed):
#   - services/dual_memory_engine.py      (186 lines, never imported)
#   - services/short_term_memory.py       (77 lines, never imported)
#   - services/memory_extraction.py       (605 lines, imports non-existent module)
#   - services/vector_store.py            (413 lines, never imported)
#   - services/trained_semantic_encoder.py (290 lines, only in try/except fallbacks)
#   - services/sphere_projection.py       (490 lines, only in try/except fallbacks)
#   - services/simhash.py                 (240 lines, only in try/except fallbacks)
#   - services/retraining_loop.py         (575 lines, retrained dead models)
#   - data/models/train_semantic_model.py (dead training script)
# Cleaned dead imports from main.py (retrain endpoint + startup loop)
# Cleaned dead try/except imports from resonance_hashing.py (simhash, trained_encoder, sphere_projection)
# Service files: 22 → 14. Lines: ~8,200 → ~4,569.

# STEP 2: BM25 FULL-TEXT SEARCH ✅
# - Added TSVECTOR column `search_tsv` to MemoryRecord model (models.py)
# - Added GIN index on search_tsv
# - Created search_bm25() method on PgVectorSearch (pgvector_search.py)
# - Wired as METHOD 5 in hash_sphere_routes.py extract endpoint
# - Updated hybrid_memory_ranker.py already uses bm25_score in RRF
# - Replaced fake word-overlap BM25 with real PostgreSQL ts_rank_cd scoring
# - Created migration: migrations/add_bm25_fulltext.sql

# STEP 3: DEDUPLICATION ✅
# - Added `content_hash` column (SHA-256 of normalized content) to MemoryRecord
# - Added DB-level exact dedup check in routers.py ingest_memory() before insert
# - Returns {"status": "duplicate"} if same content_hash exists for same user
# - Stores content_hash on every new record for future checks
# - Migration included in migrations/add_bm25_fulltext.sql

# STEP 4: TEMPORAL MEMORY FIX ✅
# - Rewrote temporal_memory.py (broken import → working async DB queries)
# - Uses SQLAlchemy async session + MemoryRecord model
# - Direct created_at time-range filtering (no rag_engine dependency)
# - Wired as METHOD 6 in hash_sphere_routes.py extract endpoint
# - Handles: yesterday, last week, last month, earlier today, recently, etc.

# ── TO DEPLOY ──────────────────────────────────────────────────────────────────

# 1. Run migration on production DB:
#    psql $DATABASE_URL -f migrations/add_bm25_fulltext.sql
#
# 2. Rebuild memory_service container:
#    cd /home/deploy && sudo docker compose -f docker-compose.unified.yml build memory_service
#    sudo docker compose -f docker-compose.unified.yml up -d memory_service

# ── TIER 2 (NEXT SESSION) ─────────────────────────────────────────────────────

# STEP 5: LLM fact extraction at ingest (4 hours)
#   - GPT-4o-mini extracts atomic facts from chat messages
#   - Store in memory_facts table (fact, entity, date, confidence)
#   - Search facts at retrieval time alongside memories

# STEP 6: Contradiction detection (3 hours)
#   - At ingest, compare new fact vs existing facts
#   - Mark old conflicting facts as superseded

# STEP 7: Memory compression (3 hours)
#   - Wire memory_summarization.py as periodic job
#   - Compress memories >7 days old into summaries

# ── TIER 3 (FUTURE) ───────────────────────────────────────────────────────────

# STEP 8: Entity resolution (6 hours)
# STEP 9: Cross-encoder reranking (3 hours)
# STEP 10: Synthesis / reflect (4 hours)

# ── FILE INVENTORY (14 service files) ──────────────────────────────────────────
# LIVE:
#   services/pgvector_search.py        ← core semantic search + BM25
#   services/hybrid_memory_ranker.py   ← RRF rank fusion
#   services/hash_sphere.py            ← ResonanceHasher wrapper
#   services/resonance_hashing.py      ← SHA-256 + trig coords (visualization)
#   services/memory_encryption.py      ← AES encrypt/decrypt
#   services/embedding_cache.py        ← in-memory embedding cache
#   services/semantic_cache.py         ← query result cache
#   services/performance_logger.py     ← timing stats
#   services/document_loaders.py       ← PDF/DOCX parsing
#   services/memory_deduplication.py   ← exact content hash dedup (NOW WIRED)
#   services/temporal_memory.py        ← time-based queries (NOW WORKING)
#   services/semantic_encoder.py       ← cluster assignment (low use)
# KEPT FOR TIER 2:
#   services/memory_summarization.py   ← LLM summarization (will wire later)
