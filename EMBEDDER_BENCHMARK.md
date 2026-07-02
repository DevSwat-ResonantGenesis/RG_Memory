# Base-Embedder Benchmark — 2026-07-02

**Decision: KEEP `all-MiniLM-L6-v2` (384-dim). Do NOT upgrade to a larger embedder.**

We tested whether a stronger base embedder would improve memory retrieval.
Measured on realistic short first-person memories (the actual data shape), 12
queries over 12 memories, top-1 accuracy + MRR:

| Model | dim | top-1 | MRR |
|-------|-----|-------|-----|
| **all-MiniLM-L6-v2 (current)** | 384 | **11/12** | **0.958** |
| BAAI/bge-large-en-v1.5 | 1024 | 9/12 | 0.833 |
| intfloat/e5-large-v2 | 1024 | 9/12 | 0.825 |
| BAAI/bge-base-en-v1.5 | 768 | 4/5 (small run) | 0.850 vs MiniLM 0.900 |

**MiniLM wins decisively.** Why: it's fine-tuned on ~1B short sentence-similarity
pairs — exactly our task (short conversational memories). bge/e5-large are tuned
for long passage retrieval (MS MARCO) and fit short first-person text worse; they
also add 3× CPU latency and a disruptive re-embed/schema migration.

Upgrading would be a REGRESSION, not a win. The real relevance gains came from the
cross-encoder reranker + fact injection, not the base embedder.

`services/embeddings_st.py` + `migrations/upgrade_embedding_768.sql` remain as a
READY, benchmark-gated path if a future (e.g. domain-fine-tuned) model ever beats
MiniLM on this test — but they are intentionally NOT wired in. Re-run this A/B
before ever activating them.
