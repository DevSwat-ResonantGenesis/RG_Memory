# RFC-0002 — Hash Sphere v-true (reconciled)

Status: Implementing. Single source of truth for the memory brain.
Supersedes the "cocktail" (XYZ-collapsed retrieval, RAG-as-primary).

## Principles (locked)
1. **Hash sphere is the primary memory/retrieval. RAG (pgvector cosine) + BM25 are the FLOOR/fallback** — they generate candidate recall; the 12-D semantic core does the primary ranking ("the magic").
2. **A memory is a point in a fixed 12-D semantic manifold. Its hash is its quantized position.** Text ≡ 12-D coordinate ≡ hash.
3. **XYZ is visualization only.** Never used in retrieval or storage math.
4. **Memories are immutable** going forward (append-only, archive-not-delete, on-chain in Wave 4). Legacy malformed data was cleared once at rebuild — nothing after that is deleted.
5. **Fixed 12 dimensions** (no sprouting axes). All plasticity lives at the anchor/cluster/edge level.
6. **Stable model + live field**: the vocab→12-D model is trained once and frozen; the per-user physics field (anchors, gravity, drift) updates live and cheap (arithmetic, not backprop).

## The 12-D semantic core
| # | Axis | Meaning | Source (Wave 1 seed) |
|---|------|---------|----------------------|
| 1–6 | α β γ δ ε ζ | probability distribution over World Vocabulary: Living, Inanimate, Abstract, Action, Quality, Relation | CLUSTER_WORDS counts, normalized |
| 7 | temperature | urgency / linguistic intensity (0–1) | warm/cold words |
| 8 | polarity | sentiment valence (0–1) | positive/negative words |
| 9–11 | spin (3-D) | **intensity / complexity / abstraction** of the thought (per Spin↔Energy swap) | punctuation+caps, lexical diversity, γ+ε vs α+β |
| 12 | resonance R(h) | derived harmonic signature — **NOT a distance axis** | bounded trig over dims 1–11 |

- **Metric core = 11 dims** (1–11). Distance/gravity run on these. Resonance (12) is a derived scalar used as a corroborating match score and for the hash; it is never an L2 coordinate (its `tan` term is unbounded → replaced with a bounded harmonic).
- **Energy** (`anchor_energy`) = ± resonance (signed sentiment strength), per the swap — distinct from spin.
- **hash** = quantized 11-D core + resonance bucket → deterministic id; similar text → nearby hash.

## Retrieval (hash sphere primary, RAG floor)
```
query → 12-D core
candidates ← pgvector cosine top-N  ∪  BM25 top-N        (FLOOR: recall)
rank ← gravity(query_core, mem_core) = exp(-β·||q-m||²_11d)  (PRIMARY: the brain)
fuse ← RRF(gravity_rank, cosine_rank, bm25_rank)          (gravity weighted highest)
confidence gate: top gravity high → answer from memory (no LLM); low → provider
```

## Waves
- **0 — recall floor**: DONE (vector(384), decrypt, BM25 plaintext, loader key, min_score).
- **1 — 12-D core (THIS)**: compute+store 12-D core for every new memory; hash = quantized core; retrieval re-ranks candidates by 12-D gravity; XYZ demoted to viz. Seed dictionaries first (works untrained). Clean-slate legacy data.
- **2 — vocab→12-axis model: DONE 2026-07-01** (hash_sphere_model.py). Prototype
  model in frozen MiniLM space: each α…ζ axis = centroid of its CLUSTER_WORDS
  embeddings; per-word SOFT (standardized-softmax) cluster assignment, cached,
  aggregated over the sentence; temperature/polarity = warm/cold & pos/neg
  centroid contrast. Generalizes seeds to the whole vocab (physician≈doctor).
  Gate (test_hash_sphere_wave2.py): gravity separation 0.278 vs Wave-1 0.125
  (2.2×), no false-positive blowup; verified live (doctor→physician gravity 0.63).
  Artifact persisted to data/models/hash_sphere_prototypes.json. A trained
  projection head could sharpen further later, but the prototype model is the
  stable, no-training-instability version.
- **3 — runtime physics + organ**: live gravity/drift in 12-D, self-organizing mesh, reinforcement, crystallization, knowledge graph.
- **4 — blockchain universe**: immutable per-relationship chains + evidence ledger; wire the memory_anchor tx that currently fires 0×.
- **5 — cognitive loop** (optional, LLM-gated).
