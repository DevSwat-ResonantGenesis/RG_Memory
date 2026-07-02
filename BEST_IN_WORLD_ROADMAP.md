# Best-in-World Memory — Roadmap & Honest Status (2026-07-02)

Goal: the best memory system in the world — measurably #1 AND with capabilities
no one else has. The bar is **LongMemEval** (OMEGA 95.4%, Hindsight 91.4%,
Mem0 ~49-66%).

## Two things "best in the world" requires
1. **Measurably #1** on a public benchmark (LongMemEval / LOCOMO). You cannot
   claim best without a number vs SOTA.
2. **Capabilities no one else has** (the moat).

## Capabilities — built vs remaining (vs the audit's 7 gaps + frontier)
| Capability | Who has it | Us |
|---|---|---|
| Vector recall | everyone | ✅ MiniLM-384 (benchmarked best for our short text) |
| BM25 keyword | leaders | ✅ |
| Cross-encoder rerank | Hindsight/OMEGA | ✅ (precision layer) |
| Fact extraction | Mem0/Zep | ✅ + injected into retrieval |
| Contradiction/supersede | Mem0/Zep | ✅ |
| **Multi-hop entity graph** | OMEGA/Hindsight | ✅ NEW (hash_sphere_graph.py) |
| **Temporal reasoning** (validity windows) | Zep/Graphiti | ❌ **GAP** |
| Cross-memory synthesis | OMEGA | ⚠️ partial (mesh + graph) |
| **Physics-informed 12-D manifold** | NOBODY | ✅ unique |
| **Self-organizing associative mesh** | NOBODY | ✅ unique |
| **Emergent gravity anchors** | NOBODY | ✅ unique |
| **Zero-LLM confident recall** | NOBODY | ✅ unique (confidence gate) |
| **Immutable, on-chain, sovereign** | NOBODY | ✅ unique |
| **Cryptographic evidence ledger** | NOBODY | ✅ unique |

We already have the entire **unique moat** + most retrieval capabilities. Two
things stand between us and provably-#1:

## The two remaining moves
### A. MEASURE — build a LongMemEval harness (THE critical next step)
- Restore/build a runnable benchmark (dataset + retrieval eval + LLM judge with
  the Anthropic key). Old logs exist in locomo_benchmark/ but no runnable script.
- Get our real number vs 95.4%. Every change after is measured. Without this,
  "best in the world" is unfalsifiable.

### B. TEMPORAL reasoning (GAP 5) — the last capability gap
- Facts get valid_from / valid_until (we have supersede; add validity windows).
- "What did I think about X before vs now", "as of last month" → temporal filter
  + recency-aware scoring. Wire the (currently dead) temporal patterns.

## Then: iterate to beat SOTA
With A (measurement) + B (temporal), plus facts + graph + reranker + the unique
physics/immutable moat, drive the LongMemEval number past 95%. Each capability is
already built; the campaign is measure → find failures → close → repeat.

## Deferred (real but not blocking)
- Wave 2.5 trained projection head (make 12-D gravity lead) — R&D; cosine+reranker
  already lead well.
- Wave 3d crystallization + periodic drift job.
- Chat consuming answer_from_memory (no-LLM recall in the product).
