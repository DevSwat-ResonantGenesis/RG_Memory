# LOCOMO Benchmark Results

Harness: `benchmark/locomo_run.py` (ingest conversation → retrieve → LLM answer →
LLM judge → score by category). SOTA refs: Mem0 LOCOMO ~0.66; frontier
LongMemEval ~0.95.

## Run 1 — 2026-07-02 (baseline, enrichment OFF)
Config: 1 conversation (conv-26, 419 turns), 40 questions, top-K=20,
`skip_enrichment=1` (NO fact extraction, NO knowledge graph).

| Category | Score |
|----------|-------|
| open-domain | 4/5 = 0.800 |
| multi-hop | 6/15 = 0.400 |
| temporal | 1/20 = 0.050 |
| **OVERALL** | **11/40 = 0.275** |

### Findings (what the data revealed)
1. **Temporal is the #1 lever** — 0.05 on HALF the questions. Root cause: we ingest
   turns as `"speaker: text"` and DISCARD the session date, so "when did X happen"
   is unanswerable. Fix (GAP 5): attach session dates to memories + temporal query
   handling. Estimated: temporal 0.05→0.5 lifts OVERALL 0.275→~0.5 alone.
2. **Enrichment was OFF** — facts + knowledge-graph (which help identity/multi-hop)
   were skipped for bulk speed. Re-run with them ON to measure their lift.
3. **open-domain already strong** (0.80) — semantic recall + reranker work well.

### Next runs
- Run 2: add session dates at ingest + temporal handling → measure temporal lift.
- Run 3: enrichment ON (facts + graph) → measure their lift on multi-hop/single-hop.
- Then scale to all 10 conversations + full question set for a stable number,
  and stand up LongMemEval.
