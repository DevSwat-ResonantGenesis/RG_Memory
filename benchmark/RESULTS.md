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

## Run 3 — 2026-07-02 (temporal fix working) — VALIDATED
Same config (1 conv, 40 Q, enrichment off). event_timestamp now sets created_at
(fixed two bugs: SQLAlchemy server_default ignored ORM set → direct UPDATE; and
the main.py ingest wrapper dropped the field → pass-through). Dates surfaced to
the answer LLM as [YYYY-MM-DD].

| Category | Run 1 | Run 2 (dates broken) | Run 3 (fixed) |
|----------|-------|----------------------|---------------|
| temporal | 0.05 | 0.00 | **0.250** (5×) |
| multi-hop | 0.40 | 0.33 | **0.467** |
| open-domain | 0.80 | 0.60 | **0.800** |
| **OVERALL** | 0.275 | 0.20 | **0.400** (+45%) |

Temporal handling lifted overall 0.275 → 0.400, mostly via temporal 0.05→0.25.
The benchmark caught the mid-fix regression (dates broken) before it shipped.

### Next levers (measured, in order)
1. **Temporal still 0.25** — retrieval isn't always surfacing the RIGHT dated
   memory, and single-date surfacing ≠ date-range reasoning. Add temporal query
   handling (before/after/during, date-range filter) → push temporal higher.
2. **Enrichment ON** — facts + knowledge-graph were OFF; turn on (fix the
   fact-extraction speed at bulk) → measure multi-hop/single-hop lift.
3. **Scale** — all 10 conversations + full question set for a stable number.
4. Stand up LongMemEval.
