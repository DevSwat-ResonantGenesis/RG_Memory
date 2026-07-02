# Resonant Memory — API & SDK

The world's first **physics-informed, immutable, sovereign** AI memory, as an API.

Unlike vector-DB memory, Resonant Memory retrieves through a **12-D hash-sphere
semantic manifold** (gravity ranking) + **emergent anchors** + a **self-organizing
associative mesh** + a **cross-encoder reranker** + **Mem0-style fact injection** +
a **multi-hop knowledge graph**, with pgvector cosine + BM25 as the recall floor.
Every memory is **immutable, encrypted, and anchored on-chain**, isolated per
user / agent / org.

## Install
```bash
pip install resonant-memory
```

## Quickstart
```python
from resonant_memory import ResonantMemory

mem = ResonantMemory(api_key="rg_live_...", user_id="user-123")

mem.ingest("My name is Marcus and I lead the payments team",
           event_timestamp="2026-05-08")

hits = mem.recall("what does the user do")          # → [{content, confidence, ...}]
full = mem.recall_full("when did Marcus join payments")
print(full["confidence"], full["answer_from_memory"], full["evidence_hash"])

for f in mem.facts():                               # distilled atomic facts
    print(f["entity"], f["attribute"], f["value"])
```

## Isolation (blockchain-block model)
Memories are cryptographically isolated. Pass identifiers to target a block:

| Scope | Pass | Block |
|-------|------|-------|
| User private | `user_id` | user block |
| Agent global | `agent_hash` | agent block |
| User + agent | `user_id` + `agent_hash` | user+agent block |

A query only ever sees memories in the caller's own blocks. Every write is
anchored on-chain (hashes only; content stays encrypted off-chain).

## Endpoints (via the gateway — auth + isolation + metering)
| Method | Path | SDK | Credits |
|--------|------|-----|---------|
| POST | `/memory/ingest` | `ingest()` | 120 |
| POST | `/memory/hash-sphere/extract` | `recall()` / `recall_full()` | 60 |
| GET  | `/memory/facts` | `facts()` | 20 |
| POST | `/memory/retrieve` | (vector-only) | 60 |

`recall_full()` returns:
- `memories` — ranked results (content, scores, timestamp)
- `confidence` + `answer_from_memory` — the **no-LLM-recall** signal
- `evidence_hash` — on-chain provenance for the recall
- `extraction_methods_used` — which layers fired (gravity, mesh, facts, knowledge_graph, cross_encoder, …)

## Billing (credits, per call)
Each call **deducts credits** from your org balance at the rates above. Buy or
top-up credits in the dashboard (min $5). When the balance is exhausted, calls
return **HTTP 402** → the SDK raises `InsufficientCreditsError`; top up to continue.

```python
from resonant_memory import InsufficientCreditsError
try:
    mem.recall("...")
except InsufficientCreditsError:
    ...  # prompt the user to top up
```

## Auth
Create an org API key in the dashboard (`Settings → API Keys`). Pass it as
`api_key`; the SDK sends `Authorization: Bearer <key>`. Keys carry scopes and are
revocable.
