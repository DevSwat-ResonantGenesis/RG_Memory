"""
LOCOMO benchmark harness for RG_Memory.

Measures our memory end-to-end against the LOCOMO dataset (10 long multi-session
conversations, ~300 QA across categories: 1=multi-hop, 2=temporal, 3=open-domain,
4=single-hop, 5=adversarial). This is the honest "are we #1?" measurement.

Pipeline per question:
  ingest all session turns → /memory/hash-sphere/extract (retrieve top-K) →
  LLM answers from the retrieved memories → LLM judges answer vs gold → score.

Run inside the memory_service container (has httpx + reaches memory + llm services):
  DATASET=/tmp/locomo10.json N_SAMPLES=2 LIMIT=20 python locomo_run.py

Env:
  DATASET      path to locomo10.json
  N_SAMPLES    how many conversations to run (default 2)
  LIMIT        retrieval top-K (default 20)
  SKIP_ENRICH  "1" to skip fact-extraction/anchoring at ingest (fast; ablation)
"""
import asyncio, hashlib, json, os, uuid
import httpx

MEM = "http://localhost:8000"
LLM = "http://llm_service:8000/llm/chat/completions"
DATASET = os.getenv("DATASET", "/tmp/locomo10.json")
N_SAMPLES = int(os.getenv("N_SAMPLES", "2"))
QA_LIMIT = int(os.getenv("QA_LIMIT", "0"))  # 0 = all questions per sample
LIMIT = int(os.getenv("LIMIT", "20"))
SKIP_ENRICH = os.getenv("SKIP_ENRICH", "1") == "1"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def bench_user(sample_id: str) -> str:
    h = hashlib.sha256(f"locomo-{sample_id}".encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


async def llm(client, system, user, max_tokens=256):
    try:
        r = await client.post(LLM, json={"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}], "temperature": 0, "max_tokens": max_tokens}, timeout=60)
        if r.status_code != 200:
            return ""
        return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        return ""


async def ingest_session(client, user_id, turns, date_time=None):
    for t in turns:
        text = f"{t.get('speaker','')}: {t.get('text','')}".strip()
        if len(text) < 3:
            continue
        try:
            await client.post(f"{MEM}/memory/ingest", json={
                "user_id": user_id, "org_id": user_id, "source": "locomo",
                "content": text, "generate_embedding": True,
                "skip_enrichment": SKIP_ENRICH,
                "event_timestamp": date_time}, timeout=30)  # temporal: event date
        except Exception:
            pass


async def retrieve(client, user_id, question):
    """Return [(date, content)] so the answer LLM can read WHEN each memory is from."""
    try:
        r = await client.post(f"{MEM}/memory/hash-sphere/extract", json={
            "user_id": user_id, "org_id": user_id, "query": question,
            "limit": LIMIT, "min_score": 0.0}, timeout=60)
        if r.status_code != 200:
            return []
        out = []
        for m in r.json().get("memories", []):
            ts = (m.get("timestamp") or "")[:10]  # YYYY-MM-DD
            out.append((ts, m["content"]))
        return out
    except Exception:
        return []


async def run():
    data = json.load(open(DATASET))
    samples = data[:N_SAMPLES]
    async with httpx.AsyncClient() as client:
        totals, correct = {}, {}
        n_total = n_correct = 0
        for si, sample in enumerate(samples):
            sid = sample.get("sample_id", str(si))
            user_id = bench_user(sid)
            conv = sample["conversation"]
            # ingest every session's turns
            sess_keys = sorted([k for k in conv if k.startswith("session_") and isinstance(conv[k], list)],
                               key=lambda k: int(k.split("_")[1]))
            for k in sess_keys:
                await ingest_session(client, user_id, conv[k], date_time=conv.get(f"{k}_date_time"))
            await asyncio.sleep(1)
            print(f"[sample {si+1}/{len(samples)}] {sid}: ingested {sum(len(conv[k]) for k in sess_keys)} turns, {len(sample.get('qa',[]))} questions", flush=True)

            qa_list = sample.get("qa", [])
            if QA_LIMIT:
                qa_list = qa_list[:QA_LIMIT]
            for qa in qa_list:
                q, gold = qa.get("question", ""), str(qa.get("answer", qa.get("adversarial_answer", "")))
                cat = qa.get("category", 0)
                mems = await retrieve(client, user_id, q)
                ctx = "\n".join(f"- [{ts}] {c}" for ts, c in mems[:LIMIT]) or "(no memories)"
                ans = await llm(client,
                    "Answer the question using ONLY the dated memories. Each memory is prefixed with its date [YYYY-MM-DD]. For 'when' questions, answer with the date. Be concise. If the memories don't contain the answer, say 'I don't know'.",
                    f"Memories:\n{ctx}\n\nQuestion: {q}\nAnswer:")
                verdict = await llm(client,
                    "You are a strict grader. Reply with exactly CORRECT or WRONG.",
                    f"Question: {q}\nGold answer: {gold}\nPredicted answer: {ans}\n\nIs the predicted answer correct (same meaning as gold)? Reply CORRECT or WRONG.",
                    max_tokens=5)
                ok = "CORRECT" in verdict.upper()
                totals[cat] = totals.get(cat, 0) + 1
                correct[cat] = correct.get(cat, 0) + (1 if ok else 0)
                n_total += 1; n_correct += ok

        print("\n===== LOCOMO RESULTS =====")
        for c in sorted(totals):
            print(f"  {CAT.get(c,c):12s}  {correct[c]}/{totals[c]} = {correct[c]/totals[c]:.3f}")
        print(f"  {'OVERALL':12s}  {n_correct}/{n_total} = {(n_correct/n_total if n_total else 0):.3f}")
        print(f"  (SOTA refs: Mem0 LOCOMO ~0.66; frontier LongMemEval ~0.95)")

asyncio.run(run())
