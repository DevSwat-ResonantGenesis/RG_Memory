"""
Wave 2 gate: prove the prototype model beats the Wave-1 seed word-count on
12-D gravity separation. Run inside the memory_service container:
    docker exec memory_service python /app/test_hash_sphere_wave2.py
"""
import asyncio

from app.embeddings import embeddings_generator
from app.services.hash_sphere_model import hash_sphere_model
from app.services import hash_sphere_core as hsc

# (related?, text_a, text_b)
PAIRS = [
    (True,  "I am building a physics-informed hash sphere memory system",
            "the hash sphere stores memories as points in a 12 dimensional space"),
    (True,  "my physician prescribed medicine for my illness",
            "the doctor gave me treatment at the hospital"),
    (True,  "our startup raised venture funding from investors",
            "the company secured capital investment for growth"),
    (False, "I love my dog and we play in the park every morning",
            "the database server crashed with a fatal error"),
    (False, "the weather is sunny and warm today",
            "please refactor the authentication module code"),
]


async def gravity_for(a, b, use_model):
    ea = (await embeddings_generator.generate([a], task="search_document"))[0]
    eb = (await embeddings_generator.generate([b], task="search_document"))[0]
    if not use_model:
        # force Wave-1 path by passing no embedding / axes to encode_core
        ca = hsc.encode_core(a, embedding=None)
        cb = hsc.encode_core(b, embedding=None)
    else:
        axa = await hash_sphere_model.axes_for_text(a, embeddings_generator)
        axb = await hash_sphere_model.axes_for_text(b, embeddings_generator)
        ca = hsc.encode_core(a, embedding=ea, axes=axa)
        cb = hsc.encode_core(b, embedding=eb, axes=axb)
    return hsc.gravity(ca.metric_vector(), cb.metric_vector())


async def main():
    ok = await hash_sphere_model.ensure_built(embeddings_generator)
    print("model built:", ok, "ready:", hash_sphere_model.ready)
    print(f"\n{'pair':6} {'W1(wordcount)':>14} {'W2(model)':>10}  text")
    rel_w1, rel_w2, unrel_w1, unrel_w2 = [], [], [], []
    for related, a, b in PAIRS:
        g1 = await gravity_for(a, b, use_model=False)
        g2 = await gravity_for(a, b, use_model=True)
        tag = "REL " if related else "UNREL"
        print(f"{tag:6} {g1:14.3f} {g2:10.3f}  {a[:34]!r}")
        (rel_w1 if related else unrel_w1).append(g1)
        (rel_w2 if related else unrel_w2).append(g2)

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    sep1 = avg(rel_w1) - avg(unrel_w1)
    sep2 = avg(rel_w2) - avg(unrel_w2)
    print(f"\nRelated avg   : W1={avg(rel_w1):.3f}  W2={avg(rel_w2):.3f}")
    print(f"Unrelated avg : W1={avg(unrel_w1):.3f}  W2={avg(unrel_w2):.3f}")
    print(f"Separation    : W1={sep1:.3f}  W2={sep2:.3f}")
    print("VERDICT:", "✅ Wave 2 improves separation" if sep2 > sep1 else "❌ no improvement")


asyncio.run(main())
