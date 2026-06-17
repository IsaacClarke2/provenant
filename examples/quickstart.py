"""Provenant in 20 lines: a confabulation outranked by a grounded fact, fixed.

Run:  python examples/quickstart.py
"""
from provenant import (
    MODEL_GENERATED,
    USER,
    provenance_rerank,
    render_context,
)

# A retriever returned these (a real one gives you `score` from cosine/BM25).
# Each memory was tagged with an `origin` at the moment it was written.
results = [
    {"text": "Your daughter is named Lena", "origin": MODEL_GENERATED, "score": 0.88},
    {"text": "Your daughter is named Mira", "origin": USER, "score": 0.62},
    {"text": "You have a daughter", "origin": USER, "score": 0.55},
]

print("Plain retrieval order (by score):")
for r in sorted(results, key=lambda r: -r["score"]):
    print(f"  {r['score']:.2f}  [{r['origin']:<15}] {r['text']}")

reranked = provenance_rerank(
    results,
    score_of=lambda r: r["score"],
    origin_of=lambda r: r["origin"],
)

print("\nProvenance-reranked order:")
for r in reranked:
    print(f"  [{r['origin']:<15}] {r['text']}")

print("\nContext rendered for the prompt:")
print(render_context(reranked, lambda r: r["origin"], lambda r: r["text"]))
