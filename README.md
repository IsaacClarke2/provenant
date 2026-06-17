# Provenant

**An epistemic memory layer for LLM apps.** Tag every memory with *where it came
from*, fold a trust weight into your existing retrieval ranking, and present
grounded facts separately from inferred ones — so your model stops confidently
repeating things the user never said.

Provenant is a **neutral adapter**: it works *with* your memory store (mem0,
LangChain, Letta, a raw vector DB), never as a replacement. The core is pure
Python stdlib, zero runtime dependencies.

```python
from provenant import provenance_rerank

reranked = provenance_rerank(
    results,                                   # whatever your retriever returned
    score_of=lambda r: r["score"],            # its similarity score
    origin_of=lambda r: r["origin"],          # "user" | "retrieved" | "model_generated" | ...
)
```

That's the whole integration. A model-generated guess that happens to be
textually similar to the query now ranks **below** a fact the user actually
stated.

---

## The problem

Agent and RAG memory stores mix two very different things into one ranked list:

- facts the **user actually told you** ("allergic to penicillin"), and
- the model's own **guesses and intermediate conclusions** ("probably allergic
  to peanuts").

A plain vector search ranks by similarity, so a fluent, on-topic *confabulation*
routinely outranks the real fact, gets pulled into context, and is repeated as
truth — and the more often it's recalled, the more confident it sounds. Existing
stores record provenance as **audit metadata**; almost none of them *use* it to
**demote inferred facts at recall**. That's the gap Provenant fills.

## How it works

Three small, pure pieces:

1. **Origins** (`provenant.origins`) — a structural classifier. You declare, by
   *write-site* (a user turn, a tool result, a reasoning step), what origin a
   fact has. It's a data table, never text parsing — no regex, no keyword
   dictionaries.
2. **Weight** (`compute_provenance_weight`) — a continuous trust multiplier in
   the open interval `(0.10, 1.30)`: an origin prior, raised by independent
   grounding anchors (diminishing returns) and lowered by inference depth. Never
   an absolute 0 or 1, fully deterministic, every term auditable via `explain()`.
3. **Rerank & partition** — multiply your retrieval score by that weight
   (`provenance_rerank`), and render the surviving context in two labelled
   sections (`render_context`): *what I know (from you)* vs *what I think
   (inferred)*, so the model sees the boundary structurally.

## The benchmark *is* the launch

The claim is only worth as much as the number behind it.

```
$ python -m provenant.bench
Provenant — confabulation benchmark (offline, deterministic)
  scenarios: 6   top_k: 3
  confabulation rate  WITHOUT provenance : 100%
  confabulation rate  WITH provenance    :  17%
  absolute reduction                      :  83%
```

### Real data: LongMemEval retrieval

`provenant/bench/longmemeval.py` runs the real thing — the official LongMemEval
haystack (~493 turns/question), a real embedder (fastembed `bge-small`), origin
tagged by conversational role (user turns grounded, assistant turns
model-generated), no LLM and no faked numbers. Reranking OFF vs ON, on the first
15 questions:

| metric | base | + provenance | Δ |
|---|---|---|---|
| gold recall@5 | 0.600 | **0.800** | **+0.200** |
| gold MRR | 0.386 | 0.456 | +0.070 |
| assistant turns in top-5 | 0.387 | **0.000** | **−0.387** |

Provenance lifts the user-grounded answer into the top-5 (+20 pp) and evicts
model-generated turns from the context entirely. Reproduce:

```bash
pip install "provenant[bench]"
# download longmemeval_s from HF dataset xiaowu0162/longmemeval
python -m provenant.bench.longmemeval --data longmemeval_s --limit 15 --threads 2
```

**Honesty boundary:** the offline `python -m provenant.bench` number above is a
*synthetic proof of mechanism*; the LongMemEval number is a **15-question
sample** (CPU, single box) — directional, not the full 500-question run. Origin
here is role-based, a fair proxy because LongMemEval answers are user-stated (gold
evidence is ~94% user-origin); the ~6% of gold that lives in assistant turns is
exactly where role-based demotion can hurt, and that honest tradeoff is already
inside these numbers. Raw result: `benchmarks/results/longmemeval_n15.json`.

## Install & run

```bash
pip install -e .            # or: pip install provenant  (once published)
python -m provenant.bench   # the before/after table
python examples/quickstart.py
python -m pytest -q         # 27 tests
```

## Adapters

```python
from provenant.adapters import rerank_mem0_results, rerank_lc_documents

# mem0
results = rerank_mem0_results(client.search(query, user_id=uid))

# LangChain
pairs = vectorstore.similarity_search_with_score(query, k=20)
docs  = rerank_lc_documents(pairs)
```

Both are duck-typed — Provenant imports nothing from mem0 or LangChain.

## Design laws

Carried over from the deterministic kernel this was extracted from:

- **No hardcoded behavior.** Origin classification is structural (by write-site),
  never by parsing meaning. No regex, no keyword lists.
- **Continuous, asymptotic.** Weights move smoothly and live strictly inside an
  open interval — never an absolute 0 or 1.
- **Traceable.** No `except: return default` that hides errors; every weight and
  ranking decision exposes its terms.

## What this is *not*

- Not a memory store. It ranks and presents what your store already holds.
- Not a hallucination *detector*. It doesn't read fact content to judge truth; it
  trusts **provenance** — where a fact came from — which is cheap, deterministic,
  and orthogonal to the model.
- The trust math is a **heuristic prior**, not ground truth. Its value is shown
  by the confabulation-rate delta, not asserted.

## License

MIT.
