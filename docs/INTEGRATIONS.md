# Integrations

Provenant is a neutral adapter — it works *with* your memory store, not instead
of it. Two steps:

1. **Tag origin at write time.** When you store a memory, record where it came
   from — using the *write-site* you already know (a user turn, a tool result,
   the agent's own reasoning). This is structural metadata, not text analysis.
2. **Rerank at read time.** Multiply your retriever's score by the provenance
   weight, so model-generated guesses rank below grounded facts. One line.

| Stack | Example | One-liner |
|-------|---------|-----------|
| mem0 | [`examples/mem0_example.py`](../examples/mem0_example.py) | `rerank_mem0_results(client.search(q, user_id=uid))` |
| LangChain | [`examples/langchain_example.py`](../examples/langchain_example.py) | `rerank_lc_documents(vs.similarity_search_with_score(q, k=20))` |
| Any agent (OpenClaw / Hermes / Letta / raw) | [`examples/agent_memory_example.py`](../examples/agent_memory_example.py) | `provenance_rerank(items, score_of=..., origin_of=...)` |

## Origins

Tag with the generic origin that matches the write-site:

| origin | when |
|--------|------|
| `user` | the user stated it |
| `shared` | both sides confirmed it |
| `retrieved` | from a tool / document / search |
| `model_inferred` | the model concluded it from evidence |
| `model_generated` | the model produced it with weak grounding |
| `self_generated` | the agent produced it about itself / unsourced |

Unknown or missing origin falls back conservatively (treated as weak model
output) — never as the user.

## Where it helps — and where it doesn't

**Helps** wherever memory mixes *user-said* and *model-generated* content and
there is a retrieval/ranking step: agents with long-term memory (mem0, Letta),
self-curating agents (OpenClaw, Hermes — they write their own
successes/failures/inferences back into memory), companions, support assistants,
multi-agent trust, conversational RAG.

**Doesn't help** for pure document RAG where every chunk has the *same* trusted
origin — there is nothing to demote. Provenant earns its keep when origins are
mixed.

Even without a top-k ranking step, `render_context()` is still useful: it splits
the prompt into "[what I know — from you]" vs "[what I think — inferred]" so the
model sees the epistemic boundary.
