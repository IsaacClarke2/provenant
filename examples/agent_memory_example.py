"""Provenant for any agent memory (OpenClaw / Hermes / Letta / raw store).

Autonomous agents write their OWN conclusions back into memory — successes,
failures, inferred user facts — right next to what the user actually said. That
is "memory poisoning": a self-generated guess gets recalled later as if it were
ground truth. Provenant fixes the recall step.

Two pieces:
  1. `provenance_rerank` — demote self-generated memories below grounded ones
     in whatever your agent retrieves before stuffing it into the prompt.
  2. `render_context` — show the model two labelled sections so it structurally
     sees "what the user told me" vs "what I inferred".

Runs standalone:
    pip install git+https://github.com/IsaacClarke2/provenant
    python examples/agent_memory_example.py
"""
from provenant import (
    MODEL_GENERATED,
    MODEL_INFERRED,
    USER,
    provenance_rerank,
    render_context,
)

# Whatever your agent's memory recall returns — a list of items with a
# similarity score and an origin you tagged when the memory was written.
memories = [
    {"text": "User's deploy target is AWS", "origin": MODEL_INFERRED, "score": 0.83},
    {"text": "User's deploy target is Hetzner", "origin": USER, "score": 0.64},
    {"text": "User dislikes vendor lock-in", "origin": USER, "score": 0.55},
    {"text": "I successfully deployed to AWS last time", "origin": MODEL_GENERATED, "score": 0.80},
]


def recall(items, k=3):
    return provenance_rerank(
        items, score_of=lambda m: m["score"], origin_of=lambda m: m["origin"], top_k=k
    )


if __name__ == "__main__":
    print("naive recall (by score) — the agent's own guesses dominate:")
    for m in sorted(memories, key=lambda m: -m["score"])[:3]:
        print(f"  {m['score']:.2f} [{m['origin']:<16}] {m['text']}")

    top = recall(memories, k=3)
    print("\nprovenance recall — grounded facts win:")
    for m in top:
        print(f"        [{m['origin']:<16}] {m['text']}")

    print("\ncontext handed to the model:\n")
    print(render_context(top, lambda m: m["origin"], lambda m: m["text"]))
