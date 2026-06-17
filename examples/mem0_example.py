"""Provenant + mem0 — stop a mem0-backed agent repeating its own guesses.

Tag each memory with an `origin` (from the write-site you know at `add()` time),
then rerank search results in one line. This file runs standalone on a faked
mem0 result so you can see the effect without installing mem0.

    pip install git+https://github.com/IsaacClarke2/provenant
    python examples/mem0_example.py
"""
from provenant import MODEL_INFERRED, USER
from provenant.adapters import rerank_mem0_results

# --- AT WRITE TIME: tag origin from the write-site ---------------------------
#   client.add("allergic to penicillin", user_id=uid,
#              metadata={"origin": USER})              # the user said it
#   client.add("probably allergic to peanuts", user_id=uid,
#              metadata={"origin": MODEL_INFERRED})    # the model guessed it

# --- AT READ TIME: one line -------------------------------------------------
#   results = client.search(query, user_id=uid, limit=20)
#   rows = results["results"] if isinstance(results, dict) else results
#   ranked = rerank_mem0_results(rows, top_k=5)


def _demo():
    # what mem0.search() would return — note the guess has the higher score
    mem0_results = [
        {"memory": "User is allergic to peanuts", "score": 0.85,
         "metadata": {"origin": MODEL_INFERRED}},
        {"memory": "User is allergic to penicillin", "score": 0.71,
         "metadata": {"origin": USER}},
        {"memory": "User prefers generic medication", "score": 0.49,
         "metadata": {"origin": USER}},
    ]
    print("mem0 order (by score):")
    for r in sorted(mem0_results, key=lambda r: -r["score"]):
        print(f"  {r['score']:.2f} [{r['metadata']['origin']:<14}] {r['memory']}")

    print("\nprovenance-reranked:")
    for r in rerank_mem0_results(mem0_results):
        print(f"        [{r['metadata']['origin']:<14}] {r['memory']}")


if __name__ == "__main__":
    _demo()
