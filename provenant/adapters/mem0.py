"""mem0 adapter (duck-typed, no mem0 import).

mem0's `search()` returns a list of dicts shaped roughly like::

    {"memory": "...", "score": 0.82, "metadata": {"origin": "model_inferred"}}

Tag each memory with an `origin` (in its metadata) when you `add()` it — using
the write-site you know at call time — then rerank search results in one line::

    from provenant.adapters import rerank_mem0_results
    results = rerank_mem0_results(client.search(query, user_id=uid))
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..origins import FALLBACK_ORIGIN
from ..rerank import provenance_rerank


def _origin_of(row: Dict[str, Any], origin_key: str) -> str:
    meta = row.get("metadata") or {}
    return meta.get(origin_key) or row.get(origin_key) or FALLBACK_ORIGIN


def rerank_mem0_results(
    results: Sequence[Dict[str, Any]],
    score_key: str = "score",
    origin_key: str = "origin",
    trust=None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank mem0 search results by provenance. Missing origin -> conservative.

    Missing score is treated as a neutral 0.0 base (the item then ranks purely on
    its provenance weight) rather than crashing — but the reason is explicit, not
    a silently swallowed error.
    """
    return provenance_rerank(
        list(results),
        score_of=lambda r: r.get(score_key, 0.0),
        origin_of=lambda r: _origin_of(r, origin_key),
        trust=trust,
        top_k=top_k,
    )
