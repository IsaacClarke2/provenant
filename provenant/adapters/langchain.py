"""LangChain adapter (duck-typed, no langchain import).

`vectorstore.similarity_search_with_score(query)` yields `(Document, score)`
pairs, where each `Document` has a `.metadata` dict and `.page_content`. Tag the
metadata with an `origin` at ingest time, then::

    from provenant.adapters import rerank_lc_documents
    pairs = vectorstore.similarity_search_with_score(query, k=20)
    docs = rerank_lc_documents(pairs)          # provenance-reranked Documents

LangChain similarity *distances* are smaller-is-better, so pass
`smaller_is_better=True` (default) to convert them to a larger-is-better score
before weighting.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..origins import FALLBACK_ORIGIN
from ..rerank import provenance_rerank


def _origin_of(doc: Any, origin_key: str) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return meta.get(origin_key, FALLBACK_ORIGIN)


def rerank_lc_documents(
    pairs: Sequence[Tuple[Any, float]],
    origin_key: str = "origin",
    smaller_is_better: bool = True,
    trust=None,
    top_k: Optional[int] = None,
) -> List[Any]:
    """Provenance-rerank `(Document, score)` pairs; returns reranked Documents.

    With `smaller_is_better=True`, a distance `d` becomes the similarity `1/(1+d)`
    — a monotonic, bounded conversion (no negatives) so the provenance weight
    scales a sensible base.
    """
    def base(pair: Tuple[Any, float]) -> float:
        _, raw = pair
        return 1.0 / (1.0 + float(raw)) if smaller_is_better else float(raw)

    ranked_pairs = provenance_rerank(
        list(pairs),
        score_of=base,
        origin_of=lambda p: _origin_of(p[0], origin_key),
        trust=trust,
        top_k=top_k,
    )
    return [doc for doc, _ in ranked_pairs]
