"""Provenance reranking — the two-line drop-in.

Most memory / RAG stacks already produce a per-item retrieval score (a cosine
similarity, a BM25 score, a reranker logit). Provenant does NOT replace that. It
multiplies that score by a provenance weight, so a model-generated guess that
happens to be textually similar to the query is demoted below a fact the user
actually stated.

    reranked = provenance_rerank(
        results,
        score_of=lambda r: r["score"],
        origin_of=lambda r: r["metadata"]["origin"],
    )

Fully generic and duck-typed: items can be dicts, dataclasses, ORM rows — the
caller supplies accessors. Stable: ties keep their original order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypeVar

from .weight import compute_provenance_weight

T = TypeVar("T")


@dataclass(frozen=True)
class Scored:
    """An item paired with the numbers behind its new rank — for traceability."""

    item: object
    base_score: float
    weight: float
    adjusted_score: float


def score_with_provenance(
    items: Sequence[T],
    score_of: Callable[[T], float],
    origin_of: Callable[[T], str],
    grounded_of: Optional[Callable[[T], int]] = None,
    depth_of: Optional[Callable[[T], int]] = None,
    trust=None,
) -> List[Scored]:
    """Return Scored(item, base, weight, adjusted) for each item (unsorted).

    Use this when you want the audit trail (every base score, weight and product)
    rather than just a reordered list.
    """
    out: List[Scored] = []
    for it in items:
        base = float(score_of(it))
        weight = compute_provenance_weight(
            origin_of(it),
            grounded_in=grounded_of(it) if grounded_of else 0,
            inference_depth=depth_of(it) if depth_of else 0,
            trust=trust,
        )
        out.append(Scored(item=it, base_score=base, weight=weight,
                          adjusted_score=base * weight))
    return out


def provenance_rerank(
    items: Sequence[T],
    score_of: Callable[[T], float],
    origin_of: Callable[[T], str],
    grounded_of: Optional[Callable[[T], int]] = None,
    depth_of: Optional[Callable[[T], int]] = None,
    trust=None,
    top_k: Optional[int] = None,
) -> List[T]:
    """Reorder `items` by (base score x provenance weight), highest first.

    Stable for equal adjusted scores (original order preserved). `top_k` trims
    the result. The input list is not mutated.
    """
    scored = score_with_provenance(items, score_of, origin_of, grounded_of, depth_of, trust)
    # Stable sort on a key that only flips order when adjusted scores actually differ.
    order = sorted(range(len(scored)), key=lambda i: -scored[i].adjusted_score)
    ranked = [scored[i].item for i in order]
    return ranked[:top_k] if top_k is not None else ranked
