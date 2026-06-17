"""Provenant — an epistemic memory layer for LLM apps.

Tag every memory with where it came from, fold a trust weight into your existing
retrieval ranking, and present grounded facts separately from inferred ones — so
your model stops repeating things the user never said.

Provenant is a *neutral adapter*: it works WITH your memory store (mem0,
LangChain, Letta, a raw vector DB), never as a replacement.

Quick start:

    from provenant import provenance_rerank

    reranked = provenance_rerank(
        results,
        score_of=lambda r: r["score"],
        origin_of=lambda r: r["origin"],   # "user" | "retrieved" | "model_generated" | ...
    )
"""
from __future__ import annotations

from . import origins
from .origins import (
    GROUNDED_ORIGINS,
    MODEL_GENERATED,
    MODEL_INFERRED,
    RETRIEVED,
    SELF_GENERATED,
    SHARED,
    USER,
    OriginClassifier,
    base_trust,
    is_grounded,
)
from .partition import SectionLabels, partition_by_origin, render_context
from .rerank import Scored, provenance_rerank, score_with_provenance
from .weight import (
    WEIGHT_CEIL,
    WEIGHT_FLOOR,
    WeightExplanation,
    compute_provenance_weight,
    explain,
)

__version__ = "0.1.0"

__all__ = [
    # weight
    "compute_provenance_weight",
    "explain",
    "WeightExplanation",
    "WEIGHT_FLOOR",
    "WEIGHT_CEIL",
    # rerank
    "provenance_rerank",
    "score_with_provenance",
    "Scored",
    # partition
    "partition_by_origin",
    "render_context",
    "SectionLabels",
    # origins
    "origins",
    "OriginClassifier",
    "base_trust",
    "is_grounded",
    "GROUNDED_ORIGINS",
    "USER",
    "SHARED",
    "RETRIEVED",
    "MODEL_INFERRED",
    "MODEL_GENERATED",
    "SELF_GENERATED",
    "__version__",
]
