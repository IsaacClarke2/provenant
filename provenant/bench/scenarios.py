"""Built-in confabulation scenarios for the offline benchmark.

Each scenario is a probe plus a small memory: one grounded answer the user
actually stated, one or more *confabulations* (model-generated/inferred claims
that are textually tempting — high base retrieval score — but unsourced), and
some grounded filler. The `score` field is the base retriever similarity a
plain vector search would assign for that probe; `confabulation=True` marks the
ground-truth decoys the bench is trying to keep out of the top-k.

These are an honest, deterministic *proof of mechanism*, not LongMemEval/LOCOMO
numbers. See `provenant.bench.confab.LongMemEvalAdapter` for the real launch
benchmark hook.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Tuple

from ..origins import (
    MODEL_GENERATED,
    MODEL_INFERRED,
    RETRIEVED,
    USER,
)


@dataclass(frozen=True)
class MemoryItem:
    content: str
    origin: str
    score: float          # base retriever similarity for the scenario's probe
    confabulation: bool    # ground-truth: is this an unsourced decoy?


@dataclass(frozen=True)
class Scenario:
    probe: str
    items: Tuple[MemoryItem, ...]


def similarity(probe: str, content: str) -> float:
    """Lightweight, deterministic text similarity (difflib) for example use.

    Not used by the built-in scenarios (they carry explicit scores so the bench
    is reproducible); offered so `examples/` can build scenarios from raw text.
    """
    return difflib.SequenceMatcher(None, probe.lower(), content.lower()).ratio()


# Five "fixable" scenarios (provenance reranking pushes the decoy out of top-3)
# and one deliberately "hard" one (the decoy is so dominant it survives) — so the
# benchmark reports a realistic, non-perfect reduction, not a suspicious 0%.
BUILTIN_SCENARIOS: List[Scenario] = [
    Scenario(
        probe="what is the user's daughter's name?",
        items=(
            MemoryItem("User's daughter is named Mira", USER, 0.62, False),
            MemoryItem("User's daughter is named Lena", MODEL_GENERATED, 0.88, True),
            MemoryItem("User has a daughter", USER, 0.55, False),
            MemoryItem("User mentioned a school event", RETRIEVED, 0.40, False),
            MemoryItem("User talked about parenting", RETRIEVED, 0.38, False),
        ),
    ),
    Scenario(
        probe="what is the user allergic to?",
        items=(
            MemoryItem("User is allergic to penicillin", USER, 0.70, False),
            MemoryItem("User is allergic to peanuts", MODEL_GENERATED, 0.85, True),
            MemoryItem("User takes ibuprofen sometimes", RETRIEVED, 0.55, False),
            MemoryItem("User visited a clinic last week", RETRIEVED, 0.52, False),
            MemoryItem("User prefers generic medication", USER, 0.48, False),
        ),
    ),
    Scenario(
        probe="where does the user live?",
        items=(
            MemoryItem("User lives in Lisbon", USER, 0.60, False),
            MemoryItem("User lives in Madrid", MODEL_INFERRED, 0.82, True),
            MemoryItem("User travels to Spain for work", RETRIEVED, 0.54, False),
            MemoryItem("User mentioned the ocean nearby", USER, 0.50, False),
            MemoryItem("User likes warm climates", RETRIEVED, 0.44, False),
        ),
    ),
    Scenario(
        probe="what pet does the user have?",
        items=(
            MemoryItem("User has a cat named Boris", USER, 0.66, False),
            MemoryItem("User has a dog", MODEL_GENERATED, 0.86, True),
            MemoryItem("User posts pet photos", RETRIEVED, 0.53, False),
            MemoryItem("User buys cat food", USER, 0.51, False),
            MemoryItem("User dislikes loud animals", RETRIEVED, 0.41, False),
        ),
    ),
    Scenario(
        probe="what is the user's favorite drink?",
        items=(
            MemoryItem("User loves green tea", USER, 0.64, False),
            MemoryItem("User loves espresso", MODEL_INFERRED, 0.83, True),
            MemoryItem("User avoids sugary drinks", RETRIEVED, 0.56, False),
            MemoryItem("User drinks tea in the morning", USER, 0.52, False),
            MemoryItem("User mentioned a tea shop", RETRIEVED, 0.45, False),
        ),
    ),
    # Hard case: the decoy's base score is overwhelming and the grounded answer is
    # weak. Provenance helps but does not fully evict it — kept honest on purpose.
    Scenario(
        probe="what does the user do for work?",
        items=(
            MemoryItem("User works as a teacher", USER, 0.40, False),
            MemoryItem("User works as a doctor", MODEL_INFERRED, 0.95, True),
            MemoryItem("User commutes by train", RETRIEVED, 0.50, False),
            MemoryItem("User likes coffee", USER, 0.45, False),
            MemoryItem("User has meetings on Monday", RETRIEVED, 0.42, False),
        ),
    ),
]
