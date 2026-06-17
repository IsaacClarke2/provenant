"""Provenance weight — the continuous trust multiplier.

`compute_provenance_weight` turns an origin (plus optional grounding anchors and
inference depth) into a single multiplier you fold into your existing retrieval
score, so a model-generated guess ranks BELOW a fact the user actually stated.

Design laws (carried over from the kernel this was extracted from):
  * Continuous, no step functions. base prior + grounding bonus, scaled by an
    exponential inference-depth penalty.
  * Asymptotic bounds. The weight lives strictly inside the OPEN interval
    (WEIGHT_FLOOR, WEIGHT_CEIL) — it never reaches an absolute floor or ceiling,
    so there is always headroom to move.
  * Pure & traceable. No I/O, no hidden state, deterministic. `explain()` returns
    every term so any ranking decision can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .origins import base_trust

# Weight lives in the OPEN interval (FLOOR, CEIL). EPS keeps it off the exact bounds.
WEIGHT_FLOOR = 0.10
WEIGHT_CEIL = 1.30
_EPS = 1e-3

# Each independent grounding anchor adds trust with diminishing returns
# (asymptotes toward +GROUNDING_MAX). One strong anchor helps a lot; the tenth
# barely moves it.
GROUNDING_MAX = 0.30
GROUNDING_DECAY = 0.5  # fraction of remaining headroom closed per anchor

# Each inference hop away from grounded evidence multiplies trust down.
DEPTH_FACTOR = 0.77


def _soft_clamp(value: float, low: float, high: float) -> float:
    """Keep `value` strictly inside the open interval (low, high)."""
    return min(high - _EPS, max(low + _EPS, value))


def grounding_bonus(grounded_in: int) -> float:
    """Diminishing-returns bonus in [0, GROUNDING_MAX) for N grounding anchors."""
    n = max(0, int(grounded_in))
    return GROUNDING_MAX * (1.0 - GROUNDING_DECAY ** n)


def depth_penalty(inference_depth: int) -> float:
    """Multiplicative penalty for distance (in inference hops) from evidence."""
    return DEPTH_FACTOR ** max(0, int(inference_depth))


def compute_provenance_weight(
    origin: str,
    grounded_in: int = 0,
    inference_depth: int = 0,
    trust: Optional[Mapping[str, float]] = None,
) -> float:
    """Trust multiplier for a fact, in the open interval (FLOOR, CEIL).

    origin          - where the fact came from (see provenant.origins)
    grounded_in     - number of independent anchors supporting it (>= 0)
    inference_depth - hops of model reasoning between evidence and this fact (>= 0)
    trust           - optional per-origin trust override profile
    """
    base = base_trust(origin, trust)
    raw = (base + grounding_bonus(grounded_in)) * depth_penalty(inference_depth)
    return _soft_clamp(raw, WEIGHT_FLOOR, WEIGHT_CEIL)


@dataclass(frozen=True)
class WeightExplanation:
    """Full provenance of a weight — every term, for audit/traceability."""

    origin: str
    base: float
    grounding_bonus: float
    depth_penalty: float
    raw: float
    weight: float


def explain(
    origin: str,
    grounded_in: int = 0,
    inference_depth: int = 0,
    trust: Optional[Mapping[str, float]] = None,
) -> WeightExplanation:
    """Same computation as `compute_provenance_weight`, with every term exposed."""
    base = base_trust(origin, trust)
    gb = grounding_bonus(grounded_in)
    dp = depth_penalty(inference_depth)
    raw = (base + gb) * dp
    return WeightExplanation(
        origin=origin,
        base=base,
        grounding_bonus=gb,
        depth_penalty=dp,
        raw=raw,
        weight=_soft_clamp(raw, WEIGHT_FLOOR, WEIGHT_CEIL),
    )
