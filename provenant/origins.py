"""Origin taxonomy + structural classifier.

An *origin* records WHERE a remembered fact came from — not what it means.
Classification is **structural**: it is decided by the write-site that produced
the fact (a user turn, a tool result, an agent's own reasoning step), never by
parsing the text of the fact itself. There is no regex and no keyword
dictionary here — the caller tells Provenant the write-site when it stores a
fact, and a small, user-extensible data table maps that write-site to an origin.

Trust is a *continuous prior* in the open interval (0, 1): no origin is ever
absolutely certain (1.0) or absolutely worthless (0.0). That headroom is what
lets grounding and inference-depth move a fact's weight smoothly in `weight.py`.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

# --- Generic origin labels for LLM-memory systems, ordered by epistemic trust ---
USER = "user"                        # the user stated it directly
SHARED = "shared"                    # co-established in dialogue (both confirmed)
RETRIEVED = "retrieved"              # pulled from an external source / tool / document
MODEL_INFERRED = "model_inferred"    # the model concluded it from available evidence
MODEL_GENERATED = "model_generated"  # the model produced it with weak / no grounding
SELF_GENERATED = "self_generated"    # the agent produced it about itself / unsourced

# Default continuous trust prior per origin. Strictly inside (0, 1) — asymptotic law.
_BASE_TRUST: Dict[str, float] = {
    USER: 0.97,
    SHARED: 0.85,
    RETRIEVED: 0.72,
    MODEL_INFERRED: 0.45,
    MODEL_GENERATED: 0.30,
    SELF_GENERATED: 0.20,
}

# Unknown origin: conservative — treated as weak model output, NEVER as the user.
# (Mirrors the Mate invariant: a missing origin must not masquerade as ground truth.)
FALLBACK_ORIGIN = MODEL_GENERATED
_FALLBACK_TRUST = 0.30

# Origins considered "grounded" (came from outside the model's own generation).
GROUNDED_ORIGINS = frozenset({USER, SHARED, RETRIEVED})

# Default structural map: write-site label -> origin. A plain data table, not a
# classifier of meaning. Callers extend it for their own write-sites.
_DEFAULT_WRITE_SITES: Dict[str, str] = {
    "user_message": USER,
    "user_profile": USER,
    "confirmed": SHARED,
    "agreed": SHARED,
    "tool_result": RETRIEVED,
    "retrieval": RETRIEVED,
    "document": RETRIEVED,
    "search": RETRIEVED,
    "agent_reasoning": MODEL_INFERRED,
    "inference": MODEL_INFERRED,
    "summary": MODEL_INFERRED,
    "generation": MODEL_GENERATED,
    "assistant_message": MODEL_GENERATED,
    "self_reflection": SELF_GENERATED,
}


def base_trust(origin: str, trust: Optional[Mapping[str, float]] = None) -> float:
    """Continuous trust prior for an origin, in (0, 1).

    `trust` lets a caller pass an explicit trust profile (e.g. a stricter app)
    that overrides individual origins; unspecified origins fall back to the
    defaults, and an unknown origin falls back to the conservative prior — never
    to the user's level. No `except: return default` masking: an unknown origin
    is a defined, conservative case, not an error swallowed silently.
    """
    table = _BASE_TRUST if trust is None else {**_BASE_TRUST, **dict(trust)}
    return table.get(origin, _FALLBACK_TRUST)


def is_grounded(origin: str) -> bool:
    """True when the fact came from outside the model's own generation."""
    return origin in GROUNDED_ORIGINS


class OriginClassifier:
    """Structural write-site -> origin map. Extensible, no text parsing.

    This is the one place an integration declares "facts written from site X are
    origin Y". It is data (a contract table), like a column type — explicitly
    allowed under the no-hardcoded-*behavior* law, because it classifies the
    *provenance channel*, never the *meaning* of the fact.
    """

    def __init__(self, write_sites: Optional[Mapping[str, str]] = None) -> None:
        self._map: Dict[str, str] = dict(_DEFAULT_WRITE_SITES)
        if write_sites:
            self._map.update(write_sites)

    def register(self, write_site: str, origin: str) -> "OriginClassifier":
        """Declare that facts written from `write_site` carry `origin`."""
        if not write_site:
            raise ValueError("write_site must be a non-empty label")
        self._map[write_site] = origin
        return self

    def classify(self, write_site: Optional[str]) -> str:
        """Map a write-site to an origin; unknown/missing -> conservative fallback."""
        if not write_site:
            return FALLBACK_ORIGIN
        return self._map.get(write_site, FALLBACK_ORIGIN)

    def known_sites(self) -> Dict[str, str]:
        return dict(self._map)
