"""Partition facts by provenance and render an epistemically-honest context.

The retrieval-ranking multiplier (`rerank`) decides *which* facts reach the
prompt. This module decides *how they are presented* once there: grounded facts
and model-generated guesses go into separate, labelled sections, so the model
structurally sees the boundary between "what I was told" and "what I inferred"
instead of reading one undifferentiated blob.

Facts are arbitrary objects; the caller supplies small accessor callables
(`origin_of`, `content_of`). Nothing here parses fact text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .origins import is_grounded


@dataclass(frozen=True)
class SectionLabels:
    """Configurable section headers. Defaults are English and neutral."""

    known: str = "What I know (from you / sources)"
    inferred: str = "What I think (inferred — may be wrong)"


def partition_by_origin(
    facts: Iterable[object],
    origin_of: Callable[[object], str],
) -> Tuple[List[object], List[object]]:
    """Split facts into (grounded, inferred) by their origin. Order preserved."""
    grounded: List[object] = []
    inferred: List[object] = []
    for fact in facts:
        (grounded if is_grounded(origin_of(fact)) else inferred).append(fact)
    return grounded, inferred


def render_context(
    facts: Sequence[object],
    origin_of: Callable[[object], str],
    content_of: Callable[[object], str],
    labels: Optional[SectionLabels] = None,
    max_per_section: Optional[int] = None,
    bullet: str = "- ",
) -> str:
    """Render facts into two labelled sections for prompt injection.

    The inferred section is omitted entirely when there is nothing inferred, so a
    fully-grounded context stays clean. `max_per_section` caps each section
    independently (None = no cap).
    """
    labels = labels or SectionLabels()
    grounded, inferred = partition_by_origin(facts, origin_of)
    if max_per_section is not None:
        grounded = grounded[:max_per_section]
        inferred = inferred[:max_per_section]

    blocks: List[str] = []
    if grounded:
        lines = "\n".join(bullet + content_of(f) for f in grounded)
        blocks.append(f"[{labels.known}]\n{lines}")
    if inferred:
        lines = "\n".join(bullet + content_of(f) for f in inferred)
        blocks.append(f"[{labels.inferred}]\n{lines}")
    return "\n\n".join(blocks)
