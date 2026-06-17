"""Confabulation-rate benchmark.

The one number that justifies shipping Provenant: *how often does an unsourced,
model-generated claim reach the top-k context?* — measured with provenance
reranking off vs on, on the same memory and the same probes.

`run` is deterministic and needs no LLM and no network: it ranks each scenario's
memory by base score (off) and by base x provenance weight (on), and counts how
often a ground-truth confabulation lands in the top-k. That is the proof of
mechanism. The *launch* benchmark — LongMemEval / LOCOMO with a real embedder
and an LLM judge — plugs into the same shape via `LongMemEvalAdapter`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..rerank import provenance_rerank
from .scenarios import BUILTIN_SCENARIOS, MemoryItem, Scenario


def _rank(items: Sequence[MemoryItem], use_provenance: bool) -> List[MemoryItem]:
    if not use_provenance:
        # Plain retriever order: by base score, stable on ties.
        order = sorted(range(len(items)), key=lambda i: -items[i].score)
        return [items[i] for i in order]
    return provenance_rerank(
        list(items),
        score_of=lambda it: it.score,
        origin_of=lambda it: it.origin,
    )


def _confab_in_topk(scenario: Scenario, top_k: int, use_provenance: bool) -> bool:
    top = _rank(scenario.items, use_provenance)[:top_k]
    return any(it.confabulation for it in top)


def confabulation_rate(
    scenarios: Sequence[Scenario] = BUILTIN_SCENARIOS,
    top_k: int = 3,
    use_provenance: bool = True,
) -> float:
    """Fraction of scenarios where a confabulation reaches the top-k (0..1)."""
    if not scenarios:
        return 0.0
    hits = sum(_confab_in_topk(s, top_k, use_provenance) for s in scenarios)
    return hits / len(scenarios)


@dataclass(frozen=True)
class BenchReport:
    top_k: int
    n_scenarios: int
    without_provenance: float  # confab rate, reranking OFF
    with_provenance: float     # confab rate, reranking ON
    reduction: float           # absolute drop in confab rate

    def as_dict(self) -> Dict[str, float]:
        return {
            "top_k": self.top_k,
            "n_scenarios": self.n_scenarios,
            "without_provenance": self.without_provenance,
            "with_provenance": self.with_provenance,
            "reduction": self.reduction,
        }


def run(scenarios: Sequence[Scenario] = BUILTIN_SCENARIOS, top_k: int = 3) -> BenchReport:
    """Run the confabulation benchmark reranking-off vs reranking-on."""
    without = confabulation_rate(scenarios, top_k, use_provenance=False)
    with_ = confabulation_rate(scenarios, top_k, use_provenance=True)
    return BenchReport(
        top_k=top_k,
        n_scenarios=len(scenarios),
        without_provenance=without,
        with_provenance=with_,
        reduction=without - with_,
    )


class LongMemEvalAdapter:
    """Hook for the real launch benchmark (LongMemEval / LOCOMO).

    Intentionally NOT implemented: it requires the external dataset, a real
    embedder, and an LLM judge — none of which ship in this package, and faking
    the result would be dishonest. To run the launch benchmark, subclass this,
    load the dataset, retrieve with your embedder to produce `MemoryItem`s
    (origin tagged at ingest, `confabulation` = the judge's hallucination label),
    and feed the resulting `Scenario`s to `run()` above. The offline
    `BUILTIN_SCENARIOS` benchmark is the deterministic proof of mechanism in the
    meantime.
    """

    def to_scenarios(self) -> List[Scenario]:  # pragma: no cover - scaffold
        raise NotImplementedError(
            "LongMemEval/LOCOMO requires the external dataset + an LLM judge. "
            "Implement to_scenarios() to map judged retrievals into Scenario objects, "
            "then call provenant.bench.confab.run(). The README explains the contract."
        )
