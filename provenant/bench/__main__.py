"""CLI: `python -m provenant.bench` — prints the confabulation before/after table."""
from __future__ import annotations

from .confab import run


def main() -> None:
    report = run()
    d = report.as_dict()
    print("Provenant — confabulation benchmark (offline, deterministic)")
    print(f"  scenarios: {d['n_scenarios']}   top_k: {d['top_k']}")
    print(f"  confabulation rate  WITHOUT provenance : {d['without_provenance']:.0%}")
    print(f"  confabulation rate  WITH provenance    : {d['with_provenance']:.0%}")
    print(f"  absolute reduction                      : {d['reduction']:.0%}")
    print()
    print("  (proof of mechanism on synthetic scenarios — not LongMemEval/LOCOMO;")
    print("   wire LongMemEvalAdapter for the launch benchmark.)")


if __name__ == "__main__":
    main()
