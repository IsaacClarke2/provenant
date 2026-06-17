from provenant.bench import (
    BUILTIN_SCENARIOS,
    LongMemEvalAdapter,
    confabulation_rate,
    run,
)


def test_provenance_reduces_confabulation():
    report = run()
    # the whole point: fewer confabulations reach the context with provenance on
    assert report.with_provenance < report.without_provenance
    assert report.reduction > 0.0


def test_baseline_surfaces_most_confabulations():
    # every scenario's decoy has the top raw retrieval score, so plain ranking
    # lets a confabulation into top-k almost always
    assert run().without_provenance >= 0.8


def test_rates_are_valid_fractions():
    for use in (True, False):
        r = confabulation_rate(use_provenance=use)
        assert 0.0 <= r <= 1.0


def test_empty_scenarios_is_zero():
    assert confabulation_rate([], use_provenance=True) == 0.0


def test_meaningful_reduction_on_builtins():
    report = run(BUILTIN_SCENARIOS, top_k=3)
    assert report.reduction >= 0.5  # large, honest drop (not perfect — one hard case remains)
    assert report.with_provenance > 0.0  # hard case is honestly still a miss


def test_longmemeval_adapter_is_honest_stub():
    try:
        LongMemEvalAdapter().to_scenarios()
    except NotImplementedError:
        return
    raise AssertionError("LongMemEvalAdapter must not fake a result")
