from provenant import (
    MODEL_GENERATED,
    MODEL_INFERRED,
    SELF_GENERATED,
    USER,
    WEIGHT_CEIL,
    WEIGHT_FLOOR,
    compute_provenance_weight,
    explain,
)


def test_monotonic_by_trust():
    assert (
        compute_provenance_weight(USER)
        > compute_provenance_weight(MODEL_INFERRED)
        > compute_provenance_weight(SELF_GENERATED)
    )


def test_open_interval_bounds():
    # extreme low: weak origin, deep inference
    lo = compute_provenance_weight(SELF_GENERATED, inference_depth=50)
    # extreme high: top origin, fully grounded
    hi = compute_provenance_weight(USER, grounded_in=100)
    for w in (lo, hi):
        assert WEIGHT_FLOOR < w < WEIGHT_CEIL  # never hits absolute floor/ceiling


def test_grounding_raises_weight():
    assert compute_provenance_weight(USER, grounded_in=3) > compute_provenance_weight(USER)


def test_inference_depth_lowers_weight():
    assert (
        compute_provenance_weight(MODEL_INFERRED, inference_depth=2)
        < compute_provenance_weight(MODEL_INFERRED)
    )


def test_unknown_origin_is_conservative_not_user():
    unknown = compute_provenance_weight("totally_unknown_origin")
    assert unknown < compute_provenance_weight(USER)
    # falls back to the weak-model prior, not the user's
    assert abs(unknown - compute_provenance_weight(MODEL_GENERATED)) < 1e-9


def test_trust_profile_override():
    strict = compute_provenance_weight(USER, trust={USER: 0.5})
    assert abs(strict - 0.5) < 1e-9


def test_explain_terms_reconstruct_weight():
    e = explain(USER, grounded_in=2, inference_depth=1)
    assert abs((e.base + e.grounding_bonus) * e.depth_penalty - e.raw) < 1e-12
    assert WEIGHT_FLOOR < e.weight < WEIGHT_CEIL
