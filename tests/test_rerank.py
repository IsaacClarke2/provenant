from provenant import (
    MODEL_GENERATED,
    USER,
    provenance_rerank,
    score_with_provenance,
)


def _mk(content, origin, score):
    return {"c": content, "o": origin, "s": score}


def test_confabulation_demoted_below_grounded():
    grounded = _mk("daughter is Mira", USER, 0.60)
    confab = _mk("daughter is Lena", MODEL_GENERATED, 0.85)  # higher base score!
    ranked = provenance_rerank(
        [confab, grounded],
        score_of=lambda r: r["s"],
        origin_of=lambda r: r["o"],
    )
    # despite a higher raw retrieval score, the unsourced claim ranks below
    assert ranked[0] is grounded
    assert ranked[1] is confab


def test_top_k_trims():
    items = [_mk(str(i), USER, i / 10.0) for i in range(5)]
    ranked = provenance_rerank(items, lambda r: r["s"], lambda r: r["o"], top_k=2)
    assert len(ranked) == 2


def test_score_with_provenance_audit_trail():
    item = _mk("x", MODEL_GENERATED, 0.8)
    [scored] = score_with_provenance([item], lambda r: r["s"], lambda r: r["o"])
    assert scored.item is item
    assert scored.base_score == 0.8
    assert abs(scored.adjusted_score - scored.base_score * scored.weight) < 1e-12
    assert scored.weight < 1.0  # model-generated is penalized


def test_stable_on_ties_preserves_input_order():
    a = _mk("a", USER, 0.5)
    b = _mk("b", USER, 0.5)  # identical adjusted score
    ranked = provenance_rerank([a, b], lambda r: r["s"], lambda r: r["o"])
    assert ranked[0] is a and ranked[1] is b
