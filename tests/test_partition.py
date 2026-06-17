from provenant import (
    MODEL_GENERATED,
    MODEL_INFERRED,
    RETRIEVED,
    USER,
    SectionLabels,
    partition_by_origin,
    render_context,
)

FACTS = [
    {"c": "allergic to penicillin", "o": USER},
    {"c": "lives near the sea", "o": RETRIEVED},
    {"c": "probably likes jazz", "o": MODEL_INFERRED},
    {"c": "owns a boat", "o": MODEL_GENERATED},
]


def _origin(f):
    return f["o"]


def _content(f):
    return f["c"]


def test_partition_splits_grounded_and_inferred():
    grounded, inferred = partition_by_origin(FACTS, _origin)
    assert [f["c"] for f in grounded] == ["allergic to penicillin", "lives near the sea"]
    assert [f["c"] for f in inferred] == ["probably likes jazz", "owns a boat"]


def test_render_has_both_sections_and_content():
    out = render_context(FACTS, _origin, _content)
    assert "allergic to penicillin" in out
    assert "probably likes jazz" in out
    # grounded section appears before inferred section
    assert out.index("allergic to penicillin") < out.index("probably likes jazz")


def test_render_omits_inferred_section_when_none():
    grounded_only = [FACTS[0], FACTS[1]]
    out = render_context(grounded_only, _origin, _content)
    assert "owns a boat" not in out
    labels = SectionLabels()
    assert labels.inferred not in out
    assert labels.known in out


def test_configurable_labels_and_caps():
    labels = SectionLabels(known="FACTS", inferred="GUESSES")
    out = render_context(FACTS, _origin, _content, labels=labels, max_per_section=1)
    assert "[FACTS]" in out and "[GUESSES]" in out
    # only one item per section
    assert out.count("\n- ") == 2
