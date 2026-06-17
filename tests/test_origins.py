from provenant import (
    MODEL_GENERATED,
    RETRIEVED,
    USER,
    OriginClassifier,
    base_trust,
    is_grounded,
)
from provenant.origins import FALLBACK_ORIGIN


def test_default_write_site_map():
    c = OriginClassifier()
    assert c.classify("user_message") == USER
    assert c.classify("tool_result") == RETRIEVED
    assert c.classify("generation") == MODEL_GENERATED


def test_unknown_and_missing_site_fall_back_conservatively():
    c = OriginClassifier()
    assert c.classify("some_new_site") == FALLBACK_ORIGIN
    assert c.classify(None) == FALLBACK_ORIGIN
    assert c.classify("") == FALLBACK_ORIGIN
    assert FALLBACK_ORIGIN != USER  # a missing origin must never become the user


def test_register_custom_site():
    c = OriginClassifier().register("my_vector_db", RETRIEVED)
    assert c.classify("my_vector_db") == RETRIEVED


def test_register_rejects_empty():
    try:
        OriginClassifier().register("", USER)
    except ValueError:
        return
    raise AssertionError("empty write_site should raise")


def test_grounded_set():
    assert is_grounded(USER)
    assert is_grounded(RETRIEVED)
    assert not is_grounded(MODEL_GENERATED)


def test_base_trust_ordering_and_fallback():
    assert base_trust(USER) > base_trust(MODEL_GENERATED)
    assert 0.0 < base_trust("nonexistent") < base_trust(USER)
