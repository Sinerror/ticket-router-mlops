from ticketflow.gate import check

PARAMS = {"min_macro_f1_delta": 0.0, "max_class_drop_pp": 5.0, "min_macro_f1": 0.7}


def m(macro, **per_class):
    return {"macro_f1": macro, "per_class_f1": per_class}


def test_first_model_only_needs_floor():
    assert check(m(0.9, A=0.9), None, PARAMS).passed
    assert not check(m(0.6, A=0.6), None, PARAMS).passed


def test_better_candidate_passes():
    r = check(m(0.95, A=0.95, B=0.95), m(0.90, A=0.9, B=0.9), PARAMS)
    assert r.passed and r.reasons == []


def test_macro_regression_blocks():
    r = check(m(0.88, A=0.88), m(0.90, A=0.90), PARAMS)
    assert not r.passed
    assert "macro-F1" in r.reasons[0]


def test_single_class_drop_blocks_even_if_macro_improves():
    # macro improves thanks to class C, but class B loses 10 pp -> blocked
    r = check(m(0.93, A=0.95, B=0.85, C=0.99), m(0.90, A=0.95, B=0.95, C=0.80), PARAMS)
    assert not r.passed
    assert r.worst_class == "B" and round(r.worst_class_drop_pp) == 10


def test_small_classes_are_not_judged_per_class():
    cand = {**m(0.93, A=0.95, B=0.80), "support": {"A": 100, "B": 5}}
    r = check(cand, m(0.90, A=0.90, B=0.95), {**PARAMS, "min_class_support": 15})
    assert r.passed


def test_classes_absent_in_production_are_ignored():
    r = check(m(0.95, A=0.95, NEW=0.9), m(0.90, A=0.95, NEW=None), PARAMS)
    assert r.passed
