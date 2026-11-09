import numpy as np

from ticketflow.inference import Prediction
from ticketflow.router import Router, in_canary


class FakePredictor:
    def __init__(self, label, version):
        self.label, self.version = label, version

    def predict_batch(self, texts):
        return [Prediction(self.label, 0.9, np.zeros(2), np.zeros(5)) for _ in texts]


def keys(n):
    return [str(i) for i in range(n)]


def test_off_mode_serves_production_only():
    r = Router(FakePredictor("A", "1"), FakePredictor("B", "2"), mode="off")
    out = r.route_batch(keys(100), ["t"] * 100)
    assert all(o.served.label == "A" and o.candidate is None for o in out)


def test_shadow_never_serves_candidate_but_scores_it():
    r = Router(FakePredictor("A", "1"), FakePredictor("B", "2"), mode="shadow")
    out = r.route_batch(keys(100), ["t"] * 100)
    assert all(o.served_by == "production" for o in out)
    assert all(o.candidate.label == "B" for o in out)


def test_canary_fraction_and_stickiness():
    r = Router(FakePredictor("A", "1"), FakePredictor("B", "2"), mode="canary", canary_fraction=0.2)
    out = r.route_batch(keys(5000), ["t"] * 5000)
    share = np.mean([o.served_by == "canary" for o in out])
    assert 0.17 < share < 0.23
    again = r.route_batch(keys(5000), ["t"] * 5000)
    assert [o.served_by for o in out] == [o.served_by for o in again]
    assert in_canary("42", 1.0) and not in_canary("42", 0.0)


def test_dropping_candidate_resets_mode():
    r = Router(FakePredictor("A", "1"), FakePredictor("B", "2"), mode="canary", canary_fraction=0.5)
    r.configure(production=FakePredictor("B", "2"), drop_candidate=True)
    assert r.mode == "off" and r.candidate is None
    assert r.route("x", "t").served_version == "2"
