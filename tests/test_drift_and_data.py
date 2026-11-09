import numpy as np
import pandas as pd
import pytest

from ticketflow.config import load_params
from ticketflow.data import _shares, clean
from ticketflow.metrics import evaluate

DRIFT = load_params()["drift"]


def _features(n, rng, shift=0.0, preds=("A", "B", "C")):
    df = pd.DataFrame(
        {
            "text_len": rng.normal(50 + 20 * shift, 10, n),
            "n_words": rng.normal(8 + 3 * shift, 2, n),
            "oov_share": rng.beta(1, 20, n) + 0.2 * shift,
            "confidence": rng.beta(20, 1, n) - 0.2 * shift,
            "pred": rng.choice(preds, n),
        }
    )
    for i in range(DRIFT["n_embedding_components"]):
        df[f"emb_{i}"] = rng.normal(shift, 1, n)
    return df


def test_no_alert_on_same_distribution():
    pytest.importorskip("evidently")
    from ticketflow.drift import check_drift

    rng = np.random.default_rng(0)
    r = check_drift(_features(800, rng), _features(150, rng), DRIFT)
    assert not r.alert


def test_alert_on_shift():
    pytest.importorskip("evidently")
    from ticketflow.drift import check_drift

    rng = np.random.default_rng(0)
    r = check_drift(_features(800, rng), _features(150, rng, shift=1.0, preds=("A", "A", "D")), DRIFT)
    assert r.alert and r.drift_share >= DRIFT["drift_share_threshold"]


def test_shift_schedule_ramps_after_drift_month():
    p = load_params()["data"]
    start = p["drift_month"] * p["days_per_month"]
    assert _shares(start - 1, p) == (p["surge_share_before"], p["noisy_share_before"])
    s, n = _shares(start + 10, p)
    assert s == pytest.approx(p["surge_share_after"]) and n == pytest.approx(p["noisy_share_after"])


def test_clean_fills_placeholders_and_dedups():
    raw = pd.DataFrame(
        {
            "instruction": ["где заказ {{Order Number}}", "где заказ {{Order Number}}", "привет {{Name}}"],
            "category": ["ORDER", "ORDER", "CONTACT"],
            "flags": ["B", "BQ", "BZ"],
        }
    )
    df = clean(raw, "QZ", seed=0)
    assert not df["text"].str.contains("{{").any()
    assert df["text"].is_unique


def test_macro_f1_ignores_absent_classes():
    r = evaluate(["A", "B"], ["A", "B"], ["A", "B", "NEW"])
    assert r["macro_f1"] == 1.0 and r["per_class_f1"]["NEW"] is None
