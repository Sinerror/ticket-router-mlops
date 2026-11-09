"""Drift monitoring with Evidently.

Monitored per ticket (computed by the service at prediction time, no labels needed):
    text_len, n_words      - length of the text
    oov_share              - share of words absent from the production model's training vocabulary
    emb_0 .. emb_{k-1}     - projection of the [CLS] embedding on the top-k PCA axes of the reference
    confidence             - max softmax probability
    pred                   - predicted class (prediction drift, chi-square)

Reference = the same features on held-out data of the production model, stored in the bundle.
Alert if share of drifted features >= drift_share_threshold, or prediction drift p-value < threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from ticketflow.inference import OnnxPredictor, features_row

__all__ = ["check_drift", "compute_features", "feature_columns", "features_row"]


def feature_columns(n_components: int) -> tuple[list[str], list[str]]:
    num = ["text_len", "n_words", "oov_share", "confidence"] + [f"emb_{i}" for i in range(n_components)]
    return num, ["pred"]


def compute_features(texts: list[str], predictor: OnnxPredictor) -> pd.DataFrame:
    preds = predictor.predict_batch(list(texts))
    return pd.DataFrame([features_row(t, p, predictor) for t, p in zip(texts, preds, strict=True)])


@dataclass
class DriftResult:
    alert: bool
    drift_share: float
    drifted: list[str] = field(default_factory=list)
    prediction_p_value: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)
    n_current: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def check_drift(
    reference: pd.DataFrame, current: pd.DataFrame, params: dict, html_path: Path | None = None
) -> DriftResult:
    num, cat = feature_columns(params["n_embedding_components"])
    cols = num + cat
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    # Fixed tests so the alert semantics do not depend on sample size: KS for numeric, chi-square for class.
    ks = params.get("ks_p_value", 0.05)
    metrics = [ValueDrift(column=c, method="ks", threshold=ks) for c in num] + [
        ValueDrift(column="pred", method="chisquare", threshold=0.05)
    ]
    report = Report(metrics)
    snap = report.run(
        Dataset.from_pandas(current[cols], data_definition=dd),
        Dataset.from_pandas(reference[cols], data_definition=dd),
    )
    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(html_path))

    scores, drifted = {}, []
    for m, col in zip(snap.dict()["metrics"], cols, strict=True):
        p_value = float(m["value"])
        threshold = float(m["config"].get("threshold") or 0.05)
        scores[col] = p_value
        if p_value < threshold:
            drifted.append(col)
    share = len(drifted) / len(cols)
    pred_p = scores["pred"]
    alert = share >= params["drift_share_threshold"] or pred_p < params["prediction_p_value"]
    return DriftResult(alert, share, drifted, pred_p, scores, len(current))
