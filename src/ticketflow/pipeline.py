"""Retraining pipeline steps. Each function is one Airflow task and is also called by the simulator.

datasets  -> train_candidate -> gate_and_register -> (shadow -> canary -> promote, see rollout)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ticketflow import gate as quality_gate
from ticketflow import registry
from ticketflow.config import ARTIFACTS_DIR, load_params
from ticketflow.data import load_history, load_labels, load_stream
from ticketflow.drift import compute_features
from ticketflow.inference import OnnxPredictor
from ticketflow.metrics import evaluate
from ticketflow.models.baseline import train_baseline
from ticketflow.models.transformer import train_transformer


@dataclass
class Datasets:
    train: pd.DataFrame
    golden: pd.DataFrame
    reference_texts: list[str]
    upto_day: int | None
    info: dict = field(default_factory=dict)


@dataclass
class Candidate:
    run_id: str
    bundle_dir: Path
    metrics: dict
    baseline_metrics: dict
    train_seconds: float


def corrupt_labels(
    df: pd.DataFrame, labels: list[str], rate: float, seed: int = 0, swap: tuple[str, str] | None = None
) -> pd.DataFrame:
    """Intentionally bad labelling for the gate drill: random flips and/or a systematic class swap."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    if swap:
        mask = (df["label"] == swap[0]) & (rng.random(len(df)) < max(rate, 0.5))
        df.loc[mask, "label"] = swap[1]
    flip = rng.random(len(df)) < rate
    df.loc[flip, "label"] = rng.choice(labels, flip.sum())
    return df


def make_datasets(upto_day: int | None = None, stream: pd.DataFrame | None = None) -> Datasets:
    """Training data = history + labelled tickets of the last `train_window_days` up to `upto_day`.

    Golden set = held-out share of the freshest labelled tickets (last `golden_days`).
    Drift reference texts = unlabelled tickets of the same fresh window (never seen in training).
    For the bootstrap model (upto_day=None) both come from a held-out share of the history.
    """
    p = load_params()
    tp, seed = p["train"], p["data"]["seed"]
    rng = np.random.default_rng(seed + (upto_day or 0))
    history = load_history().assign(day=-1, labelled=True)

    if upto_day is None:
        golden_mask = rng.random(len(history)) < tp["bootstrap_golden_fraction"]
        golden = history[golden_mask]
        train = history[~golden_mask]
        return Datasets(
            train, golden, list(golden["text"]), None, {"n_train": len(train), "n_golden": len(golden)}
        )

    stream = load_stream() if stream is None else stream
    window = stream[(stream["day"] <= upto_day) & (stream["day"] > upto_day - tp["train_window_days"])]
    labelled = window[window["labelled"]]
    fresh = labelled["day"] > upto_day - tp["golden_days"]
    golden_mask = fresh & (rng.random(len(labelled)) < tp["golden_fraction"])
    golden = labelled[golden_mask]
    train = pd.concat([history, labelled[~golden_mask]], ignore_index=True)
    fresh_unlabelled = window[(~window["labelled"]) & (window["day"] > upto_day - tp["golden_days"])]
    return Datasets(
        train,
        golden,
        list(fresh_unlabelled["text"]),
        upto_day,
        {"n_train": len(train), "n_golden": len(golden), "n_reference": len(fresh_unlabelled)},
    )


def train_candidate(
    ds: Datasets, run_name: str, out_dir: Path | None = None, tags: dict | None = None
) -> Candidate:
    p = load_params()
    labels = load_labels()
    out_dir = Path(out_dir or ARTIFACTS_DIR / "candidates" / run_name)
    registry.setup()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"stage": "candidate", **(tags or {})})
        mlflow.log_params({f"train.{k}": v for k, v in p["train"].items()})
        mlflow.log_params({"upto_day": ds.upto_day, **{f"data.{k}": v for k, v in ds.info.items()}})
        mlflow.log_dict(ds.train["label"].value_counts().to_dict(), "train_label_counts.json")

        # 1) classical baseline, logged for comparison only
        t0 = time.time()
        base = train_baseline(ds.train["text"], ds.train["label"], seed=p["data"]["seed"])
        base_metrics = evaluate(ds.golden["label"], base.predict(ds.golden["text"]), labels)
        mlflow.log_metrics(
            {
                "baseline_macro_f1": base_metrics["macro_f1"],
                "baseline_accuracy": base_metrics["accuracy"],
                "baseline_train_seconds": time.time() - t0,
            }
        )

        # 2) rubert-tiny2 fine-tuning (PyTorch loop) -> ONNX bundle
        t0 = time.time()
        train_transformer(
            ds.train["text"],
            ds.train["label"],
            labels,
            out_dir,
            p["train"],
            seed=p["data"]["seed"],
            log_fn=lambda m, s: mlflow.log_metrics(m, step=s),
        )
        train_seconds = time.time() - t0

        # 3) evaluation on the golden set with the *serving* runtime (ONNX int8)
        predictor = OnnxPredictor(out_dir, variant="int8", threads=4)
        preds = predictor.predict_batch(list(ds.golden["text"]))
        metrics = evaluate(ds.golden["label"], [x.label for x in preds], labels)
        mlflow.log_metrics(
            {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "train_seconds": train_seconds}
        )
        mlflow.log_metrics({f"f1_{k}": v for k, v in metrics["per_class_f1"].items() if v is not None})
        mlflow.log_dict(metrics, "golden_metrics.json")

        # 4) drift reference: features of fresh, unseen traffic as scored by this model
        ref = compute_features(ds.reference_texts or list(ds.golden["text"]), predictor)
        ref.to_parquet(out_dir / "reference_features.parquet", index=False)
        ds.golden.to_parquet(out_dir / "golden.parquet", index=False)
        (out_dir / "golden_metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    return Candidate(run.info.run_id, out_dir, metrics, base_metrics, train_seconds)


def evaluate_bundle(bundle_dir: Path, golden: pd.DataFrame) -> dict:
    predictor = OnnxPredictor(bundle_dir, variant="int8", threads=4)
    preds = predictor.predict_batch(list(golden["text"]))
    return evaluate(golden["label"], [x.label for x in preds], load_labels())


def gate_and_register(
    candidate: Candidate, golden: pd.DataFrame, production_bundle: Path | None
) -> tuple[quality_gate.GateResult, str | None]:
    """Score production on the same golden set, apply the gate, register the candidate if it passes."""
    p = load_params()
    prod_metrics = evaluate_bundle(production_bundle, golden) if production_bundle else None
    result = quality_gate.check(candidate.metrics, prod_metrics, p["gate"])
    registry.setup()
    with mlflow.start_run(run_id=candidate.run_id):
        mlflow.log_dict(result.to_dict(), "gate.json")
        mlflow.log_metric("gate_passed", int(result.passed))
        if prod_metrics:
            mlflow.log_metric("production_macro_f1_on_golden", prod_metrics["macro_f1"])
        mlflow.set_tag("gate", "passed" if result.passed else "blocked")
        if not result.passed:
            mlflow.set_tag("gate_reasons", "; ".join(result.reasons))
            return result, None
        version = registry.log_and_register(
            candidate.bundle_dir,
            tags={
                "macro_f1": f"{candidate.metrics['macro_f1']:.4f}",
                "run_id": candidate.run_id,
                "gate": json.dumps(result.to_dict()),
            },
        )
    registry.set_alias(registry.CHALLENGER, version)
    return result, version
