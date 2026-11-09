"""'Six months in under an hour': replay the ticket stream day by day through the full lifecycle.

Two worlds are compared on the same traffic:
    frozen  - the bootstrap model stays in production forever (no lifecycle)
    cycle   - drift monitor -> labelling -> retrain -> gate -> registry -> shadow -> canary -> promote

The code paths are the real ones: pipeline.train_candidate / gate_and_register (Airflow tasks),
registry aliases (MLflow), Router (FastAPI service), check_drift (Evidently).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ticketflow.config import ARTIFACTS_DIR, REPORTS_DIR, load_params

SIM_DIR = ARTIFACTS_DIR / "simulation"


def _isolated_mlflow() -> None:
    """Simulation gets its own throw-away MLflow DB/artifact store, so reruns are reproducible."""
    if SIM_DIR.exists():
        shutil.rmtree(SIM_DIR)
    SIM_DIR.mkdir(parents=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(SIM_DIR / 'mlflow.db').as_posix()}"
    os.environ["MLFLOW_ARTIFACT_LOCATION"] = (SIM_DIR / "mlartifacts").as_uri()
    os.environ["MLFLOW_EXPERIMENT"] = "simulation"


def _pooled_f1(df: pd.DataFrame, pred_col: str, labels: list[str]) -> float:
    from ticketflow.metrics import evaluate

    return evaluate(df["label"], df[pred_col], labels)["macro_f1"] if len(df) else float("nan")


def run(out_dir: Path = REPORTS_DIR, gate_drill: bool = True, rollback_drill: bool = True) -> dict:
    _isolated_mlflow()
    from ticketflow import registry
    from ticketflow.data import load_labels, load_stream
    from ticketflow.drift import check_drift, features_row
    from ticketflow.inference import OnnxPredictor
    from ticketflow.pipeline import gate_and_register, make_datasets, train_candidate
    from ticketflow.router import Router

    p = load_params()
    dp, rp, drp = p["data"], p["rollout"], p["drift"]
    stream, labels = load_stream(), load_labels()
    n_days = dp["n_months"] * dp["days_per_month"]
    drift_start = dp["drift_month"] * dp["days_per_month"]
    out_dir.mkdir(parents=True, exist_ok=True)
    wall0 = time.time()

    # --- bootstrap: first production model --------------------------------------------------------
    ds = make_datasets(None)
    cand = train_candidate(
        ds, "bootstrap", out_dir=SIM_DIR / "candidates" / "bootstrap", tags={"trigger": "bootstrap"}
    )
    gate, version = gate_and_register(cand, ds.golden, None)
    assert version, f"bootstrap model failed the gate: {gate.reasons}"
    registry.promote(version)
    prod = OnnxPredictor(cand.bundle_dir, version=version)
    frozen = prod
    reference = pd.read_parquet(cand.bundle_dir / "reference_features.parquet")
    router = Router(prod)

    events: list[dict] = [
        {
            "day": -1,
            "event": "bootstrap",
            "version": version,
            "golden_macro_f1": round(cand.metrics["macro_f1"], 4),
        }
    ]
    rows, feats = [], []
    state, state_until, cooldown_until, streak = "idle", -1, 0, 0
    cand_pred, cand_version, trigger_day, release_wall = None, None, None, 0.0
    shadow_stats: list[float] = []

    for day in range(n_days):
        x = stream[stream["day"] == day]
        texts, keys = list(x["text"]), [str(t) for t in x["ticket_id"]]
        routed = router.route_batch(keys, texts)
        frozen_preds = (
            [r.production for r in routed] if frozen is router.production else frozen.predict_batch(texts)
        )

        day_df = x[["ticket_id", "day", "month", "label", "labelled"]].copy()
        day_df["served"] = [r.served.label for r in routed]
        day_df["served_by"] = [r.served_by for r in routed]
        day_df["production"] = [r.production.label for r in routed]
        day_df["candidate"] = [r.candidate.label if r.candidate else None for r in routed]
        day_df["frozen"] = [f.label for f in frozen_preds]
        day_df["mode"] = router.mode
        day_df["prod_version"] = router.production.version
        rows.append(day_df)
        feats.append(
            pd.DataFrame(
                [features_row(t, r.production, router.production) for t, r in zip(texts, routed, strict=True)]
            ).assign(day=day)
        )

        # --- drift monitor on a rolling window of the *current* production model's traffic -------
        window = pd.concat(feats[-drp["window_days"] :], ignore_index=True)
        drift = check_drift(reference, window, drp)
        day_df["drift_share"] = drift.drift_share
        streak = streak + 1 if drift.alert else 0
        confirmed = streak >= drp.get("persistence_checks", 1)
        if confirmed and state == "idle" and day >= cooldown_until:
            events.append(
                {
                    "day": day,
                    "event": "drift_alert",
                    "drift_share": round(drift.drift_share, 2),
                    "drifted": drift.drifted,
                    "prediction_p_value": drift.prediction_p_value,
                }
            )
            state, state_until = "wait_labels", day + rp["labeling_lag_days"]
            check_drift(reference, window, drp, html_path=out_dir / f"drift_alert_day{day}.html")

        # --- lifecycle state machine (end of day) ------------------------------------------------
        if state == "wait_labels" and day >= state_until:
            t0 = time.time()
            ds = make_datasets(day, stream)
            cand = train_candidate(
                ds,
                f"retrain_day{day}",
                out_dir=SIM_DIR / "candidates" / f"retrain_day{day}",
                tags={"trigger": "drift_alert", "day": day},
            )
            gate, cand_version = gate_and_register(cand, ds.golden, router.production.bundle_dir)
            release_wall = time.time() - t0
            ev = {
                "day": day,
                "event": "retrain",
                "gate_passed": gate.passed,
                "reasons": gate.reasons,
                "candidate_macro_f1": round(gate.candidate_macro_f1, 4),
                "production_macro_f1": round(gate.production_macro_f1 or 0, 4),
                "n_train": ds.info["n_train"],
                "n_golden": ds.info["n_golden"],
                "train_seconds": round(cand.train_seconds, 1),
                "pipeline_seconds": round(release_wall, 1),
            }
            events.append(ev)
            if gate.passed:
                cand_pred = OnnxPredictor(cand.bundle_dir, version=cand_version)
                router.configure(candidate=cand_pred, mode="shadow")
                state, state_until, trigger_day, shadow_stats = "shadow", day + rp["shadow_days"], day, []
                cand_reference = pd.read_parquet(cand.bundle_dir / "reference_features.parquet")
            else:
                state, cooldown_until = "idle", day + drp["cooldown_days"]
        elif state == "shadow":
            shadow_stats.append(float((day_df["candidate"] != day_df["production"]).mean()))
            if day >= state_until:
                disagreement = float(np.mean(shadow_stats))
                ok = disagreement <= rp["max_shadow_disagreement"]
                events.append(
                    {"day": day, "event": "shadow_done", "disagreement": round(disagreement, 3), "passed": ok}
                )
                if ok:
                    router.configure(mode="canary", canary_fraction=rp["canary_fraction"])
                    state, state_until = "canary", day + rp["canary_days"]
                else:
                    router.configure(drop_candidate=True)
                    state, cooldown_until = "idle", day + drp["cooldown_days"]
        elif state == "canary" and day >= state_until:
            hist = pd.concat(rows[-rp["canary_days"] :], ignore_index=True)
            lab = hist[hist["labelled"]]
            can, prd = lab[lab["served_by"] == "canary"], lab[lab["served_by"] == "production"]
            acc_c = float((can["served"] == can["label"]).mean()) if len(can) else 0.0
            acc_p = float((prd["served"] == prd["label"]).mean()) if len(prd) else 0.0
            ok = (acc_p - acc_c) * 100 <= rp["max_canary_drop_pp"]
            events.append(
                {
                    "day": day,
                    "event": "canary_done",
                    "canary_accuracy": round(acc_c, 3),
                    "production_accuracy": round(acc_p, 3),
                    "n_canary": len(can),
                    "passed": ok,
                }
            )
            if ok:
                registry.promote(cand_version)
                router.configure(production=cand_pred, drop_candidate=True)
                reference, feats, streak = cand_reference, [], 0
                events.append(
                    {
                        "day": day,
                        "event": "promoted",
                        "version": cand_version,
                        "days_from_labels_to_production": day - trigger_day,
                    }
                )
            else:
                router.configure(drop_candidate=True)
            state, cooldown_until = "idle", day + drp["cooldown_days"]

    log = pd.concat(rows, ignore_index=True)
    log.to_parquet(SIM_DIR / "predictions.parquet", index=False)

    # --- summary metrics -------------------------------------------------------------------------
    monthly = []
    for m, g in log.groupby("month"):
        monthly.append(
            {
                "month": int(m),
                "frozen_macro_f1": round(_pooled_f1(g, "frozen", labels), 4),
                "cycle_macro_f1": round(_pooled_f1(g, "served", labels), 4),
                "frozen_accuracy": round(float((g["frozen"] == g["label"]).mean()), 4),
                "cycle_accuracy": round(float((g["served"] == g["label"]).mean()), 4),
            }
        )
    alerts = [e for e in events if e["event"] == "drift_alert"]
    promos = [e for e in events if e["event"] == "promoted"]
    retrains = [e for e in events if e["event"] == "retrain"]
    false_alerts = [e for e in alerts if e["day"] < drift_start]
    summary = {
        "drift_start_day": drift_start,
        "first_alert_day": next((e["day"] for e in alerts if e["day"] >= drift_start), None),
        "false_alerts_before_drift": len(false_alerts),
        "monthly": monthly,
        "events": events,
        "sim_wall_seconds": None,
    }
    if summary["first_alert_day"] is not None:
        summary["detection_delay_days"] = summary["first_alert_day"] - drift_start
    promos_after = [e for e in promos if e["day"] >= drift_start]
    if promos_after:
        first = promos_after[0]
        retrain = next(e for e in retrains if e["day"] >= drift_start and e["gate_passed"])
        summary["promotion_day"] = first["day"]
        summary["days_alert_to_production"] = first["day"] - summary["first_alert_day"]
        summary["days_labels_to_production"] = first["days_from_labels_to_production"]
        summary["automated_pipeline_seconds"] = retrain["pipeline_seconds"]
        summary["retrain_train_seconds"] = retrain["train_seconds"]
        after = log[log["day"] > first["day"]]
        summary["after_promotion"] = {
            "frozen_macro_f1": round(_pooled_f1(after, "frozen", labels), 4),
            "cycle_macro_f1": round(_pooled_f1(after, "served", labels), 4),
        }
    pre = log[log["day"] < drift_start]
    summary["before_drift"] = {"frozen_macro_f1": round(_pooled_f1(pre, "frozen", labels), 4)}

    if rollback_drill and promos:
        summary["rollback"] = _rollback_drill(router)
    if gate_drill:
        summary["gate_drill"] = _gate_drill(stream, labels, frozen)
    summary["sim_wall_seconds"] = round(time.time() - wall0, 1)

    _plot(log, labels, summary, out_dir / "macro_f1_by_day.png")
    (out_dir / "simulation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), "utf-8"
    )
    return summary


def _rollback_drill(router) -> dict:
    """Measure 'command -> 100% traffic on the previous version': alias move + download + load + swap."""
    from ticketflow import registry
    from ticketflow.inference import OnnxPredictor

    t0 = time.time()
    frm, to = registry.rollback()
    bundle = registry.download_bundle(registry.PROD, SIM_DIR / "rollback_bundle")
    router.configure(production=OnnxPredictor(bundle, version=to), drop_candidate=True)
    seconds = time.time() - t0
    registry.rollback()  # restore the promoted version afterwards
    return {"from_version": frm, "to_version": to, "seconds": round(seconds, 2)}


def _gate_drill(stream: pd.DataFrame, labels: list[str], production) -> dict:
    """Train candidates on intentionally bad labels in a stable period; the gate must stop them."""
    from ticketflow.pipeline import corrupt_labels, gate_and_register, make_datasets, train_candidate

    ds = make_datasets(80, stream)
    # Random flips are absorbed by the model (it stays good, so passing is correct behaviour);
    # systematic swaps (an annotator mixing two queues) produce a really worse model.
    scenarios = [
        ("clean", 0.0, None),
        ("random_flip_10pct", 0.10, None),
        ("random_flip_40pct", 0.40, None),
        ("swap_delivery_to_shipping", 0.0, ("DELIVERY", "SHIPPING")),
        ("swap_order_to_cancel", 0.0, ("ORDER", "CANCEL")),
        ("swap_account_to_contact", 0.0, ("ACCOUNT", "CONTACT")),
    ]
    results = []
    for name, rate, swap in scenarios:
        bad = ds.train if name == "clean" else corrupt_labels(ds.train, labels, rate, seed=7, swap=swap)
        d2 = type(ds)(bad, ds.golden, ds.reference_texts, ds.upto_day, ds.info)
        cand = train_candidate(
            d2,
            f"gate_drill_{name}",
            out_dir=SIM_DIR / "candidates" / f"gate_drill_{name}",
            tags={"trigger": "gate_drill", "scenario": name},
        )
        gate, version = gate_and_register(cand, ds.golden, production.bundle_dir)
        results.append(
            {
                "scenario": name,
                "intentionally_bad": name != "clean",
                "passed": gate.passed,
                "candidate_macro_f1": round(gate.candidate_macro_f1, 4),
                "production_macro_f1": round(gate.production_macro_f1 or 0, 4),
                "worst_class": gate.worst_class,
                "worst_drop_pp": round(gate.worst_class_drop_pp, 1),
                "reasons": gate.reasons,
            }
        )
    bad = [r for r in results if r["intentionally_bad"]]
    swaps = [r for r in bad if r["scenario"].startswith("swap")]
    return {
        "scenarios": results,
        "blocked_bad_share": round(sum(not r["passed"] for r in bad) / len(bad), 2),
        "blocked_systematic_share": round(sum(not r["passed"] for r in swaps) / len(swaps), 2),
        "clean_passed": all(r["passed"] for r in results if not r["intentionally_bad"]),
    }


def _plot(log: pd.DataFrame, labels: list[str], summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = sorted(log["day"].unique())
    roll = 7
    frozen, cycle, drift = [], [], []
    for d in days:
        w = log[(log["day"] > d - roll) & (log["day"] <= d)]
        frozen.append(_pooled_f1(w, "frozen", labels))
        cycle.append(_pooled_f1(w, "served", labels))
        drift.append(float(log.loc[log["day"] == d, "drift_share"].iloc[0]))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(days, frozen, color="#c0392b", lw=2, label="без цикла (модель на месте)")
    ax.plot(days, cycle, color="#1f6feb", lw=2, label="с циклом (дрейф → переобучение → выкатка)")
    ax.axvline(summary["drift_start_day"], color="grey", ls=":", lw=1.5)
    ax.text(
        summary["drift_start_day"] + 1,
        0.71,
        "сдвиг данных",
        color="grey",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
    )
    for e in summary["events"]:
        if e["event"] == "drift_alert":
            ax.axvline(e["day"], color="#e67e22", ls="--", lw=1.5)
            ax.text(
                e["day"] + 1,
                0.74,
                "алерт дрейфа",
                color="#e67e22",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
            )
        if e["event"] == "promoted":
            ax.axvline(e["day"], color="#2ea043", ls="--", lw=1.5)
            ax.text(
                e["day"] + 1,
                0.77,
                f"v{e['version']} в production",
                color="#2ea043",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
            )
    ax.set_ylabel(f"macro-F1 (скользящее окно {roll} дн.)")
    ax.set_ylim(0.68, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Маршрутизация обращений: качество по дням симуляции")
    for m in range(1, int(log["month"].max()) + 1):
        ax.axvline(m * (len(days) // (log["month"].max() + 1)), color="#ddd", lw=0.8, zorder=0)
    ax2.bar(days, drift, color="#e67e22", width=1.0)
    ax2.set_ylabel("доля признаков\nс дрейфом")
    ax2.set_xlabel("день симуляции")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
