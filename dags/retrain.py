"""Retraining & safe rollout of the ticket router.

    fetch_labels -> train -> quality_gate -> shadow -> canary -> promote -> release (always)

Every task calls the same functions as `make simulate` (ticketflow.pipeline / registry / rollout).
Shadow and canary phases read online counters of the running service (/metrics).
"""

from __future__ import annotations

from datetime import datetime

from airflow.sdk import Param, dag, get_current_context, task

ARTIFACTS = "/opt/ticketflow/artifacts"
LOCK = "/opt/ticketflow/data/state/retrain.lock"  # created by drift_monitor, removed by `release`
COOLDOWN = "/opt/ticketflow/data/state/retrain.cooldown"  # written by `release`: no new retrain until then


@dag(
    dag_id="retrain",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ticketflow", "training"],
    params={
        "upto_day": Param(
            120, type="integer", description="labelled data available up to this simulated day"
        ),
        "trigger": Param("manual", type="string"),
        "shadow_seconds": Param(60, type="integer"),
        "canary_seconds": Param(90, type="integer"),
        "canary_fraction": Param(0.2, type="number"),
    },
    doc_md=__doc__,
)
def retrain():
    @task
    def fetch_labels() -> dict:
        """'Pick up new labelling': labelled tickets up to `upto_day` (DVC-versioned stream)."""
        from ticketflow.pipeline import make_datasets

        day = int(get_current_context()["params"]["upto_day"])
        ds = make_datasets(day)
        return {"upto_day": day, **ds.info}

    @task
    def train(info: dict) -> dict:
        from ticketflow.pipeline import make_datasets, train_candidate

        ctx = get_current_context()
        ds = make_datasets(info["upto_day"])
        name = f"airflow_day{info['upto_day']}_{ctx['run_id'][-8:]}"
        cand = train_candidate(
            ds,
            name,
            out_dir=f"{ARTIFACTS}/candidates/{name}",
            tags={"trigger": ctx["params"]["trigger"], "airflow_run_id": ctx["run_id"]},
        )
        return {
            "run_id": cand.run_id,
            "bundle_dir": str(cand.bundle_dir),
            "macro_f1": cand.metrics["macro_f1"],
            "baseline_macro_f1": cand.baseline_metrics["macro_f1"],
            "upto_day": info["upto_day"],
        }

    # ignore_downstream_trigger_rules=False: a blocked gate skips shadow/canary/promote,
    # but `release` (trigger_rule=all_done) must still run and free the lock.
    @task.short_circuit(ignore_downstream_trigger_rules=False)
    def quality_gate(cand: dict) -> bool:
        """Score production on the same golden set; register the candidate (alias=challenger) if it passes."""
        import json
        from pathlib import Path

        import pandas as pd

        from ticketflow import registry
        from ticketflow.pipeline import Candidate, gate_and_register

        bundle = Path(cand["bundle_dir"])
        golden = pd.read_parquet(bundle / "golden.parquet")
        metrics = json.loads((bundle / "golden_metrics.json").read_text("utf-8"))
        registry.setup()
        prod = registry.download_bundle(registry.PROD, Path(ARTIFACTS) / "airflow_production")
        c = Candidate(cand["run_id"], bundle, metrics, {}, 0.0)
        gate, version = gate_and_register(c, golden, prod)
        print(f"gate: {gate.to_dict()} -> version {version}")
        return gate.passed

    @task
    def shadow() -> dict:
        from ticketflow.config import load_params
        from ticketflow.rollout import ServiceClient, shadow_disagreement, wait_for_traffic

        p = get_current_context()["params"]
        client = ServiceClient()
        info = client.reload()  # picks up the new challenger alias
        client.set_rollout("shadow")
        before, after = wait_for_traffic(client, p["shadow_seconds"])
        dis, n = shadow_disagreement(before, after, info["challenger"])
        limit = load_params()["rollout"]["max_shadow_disagreement"]
        if n == 0 or dis > limit:
            client.set_rollout("off")
            raise RuntimeError(f"shadow failed: disagreement {dis:.3f} on {n} tickets (limit {limit})")
        return {"challenger": info["challenger"], "disagreement": dis, "n": n}

    @task
    def canary(sh: dict) -> dict:
        from ticketflow.config import load_params
        from ticketflow.rollout import ServiceClient, arm_accuracy, wait_for_traffic

        p = get_current_context()["params"]
        client = ServiceClient()
        client.set_rollout("canary", p["canary_fraction"])
        before, after = wait_for_traffic(client, p["canary_seconds"])
        acc_c, n_c = arm_accuracy(before, after, "canary")
        acc_p, n_p = arm_accuracy(before, after, "production")
        limit = load_params()["rollout"]["max_canary_drop_pp"]
        if n_c == 0 or (acc_p - acc_c) * 100 > limit:
            client.set_rollout("off")
            raise RuntimeError(f"canary failed: {acc_c:.3f} vs production {acc_p:.3f} ({n_c}/{n_p} labels)")
        return {**sh, "canary_accuracy": acc_c, "production_accuracy": acc_p, "n_canary": n_c}

    @task
    def promote(res: dict) -> dict:
        from ticketflow import registry
        from ticketflow.rollout import ServiceClient

        registry.setup()
        registry.promote(res["challenger"])
        info = ServiceClient().reload()
        return {**res, "production": info["production"]}

    @task(trigger_rule="all_done")
    def release() -> None:
        """Always runs: drop a non-promoted challenger, back to production-only, free the lock, start a cooldown."""
        import os

        from ticketflow import registry
        from ticketflow.rollout import ServiceClient

        registry.setup()
        registry.drop_challenger()
        ServiceClient().reload()
        import json
        import time

        cooldown_s = float(os.environ.get("RETRAIN_COOLDOWN_SECONDS", "180"))
        with open(COOLDOWN, "w", encoding="utf-8") as f:
            json.dump({"until": time.time() + cooldown_s}, f)
        if os.path.exists(LOCK):
            os.remove(LOCK)

    cand = train(fetch_labels())
    sh = shadow()
    done = promote(canary(sh))
    quality_gate(cand) >> sh  # shadow starts only after the candidate passed the gate
    done >> release()


retrain()
