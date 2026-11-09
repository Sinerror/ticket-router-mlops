"""Drift monitor: every minute (~6 simulated days at 10 s per day in the demo) run an Evidently check over the
service prediction log; on alert, trigger the `retrain` DAG with the current simulated day."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task

LOG = "/opt/ticketflow/data/state/predictions.jsonl"
STATE = "/opt/ticketflow/data/state/current_day.json"
LOCK = "/opt/ticketflow/data/state/retrain.lock"
COOLDOWN = "/opt/ticketflow/data/state/retrain.cooldown"
LOCK_TTL_SECONDS = 30 * 60  # a lock older than this is stale (e.g. the scheduler was restarted mid-run)


@dag(
    dag_id="drift_monitor",
    schedule=timedelta(minutes=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    # conf is templated from XCom: render it as native Python (int), not a string, so the
    # retrain DAG's typed `upto_day` param validates.
    render_template_as_native_obj=True,
    tags=["ticketflow", "monitoring"],
    doc_md=__doc__,
)
def drift_monitor():
    @task
    def check() -> dict:
        from ticketflow.monitor import run_drift_check
        from ticketflow.rollout import ServiceClient

        res = run_drift_check(LOG)
        if "drift_share" in res:
            ServiceClient().report_drift(res["drift_share"], res["alert"])
        return res

    @task.short_circuit
    def should_retrain(res: dict) -> bool:
        """Alert, and no retraining in flight: no fresh lock file (set below, removed by retrain.release),
        no cooldown after the previous attempt, and no challenger loaded in the service."""
        import os
        import time

        from ticketflow.rollout import ServiceClient

        if not res.get("alert"):
            return False
        if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < LOCK_TTL_SECONDS:
            return False
        if os.path.exists(COOLDOWN):
            with open(COOLDOWN, encoding="utf-8") as f:
                if time.time() < json.load(f)["until"]:
                    return False
        return ServiceClient().health().get("challenger") is None

    @task
    def current_day() -> int:
        with open(STATE, encoding="utf-8") as f:
            day = int(json.load(f)["day"])
        with open(LOCK, "w", encoding="utf-8") as f:  # one retrain per alert
            json.dump({"requested_at_day": day}, f)
        return day

    trigger = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain",
        conf={"upto_day": "{{ ti.xcom_pull(task_ids='current_day') }}", "trigger": "drift_alert"},
        wait_for_completion=False,
    )

    res = check()
    should_retrain(res) >> current_day() >> trigger


drift_monitor()
