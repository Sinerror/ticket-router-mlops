"""Online drift check over the service's prediction log (Airflow `drift_monitor` DAG)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ticketflow import registry
from ticketflow.config import ARTIFACTS_DIR, REPORTS_DIR, load_params
from ticketflow.drift import DriftResult, check_drift


def read_log(path: Path, version: str, last_n: int) -> pd.DataFrame:
    """Tail of the prediction log for the current production version."""
    rows = []
    if not Path(path).exists():  # no traffic yet
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("production_version") == version:
                rows.append(rec)
    return pd.DataFrame(rows[-last_n:])


def run_drift_check(log_path: str | Path, window: int | None = None, min_rows: int = 100) -> dict:
    p = load_params()
    window = window or p["drift"]["window_days"] * p["data"]["tickets_per_day"]
    registry.setup()
    version = registry.get_alias_version(registry.PROD)
    bundle = registry.download_bundle(version, ARTIFACTS_DIR / "monitor" / f"v{version}")
    reference = pd.read_parquet(bundle / "reference_features.parquet")
    current = read_log(Path(log_path), version, window)
    if len(current) < min_rows:
        return {"alert": False, "skipped": f"only {len(current)} rows for v{version}", "version": version}
    html = REPORTS_DIR / "drift" / f"drift_v{version}_{int(time.time())}.html"
    res: DriftResult = check_drift(reference, current, p["drift"], html_path=html)
    out = {**res.to_dict(), "version": version, "report": str(html)}

    # Persistence: alert only if the last N checks of this production version all fired.
    history_path = Path(log_path).with_name("drift_checks.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "version": version,
                    "raw_alert": res.alert,
                    "drift_share": res.drift_share,
                }
            )
            + "\n"
        )
    lines = [json.loads(x) for x in history_path.read_text("utf-8").splitlines() if x.strip()]
    n = p["drift"].get("persistence_checks", 1)
    last = [x for x in lines if x["version"] == version][-n:]
    out["raw_alert"] = res.alert
    out["alert"] = len(last) == n and all(x["raw_alert"] for x in last)
    return out
