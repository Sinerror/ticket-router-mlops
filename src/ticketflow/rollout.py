"""Client for the running service: rollout flags, reloads and online metrics (used by Airflow tasks)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class ServiceClient:
    url: str = os.environ.get("SERVICE_URL", "http://localhost:8000")
    token: str | None = os.environ.get("ADMIN_TOKEN")
    timeout: float = 30.0

    def _post(self, path: str, json: dict | None = None) -> dict:
        r = requests.post(
            self.url + path,
            json=json,
            timeout=self.timeout,
            headers={"X-Admin-Token": self.token} if self.token else {},
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return requests.get(self.url + "/health", timeout=self.timeout).json()

    def reload(self) -> dict:
        return self._post("/admin/reload")

    def set_rollout(self, mode: str, canary_fraction: float | None = None) -> dict:
        return self._post("/admin/rollout", {"mode": mode, "canary_fraction": canary_fraction})

    def report_drift(self, drift_share: float, alert: bool) -> dict:
        return self._post("/admin/drift", {"drift_share": drift_share, "alert": alert})

    def counters(self) -> dict[tuple[str, tuple], float]:
        text = requests.get(self.url + "/metrics", timeout=self.timeout).text
        out: dict[tuple[str, tuple], float] = {}
        for fam in text_string_to_metric_families(text):
            for s in fam.samples:
                if s.name.endswith("_total"):
                    out[(s.name, tuple(sorted(s.labels.items())))] = s.value
        return out


def delta(before: dict, after: dict, name: str, **match) -> float:
    tot = 0.0
    for (n, labels), v in after.items():
        d = dict(labels)
        if n == name and all(d.get(k) == str(val) for k, val in match.items()):
            tot += v - before.get((n, labels), 0.0)
    return tot


def shadow_disagreement(before: dict, after: dict, candidate_version: str) -> tuple[float, int]:
    agree = delta(
        before, after, "ticket_shadow_comparisons_total", candidate_version=candidate_version, agree="true"
    )
    dis = delta(
        before, after, "ticket_shadow_comparisons_total", candidate_version=candidate_version, agree="false"
    )
    n = int(agree + dis)
    return (dis / n if n else 0.0), n


def arm_accuracy(before: dict, after: dict, served_by: str) -> tuple[float, int]:
    ok = delta(before, after, "ticket_feedback_total", served_by=served_by, correct="true")
    bad = delta(before, after, "ticket_feedback_total", served_by=served_by, correct="false")
    n = int(ok + bad)
    return (ok / n if n else 0.0), n


def wait_for_traffic(client: ServiceClient, seconds: float) -> tuple[dict, dict]:
    before = client.counters()
    time.sleep(seconds)
    return before, client.counters()
