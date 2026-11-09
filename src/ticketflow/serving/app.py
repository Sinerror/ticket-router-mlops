"""FastAPI inference service: ONNX Runtime on CPU, production + candidate (shadow / canary), Prometheus metrics.

Model source (env):
    MODEL_SOURCE=registry   load `models:/<MODEL_NAME>@production` and `@challenger` from MLflow (default)
    MODEL_SOURCE=local      load MODEL_DIR (and optional CANDIDATE_DIR) from disk (tests, offline demo)
    ROLLOUT_MODE=off|shadow|canary, CANARY_FRACTION=0.2
    ALIAS_POLL_SECONDS=10   re-resolve aliases periodically: promotion/rollback reach traffic without restart
    PREDICTION_LOG=/data/predictions.jsonl  per-request features for the drift monitor
    ADMIN_TOKEN             required in X-Admin-Token for /admin/* when set
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ticketflow.inference import OnnxPredictor
from ticketflow.router import Router

REQUESTS = Counter("ticket_predictions_total", "Predictions served", ["model_version", "served_by", "label"])
LATENCY = Histogram(
    "ticket_inference_seconds",
    "Model inference latency",
    ["model_version"],
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.1, 0.25),
)
CONFIDENCE = Histogram(
    "ticket_prediction_confidence",
    "Max softmax probability",
    ["model_version"],
    buckets=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
SHADOW = Counter(
    "ticket_shadow_comparisons_total", "Shadow/canary comparisons", ["candidate_version", "agree"]
)
FEEDBACK = Counter(
    "ticket_feedback_total", "Operator labels received", ["model_version", "served_by", "correct"]
)
MODEL_INFO = Gauge("ticket_model_info", "Loaded models", ["role", "version"])
ROLLOUT = Gauge("ticket_canary_fraction", "Share of traffic routed to the candidate")
DRIFT_SHARE = Gauge("ticket_drift_share", "Share of drifted features (last drift check)")
DRIFT_ALERT = Gauge("ticket_drift_alert", "1 if the last drift check raised an alert")
RELOADS = Counter("ticket_model_reloads_total", "Model reloads", ["reason"])


class State:
    router: Router | None = None
    versions: dict[str, str | None] = {"production": None, "challenger": None}
    recent: OrderedDict[str, tuple[str, str, str]] = OrderedDict()  # ticket_id -> (label, version, served_by)
    log_lock = threading.Lock()


S = State()
VARIANT = os.environ.get("MODEL_VARIANT", "int8")
THREADS = int(os.environ.get("ORT_THREADS", "1"))
CACHE = Path(os.environ.get("MODEL_CACHE", "/tmp/ticketflow-models"))


def _load_registry_version(alias: str) -> tuple[OnnxPredictor | None, str | None]:
    from ticketflow import registry

    version = registry.get_alias_version(alias)
    if version is None:
        return None, None
    if version == S.versions.get(alias) and S.router is not None:
        current = S.router.production if alias == "production" else S.router.candidate
        if current is not None:
            return current, version
    bundle = registry.download_bundle(version, CACHE / f"v{version}")
    return OnnxPredictor(bundle, variant=VARIANT, threads=THREADS, version=version), version


def load_models(reason: str = "startup") -> dict:
    mode = os.environ.get("ROLLOUT_MODE", "shadow")
    frac = float(os.environ.get("CANARY_FRACTION", "0.2"))
    if os.environ.get("MODEL_SOURCE", "registry") == "local":
        prod = OnnxPredictor(
            os.environ["MODEL_DIR"],
            variant=VARIANT,
            threads=THREADS,
            version=os.environ.get("MODEL_VERSION", "local"),
        )
        cand_dir = os.environ.get("CANDIDATE_DIR")
        cand = (
            OnnxPredictor(cand_dir, variant=VARIANT, threads=THREADS, version="candidate")
            if cand_dir
            else None
        )
        pv, cv = prod.version, cand.version if cand else None
    else:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        prod, pv = _load_registry_version("production")
        if prod is None:
            raise RuntimeError("no model with alias 'production' in the registry")
        cand, cv = _load_registry_version("challenger")
        if cv == pv:
            cand, cv = None, None
    changed = (pv, cv) != (S.versions["production"], S.versions["challenger"])
    if S.router is None:
        S.router = Router(prod, cand, mode if cand else "off", frac)
    elif changed:
        S.router.configure(
            production=prod,
            candidate=cand,
            drop_candidate=cand is None,
            mode=(S.router.mode if S.router.mode != "off" else mode) if cand else "off",
        )
    if changed or reason == "startup":
        RELOADS.labels(reason).inc()
        MODEL_INFO.clear()
        MODEL_INFO.labels("production", pv).set(1)
        if cv:
            MODEL_INFO.labels("challenger", cv).set(1)
    S.versions = {"production": pv, "challenger": cv}
    ROLLOUT.set(S.router.canary_fraction if S.router.mode == "canary" else 0)
    return {"production": pv, "challenger": cv, "mode": S.router.mode, "changed": changed}


def _poller(interval: float) -> None:
    while True:
        time.sleep(interval)
        try:
            load_models("alias_poll")
        except Exception as exc:  # keep serving the last good model
            print(f"[poll] reload failed: {exc!r}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models("startup")
    interval = float(os.environ.get("ALIAS_POLL_SECONDS", "0"))
    if interval > 0 and os.environ.get("MODEL_SOURCE", "registry") == "registry":
        threading.Thread(target=_poller, args=(interval,), daemon=True).start()
    yield


app = FastAPI(title="Ticket router", version="0.1.0", lifespan=lifespan)


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    ticket_id: str | None = None


class Feedback(BaseModel):
    ticket_id: str
    label: str


class Rollout(BaseModel):
    mode: str
    canary_fraction: float | None = Field(default=None, ge=0, le=1)


class DriftReport(BaseModel):
    drift_share: float
    alert: bool


def _check_admin(token: str | None) -> None:
    expected = os.environ.get("ADMIN_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="bad admin token")


def _log(record: dict) -> None:
    path = os.environ.get("PREDICTION_LOG")
    if not path:
        return
    with S.log_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/predict")
def predict(t: Ticket) -> dict:
    from ticketflow.inference import features_row

    key = t.ticket_id or str(time.time_ns())
    t0 = time.perf_counter()
    r = S.router.route(key, t.text)
    LATENCY.labels(r.served_version).observe(time.perf_counter() - t0)
    REQUESTS.labels(r.served_version, r.served_by, r.served.label).inc()
    CONFIDENCE.labels(r.served_version).observe(r.served.confidence)
    if r.candidate is not None:
        SHADOW.labels(S.versions["challenger"], str(r.candidate.label == r.production.label).lower()).inc()
    S.recent[key] = (r.served.label, r.served_version, r.served_by)
    if len(S.recent) > 50_000:
        S.recent.popitem(last=False)
    feats = features_row(t.text, r.production, S.router.production)
    _log(
        {
            "ts": time.time(),
            "ticket_id": key,
            "production_version": S.versions["production"],
            "served_by": r.served_by,
            "served": r.served.label,
            "candidate": r.candidate.label if r.candidate else None,
            **feats,
        }
    )
    return {
        "ticket_id": key,
        "label": r.served.label,
        "confidence": round(r.served.confidence, 4),
        "model_version": r.served_version,
        "served_by": r.served_by,
    }


@app.post("/feedback")
def feedback(f: Feedback) -> dict:
    """Operator's final routing decision: turns into online accuracy per model version in Grafana."""
    hit = S.recent.get(f.ticket_id)
    if hit is None:
        raise HTTPException(status_code=404, detail="unknown ticket_id")
    label, version, served_by = hit
    FEEDBACK.labels(version, served_by, str(label == f.label).lower()).inc()
    return {"correct": label == f.label}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **S.versions, "mode": S.router.mode if S.router else None}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload")
def admin_reload(x_admin_token: str | None = Header(default=None)) -> dict:
    _check_admin(x_admin_token)
    t0 = time.perf_counter()
    info = load_models("admin")
    return {**info, "seconds": round(time.perf_counter() - t0, 3)}


@app.post("/admin/rollout")
def admin_rollout(r: Rollout, x_admin_token: str | None = Header(default=None)) -> dict:
    _check_admin(x_admin_token)
    if S.router.candidate is None and r.mode != "off":
        raise HTTPException(status_code=409, detail="no candidate loaded")
    S.router.configure(mode=r.mode, canary_fraction=r.canary_fraction)
    ROLLOUT.set(S.router.canary_fraction if S.router.mode == "canary" else 0)
    return {"mode": S.router.mode, "canary_fraction": S.router.canary_fraction}


@app.post("/admin/drift")
def admin_drift(d: DriftReport, x_admin_token: str | None = Header(default=None)) -> dict:
    _check_admin(x_admin_token)
    DRIFT_SHARE.set(d.drift_share)
    DRIFT_ALERT.set(int(d.alert))
    return {"ok": True}
