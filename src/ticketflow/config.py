"""Paths, parameters and environment-driven settings shared by every component."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("TICKETFLOW_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
PARAMS_PATH = ROOT / "params.yaml"


@lru_cache
def load_params(path: str | None = None) -> dict[str, Any]:
    with open(path or PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Everything that differs between host and docker comes from env vars."""

    tracking_uri: str
    experiment: str
    model_name: str
    artifact_location: str | None


def settings() -> Settings:
    """Read on every call, so a process (simulator, tests) can redirect MLflow by changing env vars."""
    return Settings(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"),
        experiment=os.environ.get("MLFLOW_EXPERIMENT", "ticket-router"),
        model_name=os.environ.get("MODEL_NAME", "ticket-router"),
        artifact_location=os.environ.get("MLFLOW_ARTIFACT_LOCATION"),
    )


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
