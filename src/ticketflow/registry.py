"""MLflow tracking + Model Registry helpers.

Lifecycle is expressed with registry *aliases* (MLflow >= 2.9; stages are deprecated):

    challenger  -> passed the quality gate, running in shadow / canary
    production  -> serves 100% of traffic
    previous    -> the last production version, target of a one-command rollback
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ticketflow.config import ROOT, settings

PROD, CHALLENGER, PREVIOUS = "production", "challenger", "previous"
BUNDLE_FILES = (
    "model.onnx",
    "model.int8.onnx",
    "tokenizer.json",
    "labels.json",
    "reference.npz",
    "vocab.json",
    "meta.json",
    "reference_features.parquet",
)


def setup(experiment: str | None = None) -> MlflowClient:
    s = settings()
    mlflow.set_tracking_uri(s.tracking_uri)
    name = experiment or s.experiment
    if mlflow.get_experiment_by_name(name) is None:
        location = s.artifact_location
        if location is None and not s.tracking_uri.startswith("http"):
            # local sqlite store: keep artifacts next to the repo; a remote server proxies its own store
            location = (ROOT / "mlartifacts").as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return MlflowClient()


def log_and_register(bundle_dir: Path, model_name: str | None = None, tags: dict | None = None) -> str:
    """Log the bundle as a pyfunc model inside the *active* run and register a new version."""
    # Imported here: mlflow.pyfunc pulls pandas, which the slim serving image does not have.
    from ticketflow.pyfunc_model import TicketRouterModel

    model_name = model_name or settings().model_name
    with tempfile.TemporaryDirectory() as tmp:
        ship = Path(tmp) / "bundle"
        ship.mkdir()
        for f in BUNDLE_FILES:
            if (Path(bundle_dir) / f).exists():
                shutil.copy2(Path(bundle_dir) / f, ship / f)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TicketRouterModel(),
            artifacts={"bundle": str(ship)},
            pip_requirements=["onnxruntime", "tokenizers", "numpy"],
            registered_model_name=model_name,
        )
    version = str(info.registered_model_version)
    client = MlflowClient()
    for k, v in (tags or {}).items():
        client.set_model_version_tag(model_name, version, k, str(v))
    return version


def get_alias_version(alias: str, model_name: str | None = None) -> str | None:
    try:
        return MlflowClient().get_model_version_by_alias(model_name or settings().model_name, alias).version
    except MlflowException:
        return None


def set_alias(alias: str, version: str, model_name: str | None = None) -> None:
    MlflowClient().set_registered_model_alias(model_name or settings().model_name, alias, version)


def promote(version: str, model_name: str | None = None) -> None:
    """challenger -> production; old production -> previous."""
    model_name = model_name or settings().model_name
    client = MlflowClient()
    current = get_alias_version(PROD, model_name)
    if current and current != version:
        client.set_registered_model_alias(model_name, PREVIOUS, current)
    client.set_registered_model_alias(model_name, PROD, version)
    try:
        client.delete_registered_model_alias(model_name, CHALLENGER)
    except MlflowException:
        pass


def drop_challenger(model_name: str | None = None) -> None:
    try:
        MlflowClient().delete_registered_model_alias(model_name or settings().model_name, CHALLENGER)
    except MlflowException:
        pass


def rollback(model_name: str | None = None) -> tuple[str, str]:
    """production <- previous. Returns (from_version, to_version)."""
    model_name = model_name or settings().model_name
    prev = get_alias_version(PREVIOUS, model_name)
    cur = get_alias_version(PROD, model_name)
    if prev is None:
        raise RuntimeError("no 'previous' version to roll back to")
    client = MlflowClient()
    client.set_registered_model_alias(model_name, PROD, prev)
    if cur:
        client.set_registered_model_alias(model_name, PREVIOUS, cur)
    return cur or "", prev


def download_bundle(alias_or_version: str, dst: Path, model_name: str | None = None) -> Path:
    """Fetch the ONNX bundle of `models:/name@alias` (or `/version`) into dst; returns the bundle dir."""
    model_name = model_name or settings().model_name
    ref = f"@{alias_or_version}" if not alias_or_version.isdigit() else f"/{alias_or_version}"
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}{ref}", dst_path=str(dst))
    hits = list(Path(local).rglob("model.int8.onnx"))
    if not hits:
        raise FileNotFoundError(f"bundle not found in {local}")
    return hits[0].parent
