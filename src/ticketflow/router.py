"""Traffic routing between the production model and a candidate (shadow / canary).

Pure Python, no web framework: the FastAPI service and the simulator use the same class, so the
logic tested in simulation is exactly the logic that serves requests.

    mode = "off"     production answers everything
    mode = "shadow"  production answers, candidate is evaluated silently (logged, never returned)
    mode = "canary"  a deterministic `canary_fraction` of tickets is answered by the candidate
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from ticketflow.inference import OnnxPredictor, Prediction

MODES = ("off", "shadow", "canary")


def in_canary(key: str, fraction: float) -> bool:
    """Sticky assignment: the same ticket/customer always lands in the same arm."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < fraction * 10_000


@dataclass
class RoutedPrediction:
    served: Prediction
    served_by: str  # "production" | "canary"
    served_version: str
    production: Prediction
    candidate: Prediction | None


class Router:
    def __init__(
        self,
        production: OnnxPredictor,
        candidate: OnnxPredictor | None = None,
        mode: str = "off",
        canary_fraction: float = 0.0,
    ):
        self._lock = threading.Lock()
        self.production = production
        self.candidate = candidate
        self.mode = mode
        self.canary_fraction = canary_fraction

    def configure(
        self,
        *,
        production: OnnxPredictor | None = None,
        candidate: OnnxPredictor | None = None,
        mode: str | None = None,
        canary_fraction: float | None = None,
        drop_candidate: bool = False,
    ):
        """Atomic swap of models/flags (used by promotion, rollback and the admin API)."""
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            if production is not None:
                self.production = production
            if candidate is not None:
                self.candidate = candidate
            if drop_candidate:
                self.candidate = None
            if mode is not None:
                self.mode = mode
            if canary_fraction is not None:
                self.canary_fraction = canary_fraction
            if self.candidate is None:
                self.mode = "off"

    def route_batch(self, keys: list[str], texts: list[str]) -> list[RoutedPrediction]:
        with self._lock:
            prod, cand, mode, frac = self.production, self.candidate, self.mode, self.canary_fraction
        prod_preds = prod.predict_batch(texts)
        cand_preds = (
            cand.predict_batch(texts) if (cand is not None and mode != "off") else [None] * len(texts)
        )
        out = []
        for key, pp, cp in zip(keys, prod_preds, cand_preds, strict=True):
            if mode == "canary" and cp is not None and in_canary(key, frac):
                out.append(RoutedPrediction(cp, "canary", cand.version, pp, cp))
            else:
                out.append(RoutedPrediction(pp, "production", prod.version, pp, cp))
        return out

    def route(self, key: str, text: str) -> RoutedPrediction:
        return self.route_batch([key], [text])[0]
