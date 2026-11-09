"""CPU inference benchmark: PyTorch fp32 vs ONNX Runtime fp32 vs ONNX Runtime int8 (single request, 1 thread).

Latency is measured per request (batch=1), like a real API call; quality is macro-F1 on the given texts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ticketflow.config import REPORTS_DIR


def _percentiles(lat: list[float]) -> dict:
    a = np.array(lat) * 1000
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2)}


def run(bundle_dir: Path, eval_df: pd.DataFrame, n_latency: int = 300, threads: int = 1) -> list[dict]:
    import torch

    from ticketflow.inference import OnnxPredictor
    from ticketflow.metrics import evaluate
    from ticketflow.models.transformer import load_torch_model

    labels = json.loads((Path(bundle_dir) / "labels.json").read_text("utf-8"))
    texts, y = list(eval_df["text"]), list(eval_df["label"])
    sample = texts[:n_latency]
    rows = []

    torch.set_num_threads(threads)
    model, tok = load_torch_model(bundle_dir)
    max_len = json.loads((Path(bundle_dir) / "meta.json").read_text("utf-8"))["max_len"]

    @torch.no_grad()
    def torch_predict(batch: list[str]) -> list[str]:
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        logits, _ = model(enc["input_ids"], enc["attention_mask"])
        return [labels[i] for i in logits.argmax(-1).tolist()]

    for t in sample[:20]:
        torch_predict([t])  # warm-up
    lat = []
    for t in sample:
        t0 = time.perf_counter()
        torch_predict([t])
        lat.append(time.perf_counter() - t0)
    preds = [p for i in range(0, len(texts), 64) for p in torch_predict(texts[i : i + 64])]
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    rows.append(
        {
            "runtime": "PyTorch fp32",
            **_percentiles(lat),
            "macro_f1": round(evaluate(y, preds, labels)["macro_f1"], 4),
            "size_mb": round(size, 1),
        }
    )

    for variant, name, fname in (
        ("fp32", "ONNX Runtime fp32", "model.onnx"),
        ("int8", "ONNX Runtime int8", "model.int8.onnx"),
    ):
        pr = OnnxPredictor(bundle_dir, variant=variant, threads=threads)
        for t in sample[:20]:
            pr.predict(t)
        lat = []
        for t in sample:
            t0 = time.perf_counter()
            pr.predict(t)
            lat.append(time.perf_counter() - t0)
        preds = [p.label for p in pr.predict_batch(texts)]
        rows.append(
            {
                "runtime": name,
                **_percentiles(lat),
                "macro_f1": round(evaluate(y, preds, labels)["macro_f1"], 4),
                "size_mb": round((Path(bundle_dir) / fname).stat().st_size / 2**20, 1),
            }
        )

    base = rows[0]["p50_ms"]
    for r in rows:
        r["speedup_p50"] = round(base / r["p50_ms"], 2)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    return rows
