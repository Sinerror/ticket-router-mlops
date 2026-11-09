"""Torch-free inference over an ONNX bundle. Used by the FastAPI service and by the simulator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: np.ndarray
    projection: np.ndarray  # PCA projection of the embedding (drift features)


class OnnxPredictor:
    def __init__(
        self, bundle_dir: str | Path, variant: str = "int8", threads: int = 1, version: str = "local"
    ):
        self.bundle_dir = Path(bundle_dir)
        self.version = version
        self.variant = variant
        meta = json.loads((self.bundle_dir / "meta.json").read_text("utf-8"))
        self.labels: list[str] = json.loads((self.bundle_dir / "labels.json").read_text("utf-8"))
        self.max_len = int(meta["max_len"])
        self.tokenizer = Tokenizer.from_file(str(self.bundle_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.max_len)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]") or 0)
        ref = np.load(self.bundle_dir / "reference.npz")
        self.ref_mean, self.ref_components = ref["mean"], ref["components"]
        self.vocab = set(json.loads((self.bundle_dir / "vocab.json").read_text("utf-8")))
        name = "model.int8.onnx" if variant == "int8" else "model.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.bundle_dir / name), so, providers=["CPUExecutionProvider"]
        )

    def _encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self.tokenizer.encode_batch(texts)
        return {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }

    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            logits, emb = self.session.run(["logits", "sentence_embedding"], self._encode(chunk))
            logits = logits - logits.max(-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(-1, keepdims=True)
            proj = (emb - self.ref_mean) @ self.ref_components.T
            for p, z in zip(probs, proj, strict=True):
                k = int(p.argmax())
                out.append(Prediction(self.labels[k], float(p[k]), p, z))
        return out

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def oov_share(self, text: str) -> float:
        words = [w.lower() for w in WORD_RE.findall(text)]
        return sum(w not in self.vocab for w in words) / len(words) if words else 0.0


def features_row(text: str, pred: Prediction, predictor: OnnxPredictor) -> dict:
    """Label-free per-ticket features for drift monitoring (kept here so the service needs no pandas)."""
    row = {
        "text_len": len(text),
        "n_words": len(text.split()),
        "oov_share": predictor.oov_share(text),
        "confidence": pred.confidence,
        "pred": pred.label,
    }
    row.update({f"emb_{i}": float(v) for i, v in enumerate(pred.projection)})
    return row
