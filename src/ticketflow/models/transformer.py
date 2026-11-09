"""rubert-tiny2 fine-tuning with a hand-written PyTorch loop (no HF Trainer) and ONNX export.

Output of `train_transformer` is a *bundle* directory that the service can run without torch:

    model.onnx         fp32 graph: (input_ids, attention_mask) -> (logits, sentence_embedding)
    model.int8.onnx    dynamic int8 quantization of the same graph
    tokenizer.json     fast tokenizer (HF `tokenizers` format)
    labels.json        class order of the logits
    reference.npz      PCA basis of training embeddings (for drift monitoring)
    vocab.json         training vocabulary (for the OOV drift feature)
    meta.json          max_len, params, metrics
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)


class TicketClassifier(nn.Module):
    """Encoder + linear head on the [CLS] token. Returns logits and the L2-normalised embedding."""

    def __init__(self, model_name: str, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(self.dropout(cls))
        emb = nn.functional.normalize(cls, dim=-1)
        return logits, emb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(tokenizer, texts, labels, batch_size, max_len, shuffle, seed):
    idx = list(range(len(texts)))

    def collate(ids):
        enc = tokenizer(
            [texts[i] for i in ids], padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        y = torch.tensor([labels[i] for i in ids]) if labels is not None else None
        return enc["input_ids"], enc["attention_mask"], y

    g = torch.Generator().manual_seed(seed)
    return DataLoader(idx, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


def train_transformer(
    texts: Sequence[str],
    labels: Sequence[str],
    label_names: Sequence[str],
    out_dir: Path,
    params: dict,
    seed: int = 42,
    log_fn=None,
) -> Path:
    """Fine-tune and write the ONNX bundle to `out_dir`. `log_fn(step_metrics: dict, step: int)` is optional."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = list(texts)
    lab2id = {lab: i for i, lab in enumerate(label_names)}
    y = [lab2id[lab] for lab in labels]

    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    model = TicketClassifier(params["model_name"], len(label_names)).to(device)

    loader = _batches(tokenizer, texts, y, params["batch_size"], params["max_len"], True, seed)
    total_steps = params["epochs"] * len(loader)
    warmup = int(params["warmup_ratio"] * total_steps)

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(groups, lr=params["lr"])

    def lr_lambda(step: int) -> float:  # linear warmup, cosine decay
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    step, t0 = 0, time.time()
    model.train()
    for epoch in range(params["epochs"]):
        running, seen = 0.0, 0
        for input_ids, attn, yb in loader:
            input_ids, attn, yb = input_ids.to(device), attn.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(input_ids, attn)
                loss = loss_fn(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item() * len(yb)
            seen += len(yb)
            step += 1
        if log_fn:
            log_fn({"train_loss": running / max(1, seen), "epoch": epoch}, step)
    train_seconds = time.time() - t0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    (out_dir / "torch").mkdir(exist_ok=True)  # not shipped to the registry, kept for benchmarks
    torch.save(model.state_dict(), out_dir / "torch" / "model.pt")
    export_onnx(model.cpu(), tokenizer, out_dir, params["max_len"])
    quantize_int8(out_dir / "model.onnx", out_dir / "model.int8.onnx")
    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "labels.json").write_text(json.dumps(list(label_names), ensure_ascii=False), "utf-8")
    _write_reference(model, tokenizer, texts, out_dir, params)
    meta = {
        "max_len": params["max_len"],
        "model_name": params["model_name"],
        "train_seconds": train_seconds,
        "n_train": len(texts),
        "device": device,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    return out_dir


@torch.no_grad()
def _write_reference(model, tokenizer, texts, out_dir: Path, params: dict, n_components: int = 5) -> None:
    """PCA basis of training embeddings + training vocabulary: the drift monitor's reference."""
    embs = []
    for input_ids, attn, _ in _batches(tokenizer, texts, None, 128, params["max_len"], False, 0):
        embs.append(model(input_ids, attn)[1].numpy())
    e = np.concatenate(embs)
    mean = e.mean(0)
    _, _, vt = np.linalg.svd(e - mean, full_matrices=False)
    np.savez(
        out_dir / "reference.npz",
        mean=mean.astype(np.float32),
        components=vt[:n_components].astype(np.float32),
    )
    vocab = Counter(w.lower() for t in texts for w in WORD_RE.findall(t))
    (out_dir / "vocab.json").write_text(json.dumps(sorted(vocab), ensure_ascii=False), "utf-8")


def export_onnx(model: nn.Module, tokenizer, out_dir: Path, max_len: int) -> Path:
    enc = tokenizer(
        ["пример обращения", "второй пример"], padding="max_length", max_length=16, return_tensors="pt"
    )
    path = out_dir / "model.onnx"
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=max_len)
    try:
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"]),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "sentence_embedding"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
            dynamo=True,
            external_data=False,
        )
    except Exception as exc:  # pragma: no cover - fallback for older torch / exporter issues
        print(f"[onnx] dynamo export failed ({exc!r}); falling back to TorchScript exporter")
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"]),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The dynamo exporter writes value_info that ORT's shape inference re-derives differently;
    # dropping it lets the quantizer infer shapes from scratch.
    m = onnx.load(str(src))
    m.graph.ClearField("value_info")
    clean = src.with_name("model.noinfo.onnx")
    onnx.save(m, str(clean))
    try:
        quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8)
    finally:
        clean.unlink(missing_ok=True)
    return dst


def load_torch_model(bundle_dir: Path) -> tuple[TicketClassifier, object]:
    """Reload the PyTorch weights kept next to the bundle (used by the latency benchmark only)."""
    meta = json.loads((Path(bundle_dir) / "meta.json").read_text("utf-8"))
    labels = json.loads((Path(bundle_dir) / "labels.json").read_text("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
    model = TicketClassifier(meta["model_name"], len(labels))
    model.load_state_dict(torch.load(Path(bundle_dir) / "torch" / "model.pt", map_location="cpu"))
    return model.eval(), tokenizer
