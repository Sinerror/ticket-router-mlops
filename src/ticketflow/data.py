"""Data preparation: download the public dataset and turn it into a 'history + 6 months of traffic' stream.

The dataset (Bitext customer-support corpus translated to Russian) has 11 topic categories and
linguistic-variation flags (Q colloquial, Z typos, C coloquial, W offensive, ...). We use them to build a
controlled, realistic shift without synthetic text:

* history and months < drift_month: the surge class is almost absent, phrasing is mostly clean;
* from drift_month on (with a one-week ramp): the surge class jumps to ~25% of traffic and
  most tickets are written colloquially / with typos.

Every unique text is used at most once, so there is no train/test leakage between days.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ticketflow.config import PROCESSED_DIR, RAW_DIR, ensure_dirs, load_params

RAW_FILE = RAW_DIR / "support_ru.csv"
HISTORY_FILE = PROCESSED_DIR / "history.parquet"
STREAM_FILE = PROCESSED_DIR / "stream.parquet"
LABELS_FILE = PROCESSED_DIR / "labels.json"

_PLACEHOLDER = re.compile(r"\{\{([^}]*)\}\}")
_NUMBER_HINTS = ("number", "номер", "id", "invoice", "order", "счет", "счёт", "заказ")


def download_raw(force: bool = False) -> Path:
    """Fetch the raw CSV from the Hugging Face Hub into data/raw (tracked by DVC, not git)."""
    from huggingface_hub import hf_hub_download

    ensure_dirs()
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    p = load_params()["data"]
    src = hf_hub_download(p["hf_dataset"], p["hf_file"], repo_type="dataset")
    RAW_FILE.write_bytes(Path(src).read_bytes())
    return RAW_FILE


def _fill_placeholders(text: str, rng: np.random.Generator) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1).lower()
        if any(h in name for h in _NUMBER_HINTS):
            return str(rng.integers(100000, 999999))
        return ""

    text = _PLACEHOLDER.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def clean(raw: pd.DataFrame, noisy_flags: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = raw.rename(columns={"instruction": "text", "category": "label"})[["text", "label", "flags"]].copy()
    df["flags"] = df["flags"].astype(str)
    df = df.dropna(subset=["text", "label"]).drop_duplicates("text")
    df["text"] = [_fill_placeholders(t, rng) for t in df["text"]]
    df = df[df["text"].str.len() >= 5].drop_duplicates("text").reset_index(drop=True)
    df["noisy"] = df["flags"].apply(lambda f: any(c in f for c in noisy_flags))
    return df


class _Pools:
    """Without-replacement samplers keyed by (label, noisy)."""

    def __init__(self, df: pd.DataFrame, rng: np.random.Generator):
        self.rng = rng
        self.pools: dict[tuple[str, bool], list[int]] = {}
        for key, idx in df.groupby(["label", "noisy"]).groups.items():
            ids = list(idx)
            rng.shuffle(ids)
            self.pools[(str(key[0]), bool(key[1]))] = ids
        self.reused = 0
        self.df = df

    def take(self, label: str, noisy: bool) -> int:
        for k in ((label, noisy), (label, not noisy)):
            if self.pools.get(k):
                return self.pools[k].pop()
        # Pool exhausted: reuse a random text of the class (counted and reported).
        self.reused += 1
        cand = self.df.index[self.df["label"] == label]
        return int(self.rng.choice(cand))


def _shares(day: int, p: dict) -> tuple[float, float]:
    """Surge-class share and noisy share for a given simulation day (linear 7-day ramp)."""
    start = p["drift_month"] * p["days_per_month"]
    w = float(np.clip((day - start + 1) / 7.0, 0.0, 1.0)) if day >= start else 0.0
    surge = p["surge_share_before"] + w * (p["surge_share_after"] - p["surge_share_before"])
    noisy = p["noisy_share_before"] + w * (p["noisy_share_after"] - p["noisy_share_before"])
    return surge, noisy


def build_stream(df: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(p["seed"])
    labels = sorted(df["label"].unique())
    surge_cls = p["surge_class"]
    base = df["label"].value_counts(normalize=True).drop(surge_cls)
    base = base / base.sum()
    pools = _Pools(df, rng)

    def sample(n: int, surge_share: float, noisy_share: float) -> list[int]:
        out = []
        for _ in range(n):
            lab = surge_cls if rng.random() < surge_share else str(rng.choice(base.index, p=base.values))
            out.append(pools.take(lab, bool(rng.random() < noisy_share)))
        return out

    hist_idx = sample(p["history_size"], p["surge_share_before"], p["noisy_share_before"])
    history = df.loc[hist_idx, ["text", "label", "noisy"]].reset_index(drop=True)

    rows = []
    n_days = p["n_months"] * p["days_per_month"]
    for day in range(n_days):
        s, nz = _shares(day, p)
        for i in sample(p["tickets_per_day"], s, nz):
            rows.append((day, day // p["days_per_month"], i))
    stream = pd.DataFrame(rows, columns=["day", "month", "idx"])
    stream = stream.join(df[["text", "label", "noisy"]], on="idx").drop(columns="idx")
    stream["labelled"] = rng.random(len(stream)) < p["label_fraction"]
    stream["ticket_id"] = np.arange(len(stream))
    if pools.reused:
        print(f"[data] note: {pools.reused} texts reused because a pool was exhausted")
    return history, stream.reset_index(drop=True), labels


def prepare() -> dict:
    ensure_dirs()
    p = load_params()["data"]
    raw = pd.read_csv(download_raw(), sep=";")
    df = clean(raw, p["noisy_flags"], p["seed"])
    history, stream, labels = build_stream(df, p)
    history.to_parquet(HISTORY_FILE, index=False)
    stream.to_parquet(STREAM_FILE, index=False)
    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "unique_texts": int(len(df)),
        "history": int(len(history)),
        "stream": int(len(stream)),
        "labels": labels,
        "surge_share_by_month": stream.groupby("month")["label"]
        .apply(lambda s: round(float((s == p["surge_class"]).mean()), 3))
        .to_dict(),
        "noisy_share_by_month": stream.groupby("month")["noisy"].mean().round(3).to_dict(),
    }
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return summary


def load_history() -> pd.DataFrame:
    return pd.read_parquet(HISTORY_FILE)


def load_stream() -> pd.DataFrame:
    return pd.read_parquet(STREAM_FILE)


def load_labels() -> list[str]:
    return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
