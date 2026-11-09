"""Classification metrics shared by training, the quality gate and the simulator."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """macro-F1 over classes that occur in y_true or y_pred (sklearn default).

    Classes absent from both are reported as None in `per_class_f1` instead of a misleading 0.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    present = [lab for lab in labels if lab in set(y_true) | set(y_pred)]
    per_class = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    f1_by_label = dict(zip(present, per_class, strict=True))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {lab: (float(f1_by_label[lab]) if lab in f1_by_label else None) for lab in labels},
        "support": {lab: int(sum(t == lab for t in y_true)) for lab in labels},
        "n": len(y_true),
    }
