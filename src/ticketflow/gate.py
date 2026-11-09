"""Quality gate: a candidate is registered only if it is not worse than production.

Rules (thresholds in params.yaml -> gate):
1. macro-F1 on the golden set >= absolute floor;
2. macro-F1 >= production macro-F1 + min_macro_f1_delta;
3. no class (with at least min_class_support golden examples) loses more than max_class_drop_pp
   percentage points of F1 versus production.
Both models are scored on the *same* golden set, so the comparison is apples to apples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    candidate_macro_f1: float = 0.0
    production_macro_f1: float | None = None
    worst_class: str | None = None
    worst_class_drop_pp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def check(candidate: dict, production: dict | None, params: dict) -> GateResult:
    res = GateResult(passed=True, candidate_macro_f1=candidate["macro_f1"])
    if candidate["macro_f1"] < params["min_macro_f1"]:
        res.passed = False
        res.reasons.append(f"macro-F1 {candidate['macro_f1']:.3f} < floor {params['min_macro_f1']:.3f}")

    if production is None:
        return res

    res.production_macro_f1 = production["macro_f1"]
    delta = candidate["macro_f1"] - production["macro_f1"]
    if delta < params["min_macro_f1_delta"]:
        res.passed = False
        res.reasons.append(
            f"macro-F1 {candidate['macro_f1']:.3f} vs production {production['macro_f1']:.3f} (delta {delta:+.3f})"
        )

    support = candidate.get("support", {})
    for label, prod_f1 in production["per_class_f1"].items():
        cand_f1 = candidate["per_class_f1"].get(label)
        if prod_f1 is None or cand_f1 is None:
            continue
        if support.get(label, 10**9) < params.get("min_class_support", 0):
            continue
        drop_pp = (prod_f1 - cand_f1) * 100
        if drop_pp > res.worst_class_drop_pp:
            res.worst_class, res.worst_class_drop_pp = label, drop_pp
    if res.worst_class_drop_pp > params["max_class_drop_pp"]:
        res.passed = False
        res.reasons.append(
            f"class {res.worst_class} dropped by {res.worst_class_drop_pp:.1f} pp (> {params['max_class_drop_pp']})"
        )
    return res
