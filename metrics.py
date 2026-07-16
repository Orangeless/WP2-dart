"""
metrics.py — the quality axis for conflict resolution. Deterministic string matching
against the gold object. No LLM judge needed for scoring.

Per instance we record whether the resolver:
  - answered correctly (adjudicated object == gold object),
  - was misled (returned the planted-conflict object),
  - abstained (returned None), and
  - cited the right source (provenance correctness, when available).

Aggregate broken down by conflict type is the headline result.
"""
import re
from typing import Dict, List


def _norm(s):
    return re.sub(r"[\s_\"'.]+", "", str(s) if s is not None else "").lower()


def match(a: str, b: str) -> bool:
    """Normalised string equality — strict, not substring."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb


def score_instance(pred: Dict, gold_object: str, conflict_object: str = None) -> Dict:
    obj = pred.get("object")
    abstained = obj is None or _norm(obj) == "" or _norm(obj) == "null"
    correct = (not abstained) and match(obj, gold_object)
    misled = (not abstained) and conflict_object is not None and match(obj, conflict_object)
    return {"correct": int(correct), "misled": int(misled),
            "abstained": int(abstained), "pred_object": obj, "gold_object": gold_object}


def aggregate(rows: List[Dict]) -> Dict:
    """Overall + per-conflict-type accuracy / misled-rate / abstain-rate."""
    def frac(sel, key):
        s = [r for r in rows if sel(r)]
        return round(sum(r[key] for r in s) / len(s), 3) if s else None
    out = {"n": len(rows),
           "accuracy": frac(lambda r: True, "correct"),
           "misled_rate": frac(lambda r: True, "misled"),
           "abstain_rate": frac(lambda r: True, "abstained")}
    for ct in sorted({r.get("conflict_type") for r in rows if r.get("conflict_type")}):
        out[f"acc[{ct}]"] = frac(lambda r, ct=ct: r.get("conflict_type") == ct, "correct")
    return out
