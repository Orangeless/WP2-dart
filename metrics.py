"""
metrics.py — the quality axis for conflict resolution. Deterministic string matching
against the gold object, plus the error bar every reported number carries.

Per instance we record whether the resolver:
  - answered correctly (adjudicated object == gold object),
  - was misled (returned the planted-conflict object), or
  - abstained (returned None).

Answers that name the same entity in different words are recovered separately, by the
equivalence judge in rescore.py — deliberately not here, so this module stays offline
and deterministic.
"""
import math
import re
from typing import Dict


def _canon(s):
    """Canonicalise an answer for fair equality: drop an appended UPPERCASE acronym
    like "(MIT)" and a leading article, then strip punctuation/space. This recovers
    "MIT" == "Massachusetts Institute of Technology (MIT)" and "The Azrieli…" == "Azrieli…"
    WITHOUT the false positives of substring matching (it still rejects
    "John Entwistle" vs "The Who" and lowercase disambiguators like "Mercury (planet)").
    """
    s = re.sub(r"\(\s*[A-Z0-9&]{2,}\s*\)", " ", str(s) if s is not None else "")  # drop "(MIT)"
    s = re.sub(r"^\s*(the|a|an)\s+", "", s.lower())                               # drop leading article
    return re.sub(r"[\s_\"'.,:]+", "", s)


def match(a: str, b: str) -> bool:
    """Equality after canonicalisation — exact, not substring (no false positives)."""
    na, nb = _canon(a), _canon(b)
    if not na or not nb:
        return False
    return na == nb


def score_instance(pred: Dict, gold_object: str, conflict_object: str = None) -> Dict:
    obj = pred.get("object")
    # A blank or literal "null" reply is an abstain, not a wrong answer — with
    # config.ALLOW_ABSTAIN off it means the call failed and the instance was skipped.
    flat = re.sub(r"[\s_\"'.]+", "", str(obj) if obj is not None else "").lower()
    abstained = flat in ("", "null")
    correct = (not abstained) and match(obj, gold_object)
    misled = (not abstained) and conflict_object is not None and match(obj, conflict_object)
    return {"correct": int(correct), "misled": int(misled),
            "abstained": int(abstained), "pred_object": obj, "gold_object": gold_object}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion — the error bar on every accuracy we
    report. Wilson rather than normal-approximation because n is small (~70-250) and the
    proportions sit near 1, where the naive interval runs past 100%."""
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
