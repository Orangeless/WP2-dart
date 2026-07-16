"""
telecom_inject.py — the DART transfer path.

No public telecom dataset has cross-source *conflicting* records (OSS/BSS vs topology
vs field are operator-confidential). So we do what ConflictBank itself does: take a
CLEAN seed KG we trust as ground truth and INJECT controlled conflicts. Because we
inject them, we have perfect gold — resolution accuracy is measured against the clean KG.

This emits instances in the SAME format the resolvers/metrics already consume, so the
identical pipeline that ran on ConflictBank runs on telecom with zero changes.

Input: clean triples [(subject, relation, object), ...] over the TAN ontology
       (e.g. aggregation-node capacity, fibre-segment status, site vendor).
Output: instances with a correct evidence + an injected-conflict evidence + gold.
"""
import random
from typing import List, Tuple, Dict

# characteristic error profile per source view (the reliability regime to study)
SOURCE_PROFILES = {
    "oss_inventory": {"reliability": 0.7, "bias": "stale"},    # attributes lag reality
    "live_topology": {"reliability": 0.85, "bias": "noisy"},   # current but noisy
    "field_ticket":  {"reliability": 0.5, "bias": "typo"},     # human-entered, error-prone
}


def _evidence(source: str, subject: str, relation: str, value: str) -> str:
    tmpl = {
        "oss_inventory": f"OSS inventory record: {subject} has {relation} = {value}.",
        "live_topology": f"Live topology export shows {subject} {relation} {value}.",
        "field_ticket":  f"Field ticket note: engineer reports {subject} {relation} is {value}.",
    }
    return tmpl.get(source, f"{subject} {relation} {value}.")


def inject_conflicts(clean_triples: List[Tuple[str, str, str]],
                     value_pool: Dict[str, List[str]],
                     injection_rate: float = 0.5,
                     conflict_type: str = "misinformation",
                     seed: int = 7) -> List[Dict]:
    """
    clean_triples : ground-truth (subject, relation, object).
    value_pool    : relation -> list of plausible wrong objects (same type) to substitute.
    injection_rate: fraction of triples that get a conflicting source (the IV to sweep).
    conflict_type : "misinformation" | "temporal" | "semantic" (mirrors ConflictBank).
    Returns instances in the resolver/metrics format.
    """
    rng = random.Random(seed)
    out = []
    for i, (s, r, o) in enumerate(clean_triples):
        inst = {"id": f"tel_{i}", "subject": s, "relation": r, "object": o,
                "conflict_type": conflict_type + "_conflict",
                "correct_evidence": _evidence("live_topology", s, r, o)}
        if rng.random() < injection_rate and value_pool.get(r):
            wrong = rng.choice([v for v in value_pool[r] if v != o] or [o])
            # the conflicting source is the less reliable OSS/field view
            src = "oss_inventory" if conflict_type == "temporal" else "field_ticket"
            field = {"misinformation": "fact_conflict_evidence",
                     "temporal": "temporal_conflict_evidence",
                     "semantic": "semantic_conflict_evidence"}[conflict_type]
            inst[field] = _evidence(src, s, r, wrong)
            inst["replaced_object"] = wrong
            if conflict_type == "temporal":
                inst["conflict_time_span"] = ["2019"]   # stale value valid only in the past
        out.append(inst)
    return out


if __name__ == "__main__":
    # tiny demo on made-up TAN-style triples
    clean = [("AGG-Node-42", "capacity_gbps", "100"),
             ("Fibre-Seg-7", "status", "in_service"),
             ("Site-Ryde", "vendor", "Ericsson")]
    pool = {"capacity_gbps": ["40", "100", "400"],
            "status": ["in_service", "decommissioned", "planned"],
            "vendor": ["Ericsson", "Nokia", "Huawei"]}
    import json
    for inst in inject_conflicts(clean, pool, injection_rate=1.0, conflict_type="misinformation"):
        print(json.dumps(inst, indent=1))
