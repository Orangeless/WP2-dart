"""
dataio.py — load ConflictBank instances and assemble the "evidence bundle" an agent
must adjudicate.

A ConflictBank instance (fields used, per the upstream scripts):
  subject, relation, object                 : the underlying Wikidata fact
  question, options[4], correct_option, replaced_option   : its MCQ framing
  correct_evidence                          : a passage supporting the true object
  fact_conflict_evidence                    : a passage supporting a WRONG object
  temporal_conflict_evidence                : a passage true only in another time span
  semantic_conflict_evidence                : a passage shifting the entity's meaning
  conflict_time_span, semantic_description  : extra metadata for temporal/semantic

We reframe the MCQ into open adjudication: build a list of evidence passages
(correct + conflicting, optionally shuffled, optionally source-tagged) and keep the
gold object so we can score whatever the agent outputs.
"""
import json, random, re
from typing import List, Dict
import config

# Source types are assigned by passage content role, NOT by which field they come
# from.  When metadata is ON, every passage draws from a pool so the correct
# evidence is not identifiable by source type alone.  We store a per-field
# annotation but the resolver must not be able to back out "which field" from it.
_SOURCE_OF = {
    "correct_evidence":             "wikipedia",
    "fact_conflict_evidence":       "wikipedia",   # could also be correct — do not leak
    "temporal_conflict_evidence":   "news",
    "semantic_conflict_evidence":   "news",
}


def load_instances(n: int = None) -> List[Dict]:
    n = n or config.N_INSTANCES
    if config.USE_LOCAL_SAMPLE:
        rows = [json.loads(l) for l in open(config.SAMPLE_PATH) if l.strip()]
        return rows[:n]
    # streaming avoids downloading all 513k CB_qa records
    from datasets import load_dataset
    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    rename = {"default_evidence": "correct_evidence",                    # CB_qa cols -> our field names
              "misinformation_conflict_evidence": "fact_conflict_evidence",
              "misinformation_conflict_evidence_evidence": "fact_conflict_evidence",
              "temporal_conflict_time_span": "conflict_time_span",
              "replace_option": "replaced_option"}
    return [{rename.get(k, k): v for k, v in r.items()} for _, r in zip(range(n), ds)]


_PID = re.compile(r"^P\d+$")


def query_phrase(inst: Dict) -> str:
    """The natural-language question a resolver must answer.

    CB_qa stores `relation` as a bare Wikidata property id ("P166"), so prompting with
    it asks the model about a symbol it has never seen — it echoes "P166" back as the
    object. The dataset ships a `question` field for exactly this; prefer it, and only
    synthesise one when it is absent (e.g. sample_data.jsonl, whose relations are
    already human-readable).
    """
    q = str(inst.get("question") or "").strip()
    if q:
        return q
    relation = str(inst.get("relation") or "").strip()
    subject = str(inst.get("subject") or "").strip()
    if relation and not _PID.match(relation):
        return f"What is the {relation} of {subject}?"
    raise ValueError(
        f"instance {inst.get('id', subject)!r} has no `question` and its relation "
        f"{relation!r} is an opaque property id — cannot build a promptable query."
    )


def gold_object(inst: Dict) -> str:
    """The true object string (what a correct adjudication should return)."""
    if "object" in inst:
        return inst["object"]
    # fall back to the correct MCQ option
    opts, ci = inst.get("options"), inst.get("correct_option")
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ci, ci)
    return opts[idx] if opts and idx is not None else inst.get("correct_object", "")


def build_bundle(inst: Dict, rng: random.Random) -> List[Dict]:
    """
    Assemble the evidence the resolver sees: {source, text, timestamp?} passages.
    Correct passage optional; N conflicting passages drawn from CONFLICT_TYPES present.
    """
    bundle = []
    if config.INCLUDE_CORRECT_EVIDENCE and inst.get("correct_evidence"):
        bundle.append(_passage("correct_evidence", inst))
    available = [t + "_evidence" for t in config.CONFLICT_TYPES
                 if inst.get(t + "_evidence")]
    rng.shuffle(available)
    for field in available[: config.N_CONFLICT_SOURCES]:
        bundle.append(_passage(field, inst))
    if config.SHUFFLE_SOURCES:
        rng.shuffle(bundle)
    return bundle


def _passage(field: str, inst: Dict) -> Dict:
    p = {"text": inst[field]}
    if config.ADD_SUBJECT_CONTEXT:
        # neutral: the subject's own description, independent of either answer.
        sd = str(inst.get("subject_description") or "").strip()
        if sd:
            p["subject_desc"] = sd
            p["subject_name"] = str(inst.get("subject") or "").strip()
    if config.TAG_SOURCE_METADATA:
        p["source"] = _SOURCE_OF.get(field, "unknown")
        # Timestamps: use conflict_time_span for temporal-conflict passages;
        # everything else gets a neutral placeholder that does not sort above
        # any plausible year.  "current" is not a year — it sorts AFTER numbers
        # in string comparison, which was backwards.  Use a fixed sentinel.
        span = inst.get("conflict_time_span")
        if field == "temporal_conflict_evidence" and span:
            p["timestamp"] = span[0] if isinstance(span, list) else span
        else:
            # neutral timestamp: not "current" (which sorts after digits)
            p["timestamp"] = None
    return p


def render_bundle(bundle: List[Dict]) -> str:
    """Human/LLM-readable rendering of the evidence bundle for a prompt."""
    lines = []
    for i, p in enumerate(bundle, 1):
        meta_bits = []
        if config.TAG_SOURCE_METADATA:
            meta_bits.append(f"source: {p.get('source','?')}, time: {p.get('timestamp','?')}")
        if config.ADD_SUBJECT_CONTEXT and p.get("subject_desc"):
            name = p.get("subject_name") or "subject"
            meta_bits.append(f"context — {name}: {p['subject_desc']}")
        meta = f" [{'; '.join(meta_bits)}]" if meta_bits else ""
        lines.append(f"Source {i}{meta}: {p['text']}")
    return "\n\n".join(lines)
