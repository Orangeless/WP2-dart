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

# ConflictBank ships the real provenance of every passage in its own *_category column
# ("book" / "new" / "wikipedia"), and it genuinely varies per record.  Use it.
#
# This replaces a hardcoded map that tagged correct+fact "wikipedia" and
# temporal+semantic "news" for EVERY record — which meant a resolver could identify the
# planted passage from its source tag alone, with no reasoning at all.  That leak was
# ours, not the dataset's.
_CATEGORY_OF = {
    "correct_evidence":             "default_evidence_category",
    "fact_conflict_evidence":       "misinformation_conflict_evidence_category",
    "temporal_conflict_evidence":   "temporal_conflict_evidence_category",
    "semantic_conflict_evidence":   "semantic_conflict_evidence_category",
}


# Gold values that no model can be expected to produce, because they are artifacts of the
# Wikidata relation rather than an answer to the question. Measured on a 300-instance
# Opus 4.5 control run (correct evidence, no conflict): the model scored 0/16 on these,
# so they are pure denominator — they depress the knowledge ceiling without measuring
# knowledge, and the conflict run can never reach them because the knowledge filter
# excludes them anyway.
#   "filing"                    — 12/300, the instance-of value for patent records
#   "anciens cadres", ...       — French INSEE occupation categories, asked in English
#   "Wikipedia:List of ..."     — a Wikipedia project page, not an entity
# Set DROP_UNANSWERABLE = False to keep them and measure the size of the effect.
_UNANSWERABLE_GOLD = {"filing"}
_UNANSWERABLE_PREFIX = ("anciens ", "anciennes ", "wikipedia:")


def _answerable(inst: Dict) -> bool:
    g = str(gold_object(inst) or "").strip().lower()
    return bool(g) and g not in _UNANSWERABLE_GOLD and not g.startswith(_UNANSWERABLE_PREFIX)


def load_instances(n: int = None) -> List[Dict]:
    """The first `n` ANSWERABLE instances. Dropping the unanswerable ones costs extra
    records off the stream, not a smaller sample — `n` is honoured either way."""
    n = n or config.N_INSTANCES
    if config.USE_LOCAL_SAMPLE:
        rows = [json.loads(l) for l in open(config.SAMPLE_PATH) if l.strip()]
        return [r for r in rows if not config.DROP_UNANSWERABLE or _answerable(r)][:n]
    # streaming avoids downloading all 513k CB_qa records
    from datasets import load_dataset
    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)
    rename = {"default_evidence": "correct_evidence",                    # CB_qa cols -> our field names
              "misinformation_conflict_evidence": "fact_conflict_evidence",
              "misinformation_conflict_evidence_evidence": "fact_conflict_evidence",
              "temporal_conflict_time_span": "conflict_time_span",
              "replace_option": "replaced_option"}
    out, dropped = [], 0
    for r in ds:
        inst = {rename.get(k, k): v for k, v in r.items()}
        if config.DROP_UNANSWERABLE and not _answerable(inst):
            dropped += 1
            continue
        out.append(inst)
        if len(out) >= n:
            break
    if dropped:
        print(f"[data] dropped {dropped} instance(s) with unanswerable gold "
              f"(see dataio._UNANSWERABLE_GOLD); kept {len(out)}")
    return out


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
        cat = str(inst.get(_CATEGORY_OF.get(field, ""), "") or "").strip().lower()
        p["source"] = {"new": "news"}.get(cat, cat) or "unknown"
        # Only temporal-conflict passages carry a date. It stays on the bundle for the
        # `recency` baseline but is NOT rendered by default (config.SHOW_TIMESTAMPS) —
        # a timestamp that only ever appears on the planted passage identifies it by
        # presence alone.
        span = inst.get("conflict_time_span")
        if field == "temporal_conflict_evidence" and span:
            p["timestamp"] = span[0] if isinstance(span, list) else span
    return p


def render_bundle(bundle: List[Dict]) -> str:
    """Human/LLM-readable rendering of the evidence bundle for a prompt."""
    lines = []
    for i, p in enumerate(bundle, 1):
        meta_bits = []
        if config.TAG_SOURCE_METADATA:
            bits = f"source: {p.get('source', 'unknown')}"
            if config.SHOW_TIMESTAMPS and p.get("timestamp"):
                bits += f", time: {p['timestamp']}"
            meta_bits.append(bits)
        if config.ADD_SUBJECT_CONTEXT and p.get("subject_desc"):
            name = p.get("subject_name") or "subject"
            meta_bits.append(f"context — {name}: {p['subject_desc']}")
        meta = f" [{'; '.join(meta_bits)}]" if meta_bits else ""
        lines.append(f"Source {i}{meta}: {p['text']}")
    return "\n\n".join(lines)
