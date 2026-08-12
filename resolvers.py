"""
resolvers.py — the "method" axis. A resolver reads an evidence bundle that disagrees
about a (subject, relation) and returns:
    {"object": <adjudicated value or None>, "chosen_source": <int or None>, "rationale": str}

These are the conflict-resolution techniques an intern implements and compares. Start
with llm_judge; the non-LLM ones (majority/recency/source_trust) are cheap, transparent
baselines that a good paper needs. All are deterministic given the same inputs except
the LLM ones.
"""
import re, json
from collections import Counter
from typing import List, Dict
import config
from llm import chat, CallLog


# ---- helpers ----------------------------------------------------------------
# Answer-granularity guidance. Gold comes from Wikidata, which names the most specific
# entity ("The Azrieli Faculty of Medicine", not "Bar-Ilan University"); models tend to
# answer with the parent, which then scores as wrong for a reason that has nothing to do
# with conflict resolution. This says nothing about WHICH source is right, so it is a
# formatting instruction, not a hint — and every resolver gets the same one.
_SPECIFICITY = ("Name the most specific entity the evidence gives — the faculty rather "
                "than its parent university, the exact award rather than the body that "
                "awards it.")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _parse_json(out: str) -> Dict:
    """The JSON object in a model reply, or {} if there isn't a usable one.

    Models wrap the object in prose or fences, and thinking models sometimes truncate
    mid-object. Every failure mode lands on {}, which `_forced_object` then recovers
    from the raw text — so a formatting slip never becomes a false abstain.
    """
    m = re.search(r"\{.*\}", out or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def _forced_object(out: str, parsed: Dict):
    """The object to return, honouring config.ALLOW_ABSTAIN.

    When abstaining is allowed we pass the parsed value through untouched (None => abstain).
    When it is NOT allowed the resolver must never abstain, so if the JSON was empty or
    unparseable we recover the answer from the raw reply — first the value after "object",
    then, failing that, the stripped raw text — so the instance always gets an answer.
    """
    _nullish = ("", "null", "none", "n/a", "na", "cannot determine", "unknown")
    obj = parsed.get("object")
    if config.ALLOW_ABSTAIN or (obj is not None and str(obj).strip().lower() not in _nullish):
        return obj
    m = re.search(r'"?object"?\s*[:=]\s*"?([^"\n}]+)', out or "", re.I)
    cand = (m.group(1) if m else re.sub(r'[`{}"]', " ", out or "")).strip().strip(".,")
    return cand[:120] if cand and cand.lower() not in _nullish else None


def _extract_object_from_text(subject: str, query: str, text: str,
                              model: str, inst_id: str, resolver: str,
                              log: List[CallLog]) -> str:
    """Ask the model what object a single passage asserts, given the natural-language query.
    The subject is included for disambiguation (e.g. 'Mercury' the planet vs element)."""
    prompt = (f"Subject: {subject}\n"
              f"Question: {query}\n"
              f"From the passage below, extract the answer to the question.\n"
              f"Answer with only the value, no extra words or punctuation. {_SPECIFICITY}\n\n"
              f"Passage: {text}")
    out = chat(prompt, model, resolver=resolver, stage="extract", inst_id=inst_id,
               temperature=0.0, max_tokens=config.MAX_TOKENS, seed=config.SEED, log=log)
    return out.strip().strip('".')


# ---- 1. non-LLM baselines ---------------------------------------------------
def majority_vote(subject, query, bundle, model, inst_id, log) -> Dict:
    """Extract an object from each source, return the most frequent."""
    vals = [_extract_object_from_text(subject, query, p["text"], model, inst_id,
                                      "majority_vote", log) for p in bundle]
    if not vals:
        return {"object": None, "chosen_source": None, "rationale": "no evidence"}
    top, _ = Counter(_norm(v) for v in vals).most_common(1)[0]
    idx = next(i for i, v in enumerate(vals) if _norm(v) == top)
    return {"object": vals[idx], "chosen_source": idx, "rationale": "majority of sources"}


def recency(subject, query, bundle, model, inst_id, log) -> Dict:
    """Trust the most recent source — requires numeric timestamps (years).
    When timestamps are missing or equal, picks the first source (arbitrary but
    deterministic)."""
    if not bundle:
        return {"object": None, "chosen_source": None, "rationale": "no evidence"}
    def _ts_key(p) -> int:
        # Spans are not always bare years — "April, 2031", "8 August, 2026", "2024".
        # int() threw on ~half of them and silently scored them 0, so this baseline was
        # picking arbitrarily. Pull the first 4-digit year out of whatever we got.
        yrs = re.findall(r"(?:19|20)\d{2}", str(p.get("timestamp") or ""))
        return int(yrs[0]) if yrs else 0  # no timestamp sorts lowest — not most recent
    idx = max(range(len(bundle)), key=lambda i: _ts_key(bundle[i]))
    obj = _extract_object_from_text(subject, query, bundle[idx]["text"], model,
                                    inst_id, "recency", log)
    return {"object": obj, "chosen_source": idx, "rationale": "most recent source"}


def source_trust(subject, query, bundle, model, inst_id, log) -> Dict:
    """Trust the source with the highest prior reliability."""
    idx = max(range(len(bundle)),
              key=lambda i: config.SOURCE_TRUST_PRIOR.get(bundle[i].get("source", "unknown"), 0.5)) if bundle else None
    if idx is None:
        return {"object": None, "chosen_source": None, "rationale": "no evidence"}
    obj = _extract_object_from_text(subject, query, bundle[idx]["text"], model,
                                    inst_id, "source_trust", log)
    return {"object": obj, "chosen_source": idx,
            "rationale": f"highest-trust source ({bundle[idx].get('source')})"}


# ---- 2. LLM adjudicators ----------------------------------------------------
def _llm_adjudicate(subject, query, bundle, model, inst_id, log, ask_provenance,
                    temperature=None, seed=None) -> Dict:
    import dataio
    rendered = dataio.render_bundle(bundle)
    abstain_instruction = (
        'If the evidence is insufficient, set "object" to null. '
        if config.ALLOW_ABSTAIN else
        'Do not return null or "cannot determine"; always choose the single most likely value. '
    )
    prov = (', "chosen_source": <the Source number you trusted>, '
            '"rationale": "<one sentence why>"') if ask_provenance else ''
    prompt = (
        f"Several sources disagree about the answer to this question: {query}\n\n"
        f"Decide the single most likely correct value. {abstain_instruction}Weigh source "
        f"reliability and internal consistency; do not just pick the majority. Some passages "
        f"may be outdated (true only at an earlier time) or describe a different entity that "
        f"shares the name — disregard those and give the value that is correct NOW.\n\n"
        f"Answer with ONLY the canonical name of the answer entity — no sentence, no "
        f"explanation, no extra qualifiers. {_SPECIFICITY}\n\n{rendered}\n\n"
        f'Respond as JSON: {{"object": "<value>"{prov}}}')
    out = chat(prompt, model, resolver="llm_judge_provenance" if ask_provenance else "llm_judge",
               stage="adjudicate", inst_id=inst_id,
               temperature=config.TEMPERATURE if temperature is None else temperature,
               max_tokens=config.MAX_TOKENS, seed=config.SEED if seed is None else seed, log=log)
    d = _parse_json(out)
    return {"object": _forced_object(out, d),
            "chosen_source": d.get("chosen_source"),
            "rationale": d.get("rationale", "")}


def llm_judge(subject, query, bundle, model, inst_id, log) -> Dict:
    return _llm_adjudicate(subject, query, bundle, model, inst_id, log, False)


# Self-consistency: sample the adjudication SC_SAMPLES times at a non-zero temperature and
# majority-vote the answer. Cancels one-off errors — the standard accuracy boost for
# knowledge tasks (Wang et al. 2022, "Self-Consistency Improves Chain of Thought"). Costs
# SC_SAMPLES x the calls; this is the lever meant to push a strong model past its single-shot
# ceiling toward the 90s (pair it with the knowledge filter).
SC_SAMPLES = 5
SC_TEMPERATURE = 0.5


def llm_judge_sc(subject, query, bundle, model, inst_id, log) -> Dict:
    votes = []
    for k in range(SC_SAMPLES):
        d = _llm_adjudicate(subject, query, bundle, model, inst_id, log, False,
                            temperature=SC_TEMPERATURE, seed=config.SEED + k)
        if d.get("object"):
            votes.append(d["object"])
    if not votes:
        return {"object": None, "chosen_source": None, "rationale": "no answer in any sample"}
    top = Counter(_norm(v) for v in votes).most_common(1)[0][0]
    obj = next(v for v in votes if _norm(v) == top)
    agree = sum(1 for v in votes if _norm(v) == top)
    return {"object": obj, "chosen_source": None,
            "rationale": f"majority {agree}/{len(votes)} samples"}


def llm_judge_provenance(subject, query, bundle, model, inst_id, log) -> Dict:
    return _llm_adjudicate(subject, query, bundle, model, inst_id, log, True)


# ---- 3. multi-agent debate (the "agentic" headline; cf. MADAM-RAG 2504.13079) ----
def multi_agent_debate(subject, query, bundle, model, inst_id, log) -> Dict:
    """Two-round debate: each advocate extracts its value (round 1), then sees
    competing claims and argues why its source is more reliable (round 2).
    A judge then adjudicates over the arguments. This is the KARMA-style /
    MADAM-RAG-style method meant to beat the single-call baselines.

    Cost: 2N+1 LLM calls (N extracts + N rebuttals + 1 judge) where N = number
    of sources. The cost/quality trade-off vs llm_judge is the study axis."""
    # Round 1: each advocate extracts its value from its source passage.
    advocates = []
    for i, p in enumerate(bundle, 1):
        obj = _extract_object_from_text(subject, query, p["text"], model, inst_id,
                                        "multi_agent_debate", log)
        advocates.append({"agent": i, "source": p.get("source", "?"),
                          "answer": obj, "basis": p["text"][:300]})

    # Round 2: each advocate sees the other advocates' claims and argues why its
    # own source is more reliable.  This is the actual "debate" — without it the
    # method collapses into majority_vote + judge.
    round1_summary = "\n".join(
        f"Agent {a['agent']} (source: {a['source']}) claims the answer is: {a['answer']}."
        for a in advocates
    )
    rebuttals = []
    for a in advocates:
        prompt = (
            f"You are Agent {a['agent']}, advocating for the reliability of your source "
            f"(source type: {a['source']}).\n\n"
            f"Question: {query}\n"
            f"Your source passage: {a['basis']}\n"
            f"Your extracted answer: {a['answer']}\n\n"
            f"Other agents claim:\n{round1_summary}\n\n"
            f"In one or two sentences, argue why your answer is more trustworthy than "
            f"the competing claims. Consider source reliability, internal consistency, "
            f"and whether the other answers could be outdated or wrong."
        )
        out = chat(prompt, model, resolver="multi_agent_debate", stage="rebuttal",
                   inst_id=inst_id, temperature=config.TEMPERATURE,
                   max_tokens=config.MAX_TOKENS, seed=config.SEED, log=log)
        rebuttals.append({"agent": a["agent"], "source": a["source"],
                          "answer": a["answer"], "argument": out.strip()})

    # Judge: sees both the extracted answers and the rebuttal arguments.
    joined = "\n\n".join(
        f"Agent {r['agent']} (source: {r['source']}): "
        f"answer = {r['answer']}. Argument: {r['argument']}"
        for r in rebuttals
    )
    value_placeholder = "<value or null>" if config.ALLOW_ABSTAIN else "<value>"
    prompt = (f"You are the judge. Agents disagree about the answer to this question: {query}\n\n"
              f"Weigh reliability and consistency; suppress misinformation.\n\n{joined}\n\n"
              f'Respond as JSON: {{"object": "{value_placeholder}", '
              f'"chosen_source": <agent number>, "rationale": "<one sentence>"}}')
    out = chat(prompt, model, resolver="multi_agent_debate", stage="judge", inst_id=inst_id,
               temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
               seed=config.SEED, log=log)
    d = _parse_json(out)
    return {"object": _forced_object(out, d), "chosen_source": d.get("chosen_source"),
            "rationale": d.get("rationale", "")}


_REGISTRY = {
    "majority_vote": majority_vote,
    "recency": recency,
    "source_trust": source_trust,
    "llm_judge": llm_judge,
    "llm_judge_sc": llm_judge_sc,
    "llm_judge_provenance": llm_judge_provenance,
    "multi_agent_debate": multi_agent_debate,
}


def get_resolver(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown resolver '{name}'. known: {list(_REGISTRY)}")
    return _REGISTRY[name]
