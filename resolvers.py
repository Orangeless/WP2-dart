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
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _extract_object_from_text(subject: str, query: str, text: str,
                              model: str, inst_id: str, resolver: str,
                              log: List[CallLog]) -> str:
    """Ask the model what object a single passage asserts, given the natural-language query.
    The subject is included for disambiguation (e.g. 'Mercury' the planet vs element)."""
    prompt = (f"Subject: {subject}\n"
              f"Question: {query}\n"
              f"From the passage below, extract the answer to the question.\n"
              f"Answer with only the value, no extra words or punctuation.\n\n"
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
        ts = p.get("timestamp")
        try:
            return int(ts)
        except (TypeError, ValueError):
            return 0  # no timestamp sorts lowest — not most recent
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
def _llm_adjudicate(subject, query, bundle, model, inst_id, log, ask_provenance) -> Dict:
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
        f"reliability and internal consistency; do not just pick the majority.\n\n{rendered}\n\n"
        f'Respond as JSON: {{"object": "<value>"{prov}}}')
    out = chat(prompt, model, resolver="llm_judge_provenance" if ask_provenance else "llm_judge",
               stage="adjudicate", inst_id=inst_id, temperature=config.TEMPERATURE,
               max_tokens=config.MAX_TOKENS, seed=config.SEED, log=log)
    try:
        m = re.search(r"\{.*\}", out, re.S)
        if m is None:
            d = {}
        else:
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                d = {}
    except Exception:
        d = {}
    return {"object": d.get("object"),
            "chosen_source": d.get("chosen_source"),
            "rationale": d.get("rationale", "")}


def llm_judge(subject, query, bundle, model, inst_id, log) -> Dict:
    return _llm_adjudicate(subject, query, bundle, model, inst_id, log, False)


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
    try:
        m = re.search(r"\{.*\}", out, re.S)
        if m is None:
            d = {}
        else:
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                d = {}
    except Exception:
        d = {}
    return {"object": d.get("object"), "chosen_source": d.get("chosen_source"),
            "rationale": d.get("rationale", "")}


_REGISTRY = {
    "majority_vote": majority_vote,
    "recency": recency,
    "source_trust": source_trust,
    "llm_judge": llm_judge,
    "llm_judge_provenance": llm_judge_provenance,
    "multi_agent_debate": multi_agent_debate,
}


def get_resolver(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown resolver '{name}'. known: {list(_REGISTRY)}")
    return _REGISTRY[name]
