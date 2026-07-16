# Code Changes — resolvers.py

## File: `resolvers.py`
**Date:** 16 July 2026  
**Scope:** Fix `multi_agent_debate` to implement an actual debate loop.

---

## What was broken

The `multi_agent_debate` resolver claimed to implement a KARMA/MADAM-RAG-style
multi-agent debate, but the actual code had no debate — it was structurally
identical to `majority_vote` with an extra judge call:

1. Each advocate extracted a value from its source (same call `majority_vote` makes)
2. The extracted values were concatenated into a list
3. A judge picked one from the list

No advocate ever saw another advocate's claim. No advocate defended its position.
No rebuttals. No iterative reasoning. The "debate" was a single-pass extraction
followed by a judge call — adding no reasoning that the judge couldn't do from
the raw evidence directly.

---

## What changed

### 1. Advocates now store structured data (Round 1)

**Before:** advocates were plain strings — `"Agent 1 (source: ?) answers: X. Basis: ..."`

**After:** advocates are dicts with `agent`, `source`, `answer`, and `basis` fields,
so the rebuttal round can reference each advocate's claim individually.

```python
# Before
advocates.append(f"Agent {i} (source: {p.get('source','?')}) answers: {obj}. "
                 f"Basis: {p['text'][:300]}")

# After
advocates.append({"agent": i, "source": p.get("source", "?"),
                  "answer": obj, "basis": p["text"][:300]})
```

### 2. New rebuttal round (Round 2)

Each advocate now sees the other advocates' extracted claims and is prompted to
argue why its own source is more reliable. This is the actual debate — the
advocates must engage with competing claims and justify their position.

```python
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
```

### 3. Judge sees arguments, not just answers

**Before:** judge saw `"Agent 1 answers: X. Basis: [300 chars of source text]"`

**After:** judge sees `"Agent 1 (source: wikipedia): answer = X. Argument: [advocate's reasoned defence]"`

The judge now adjudicates over reasoned arguments, not bare assertions.

```python
joined = "\n\n".join(
    f"Agent {r['agent']} (source: {r['source']}): "
    f"answer = {r['answer']}. Argument: {r['argument']}"
    for r in rebuttals
)
```

### 4. Updated docstring

Reflects the actual two-round structure and the true cost (2N+1 calls, not N+1).

---

## Cost impact

| Bundle size | Before (N+1 calls) | After (2N+1 calls) |
|-------------|--------------------|--------------------|
| 1 source    | 2 calls            | 3 calls            |
| 2 sources   | 3 calls            | 5 calls            |
| 3 sources   | 4 calls            | 7 calls            |

The added cost is the point — the study measures whether the extra reasoning
round improves resolution accuracy enough to justify the cost.

---

## What did NOT change

- The function signature: `multi_agent_debate(subject, query, bundle, model, inst_id, log) -> Dict`
- The return format: `{"object": ..., "chosen_source": ..., "rationale": ...}`
- The `_REGISTRY` entry — still registered as `"multi_agent_debate"`
- The judge prompt structure and JSON parsing — identical to before
- All other resolvers — untouched
- `config.py`, `metrics.py`, `run_experiment.py` — untouched

The change is backwards-compatible: any code that called `multi_agent_debate`
before will work identically, just with the improved debate logic.
