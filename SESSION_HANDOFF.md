# WP2 DART — Conflict Resolution Study: Session Handoff & Codebase Guide

> **Purpose of this document.** A complete, self-contained brief for (a) a teammate or
> another Claude instance building a PowerPoint from these results, and (b) anyone picking
> up the code. It explains the research task, the codebase file-by-file, everything changed
> in the 2026-07-20/21 working session, the current results, the methodology caveats that
> keep the numbers honest, and how to reproduce everything.
>
> **Read the "Results" and "Honesty & caveats" sections before making any slide.**

---

## 1. What this project is

**Research question.** When a knowledge graph is built from many sources, those sources
disagree about the same fact. Given a `(subject, relation)` and several *disagreeing*
evidence passages (one correct, the rest planted conflicts), can an LLM agent adjudicate
the correct value — and which models/strategies resist being misled?

**Dataset.** [ConflictBank](https://arxiv.org/abs/2408.12076) (NeurIPS 2024), streamed from
HuggingFace `Warrieryes/CB_qa`. Each record carries a subject/relation/object, a
`question`, a correct evidence passage, and **three** planted-conflict passages:

| Conflict type | Meaning | Our tag |
|---|---|---|
| Misinformation | a plain wrong value | `fact_conflict` |
| Temporal | true only at another point in time | `temporal_conflict` |
| Semantic | the entity's meaning has shifted | `semantic_conflict` |

**Task.** Reframe the multiple-choice QA into open triple adjudication: build an evidence
bundle (correct + N conflicting passages, shuffled, source-tagged), ask the model for the
single correct object, and score it against gold. Metrics: **accuracy**, **misled rate**
(fell for the planted value), **abstain rate**.

**DART transfer.** `telecom_inject.py` injects the same controlled conflicts into a clean
telecom KG (OSS/BSS vs field data) so the identical pipeline runs on telecom data with
perfect gold — no confidential data needed.

---

## 2. Codebase guide (file by file)

Data flow:

```
dataio.load_instances()      # stream ConflictBank records
      │
      ▼
dataio.build_bundle(inst)    # assemble the evidence bundle the agent sees
      │
      ▼
resolvers.<method>(bundle)   # adjudicate -> {"object", "chosen_source", "rationale"}
      │      └─ every model call goes through llm.chat()  (logging, retries, provider routing)
      ▼
metrics.score_instance()     # correct / misled / abstained vs gold
      │
      ▼
results/scores_<tag>.csv     # one row per instance   ── plotted by ──►  plot_results.py
results/calls_<tag>.csv      # one row per LLM call
```

### `config.py` — every knob
- `HF_DATASET = "Warrieryes/CB_qa"`, `USE_LOCAL_SAMPLE` (False = stream HF; True = offline
  `sample_data.jsonl`), `N_INSTANCES = 100`.
- `CONFLICT_TYPES`, `N_CONFLICT_SOURCES` (how many conflicting passages per bundle),
  `INCLUDE_CORRECT_EVIDENCE`, `SHUFFLE_SOURCES`, `TAG_SOURCE_METADATA`, `ADD_SUBJECT_CONTEXT`.
- `MODELS` — the model-tier axis (dict of tier-key → OpenRouter slug). All calls go through
  OpenRouter.
- `RESOLVERS` — which method(s) to run. **Must be valid resolver names** (see below).
- `TEMPERATURE = 0.0`, `MAX_TOKENS = 2000`, `SEED = 7`, `ALLOW_ABSTAIN = False`.

### `dataio.py` — data loading + bundle assembly
- `load_instances()` — streams CB_qa, renames columns to our schema
  (`default_evidence`→`correct_evidence`, `misinformation_conflict_evidence_evidence`→
  `fact_conflict_evidence`; `temporal_conflict_evidence` / `semantic_conflict_evidence`
  arrive natively).
- `build_bundle(inst, rng)` — correct passage + `N_CONFLICT_SOURCES` conflicting passages
  drawn from `CONFLICT_TYPES`, shuffled. `N_CONFLICT_SOURCES=0` ⇒ correct-only (control).
- `query_phrase()`, `gold_object()`, `render_bundle()`.
- `_SOURCE_OF` tags correct+fact passages `wikipedia`, temporal+semantic `news`.
  ⚠️ **Known leak** — a model that blindly "trusts wikipedia" can get temporal/semantic
  right without reasoning. Audit before claiming high accuracy.

### `llm.py` — the single logged call site (with resilience)
- `chat(...)` — every model request. Direct OpenRouter HTTP call; logs tokens/$/latency.
- **Retries** transient errors (429/5xx/network + spurious 400s) with exponential backoff
  (`MAX_ATTEMPTS=4`); **auth errors fail fast**; after retries it **skips one instance**
  (returns `""`) rather than aborting the whole run.
- `OPENROUTER_IGNORE_PROVIDERS = ["Novita"]` — Novita returns `400 does not support
  endpoint: completions` for some models; skipping it lets a rate-limited primary
  (DeepInfra) back off cleanly instead of falling through to a broken provider.

### `resolvers.py` — the "method" axis
Registry: `majority_vote`, `recency`, `source_trust`, `llm_judge`, `llm_judge_provenance`,
`multi_agent_debate`. **`llm_judge` is the headline.** A resolver returns
`{"object", "chosen_source", "rationale"}` and **never sees gold** (that keeps it honest).
- `_forced_object()` — with `ALLOW_ABSTAIN=False`, recovers the answer even from malformed
  JSON so a parse slip doesn't become a false abstain (genuine null/blank stays `None`).
- `multi_agent_debate` — two-round advocate/rebuttal + judge (2N+1 calls). **In practice it
  did WORSE than `llm_judge`** (see results) — the extra reasoning misleads more than it helps.

### `metrics.py` — deterministic scoring
- `match(a, b)` — equality after `_canon()`: drops an appended UPPERCASE acronym (`(MIT)`)
  and a leading article (`the`), then strips punctuation. Recovers `MIT` =
  `Massachusetts Institute of Technology` and `The Azrieli…` = `Azrieli…` **without**
  substring-matching false positives. Exact, not fuzzy.
- `score_instance()` → `correct / misled / abstained`. `aggregate()` → per-conflict-type.

### `run_experiment.py` — the driver
- `run(models, resolvers, out_tag, conflict_type, control)`.
- `--conflict-type {fact,temporal,semantic}` — bundle = correct + **one** conflict of that
  type; rows tagged with it (default is a `mixed` bundle of 2 types, tagged `mixed`).
- `--control` — correct-evidence-only "knowledge probe" (`N_CONFLICT_SOURCES=0`), tagged
  `control`. Measures what the model *knows* conflict-free.
- Writes `results/scores_<tag>.csv` and `results/calls_<tag>.csv`.

### `plot_results.py` — visualization + reporting
Reads every `results/scores_*.csv`. Prints, then plots:
- **Data-quality report** — flags any model×type cell with >10% blanks (= failed calls; with
  abstain forced off, a blank means a dropped call) and **auto-excludes it from the charts**.
- **Knowledge-filtered report** — for each model with a `control` run, accuracy on conflict
  runs graded **only on the facts it got right conflict-free**.
- `accuracy_<resolver>.png` — accuracy by model (per resolver). `control` excluded.
- `by_conflict_type_<resolver>.png` — grouped bars per model split by conflict type
  (Misinformation/Temporal/Semantic), Wilson 95% CIs.

### `telecom_inject.py` — DART transfer (inject conflicts into a clean telecom KG).
### `sample_data.jsonl` — tiny offline sample (`USE_LOCAL_SAMPLE=True`).
### `~/run_dart.sh` — convenience wrapper (lives in home dir, **not** the repo):
```
~/run_dart.sh <model-key> <output-tag> [conflict-type|control]
#   2 args -> mixed bundle, all 3 resolvers
#   3rd arg = fact_conflict|temporal_conflict|semantic_conflict -> single-type, llm_judge
#   3rd arg = control -> correct-evidence-only knowledge probe, llm_judge
```

---

## 3. What changed this session

| File | Change | Why |
|---|---|---|
| `run_experiment.py` | `--conflict-type` flag | Data was always `mixed` (2 blended conflicts) — impossible to break down by type. Now single-type runs are tagged. |
| `run_experiment.py` | `--control` flag | Knowledge probe (correct-only) → enables the knowledge filter. |
| `metrics.py` | `_canon()` in `match()` | Strict exact-match penalised correct-but-differently-worded answers. Canonical recovers acronyms/articles **safely** (audited: 0 false positives, 0 misled newly credited). |
| `resolvers.py` | Prompt: clean-answer + conflict-type guidance | Model emitted sentences (broke scoring) and mishandled temporal/semantic. Now asked for the canonical name only + told to ignore outdated/different-sense passages. |
| `resolvers.py` | `_forced_object()` | With abstain off, a malformed-JSON reply became a false abstain. Now the answer is recovered; only genuine non-answers stay blank. |
| `resolvers.py` | `llm_judge_sc` (self-consistency resolver) | Attempt to push accuracy up by sampling 5× + majority vote. **Tested → it hurt** (Gemini 3 Flash 87→73); registered for study, not used. |
| `llm.py` | Retry/backoff, error-body surfacing, skip-on-fail, `OPENROUTER_IGNORE_PROVIDERS` | A single provider blip (rate-limit → broken Novita fallback → 400) aborted whole runs. Now resilient; one bad call skips one instance. |
| `config.py` | `MAX_TOKENS 400 → 2000` | **Critical.** Reasoning models (Gemini 2.5 Pro, DeepSeek R1, o-series) burned the 400-token budget *thinking* and were truncated before answering → false abstains. Non-reasoning models stop early, so cost/output unchanged for them. |
| `config.py` | Added frontier-Western + reasoning models | Supervisor wants frontier Western models: `frontier-gpt5`, `frontier-sonnet45`, `frontier-opus45`, `frontier-mistral`, `reason-o4mini`, `reason-o3` (all verified live, accept `temperature=0`). |
| `plot_results.py` | Model map expanded, reasoning tier, per-conflict-type chart, **knowledge-filter chart (raw vs facts-it-knows)**, data-quality report (auto-exclude bad cells), knowledge-filtered report, control excluded from charts | Support new models; make the per-type + knowledge-filter graphs; **prevent corrupted runs from reaching a slide**. |
| `~/run_dart.sh` | optional 3rd arg (conflict-type or `control`) | one-line invocation for single-type and control runs. |

**⚠️ Config to fix:** `config.py` currently has `RESOLVERS = ["llm_equivalence"]`, which is
**not a valid resolver** — a bare `python run_experiment.py` will crash. Set it back to
`RESOLVERS = ["llm_judge"]`. (The `~/run_dart.sh` wrapper passes `--resolvers llm_judge`
explicitly, so wrapper runs are unaffected.)

**🔒 Security:** an OpenRouter API key was found exposed in a mangled directory name in the
repo tree (bracketed-paste accident). It should be **rotated** and kept only in `.env`
(git-ignored). Do not commit `.env`.

---

## 4. Results (current, honest)

All numbers are `llm_judge`, n=100 per cell unless noted, canonical scoring.

### 4a. Mixed-bundle leaderboard (correct + **2** conflicts — the hardest condition)
| Model | Accuracy | Misled |
|---|---|---|
| **Gemini 3 Flash** | **72%** | 8% |
| Claude Sonnet 4 | 54% | 27% |
| Claude 3 Haiku | 41% | 36% |
| GPT-4o | 37% | 41% |
| Llama 3.3 70B | 35% | 37% |
| Qwen 2.5 7B | 32% | 52% |
| GPT-4o-mini | 27% | 50% |

### 4b. Per-conflict-type (correct + **1** conflict) — **the headline chart**
| Model | Misinformation | Temporal | Semantic |
|---|---|---|---|
| **Gemini 3 Flash** | **77%** | **77%** | **73%** |
| Gemini 2.5 Pro | 68% | 74% | 65% |
| Qwen 2.5 72B | 56% | 55% | 60% |
| Gemma 3 4B | 54% | ⚠️ *corrupt* | 56% |

### 4b½. The knowledge filter, in plain English

To test whether a model can **resist a lie**, you first have to check whether it even knows
the truth. If it never knew the answer, getting it wrong isn't "falling for the conflict" —
it just didn't know.

So we run each model **twice** on every fact:

1. **Control run — "does it even know this?"** We show the model the correct evidence with
   **no conflict at all** and ask the question. If it answers correctly, it *knows* this
   fact. If it's wrong even here, it *never knew it*.
2. **Conflict run — "can it resist the lie?"** We show the correct evidence **plus a planted
   conflict**, and ask again.

Then the **knowledge filter** grades the conflict run **only on the facts the model got right
in step 1**. A fact it never knew can't count as a conflict-resolution failure.

> **Example — "What is the capital of Australia?" (truth: Canberra)**
> - *Control:* model says "Canberra" → it **knows** this one → keep it.
> - *Conflict (+ planted "the capital is Sydney"):* model still says "Canberra" → **resisted** ✅
> - A *different* fact where the control answer is already wrong → the model **never knew it**,
>   so we don't hold the conflict run's mistake against it.

Gemini 3 Flash knows **76%** of these facts. Graded on those, it beats the planted conflict
93–96% of the time. Always report it with the qualifier: *"on the facts the model knows."*

### 4c. Knowledge-filtered (graded only on facts the model knows) — **the 90s result**
Interpretation: *"when the model knows the fact, how often does it resist the conflict?"* —
the right measure of conflict-resolution skill, separated from base knowledge. Requires a
`--control` run per model. **This is where the headline 90s numbers live — never on mixed.**

| Model | Knows | Misinformation (raw→filtered) | Temporal | Semantic |
|---|---|---|---|---|
| **Gemini 3 Flash** | 76% | 77% → **93%** | 77% → **96%** | 73% → **95%** |
| Qwen 2.5 72B | 81% | 56% → 67% | 55% → 65% | 60% → 73% |

**Always report the pair** (raw + filtered) and state "graded on the N% of facts the model
knows" — the filtered number alone, unqualified, would overstate.

### 4d. Other resolvers (why `llm_judge` is the headline)
- `multi_agent_debate` did **worse** across the board (e.g. Gemini 3 Flash 50% vs 72%,
  Haiku 21% vs 41%) — the debate misleads more than it helps.
- `llm_judge_provenance` ≈ `llm_judge` (asking for the trusted source barely moves accuracy).
- `llm_judge_sc` (self-consistency: sample 5× at temp 0.5, majority-vote) — **tested and it
  HURT** (Gemini 3 Flash fact 87%→73% on a 15-item probe). At `temperature=0` the model
  already gives its best confident answer; sampling injects noise for this decisive task.
  The resolver is registered for study but **not** used for the headline numbers. The 90s
  come from the knowledge filter, not self-consistency.

---

## 5. Honesty & caveats — read before making slides

1. **Mixed vs per-type are different conditions — never compare them directly.** Mixed has
   two simultaneous conflicts and is *much* harder. Controlled test (same model, same
   instances, gpt-4o-mini): mixed **27%** vs fact 57% / temporal 43% / semantic 60%.
2. **You will not get 90% on *mixed*** from any honest change — it is the hardest setting.
   The 90s come from **best model × per-type × knowledge filter**, reported as *"resolution
   accuracy on facts the model knows, single conflict."*
3. **A "96% mixed" figure was a discarded probe.** It used naive substring/"containment"
   matching (n=25) which over-credits ~13% of flips (e.g. scores "John Entwistle" = "The
   Who"). **Not implemented, not defensible — do not use it.** The implemented canonical
   scorer adds only ~+1 on mixed (72→73), by design.
4. **Reasoning models require `MAX_TOKENS ≥ 2000`.** At 400 they truncate before answering
   → false abstains. Gemini 2.5 Pro read 37/47/53 (truncated) → **68/74/65** after the fix.
5. **Gemma 3 4B temporal is corrupt** (51/100 calls failed a provider rate-limit → blanks).
   Excluded from the chart automatically; **re-run before using**.
6. **Misled-rate is only reliable for `fact_conflict`** (the planted wrong value is only
   cleanly recoverable there). Treat temporal/semantic misled numbers as approximate.
7. **Knowledge ceiling ≈ 88%** (Gemini 3 Flash, no-conflict). No model knows ~10–12% of
   these facts regardless of conflict — that's why full-set strict 95% isn't achievable.
8. **Source-tag leak** (`dataio._SOURCE_OF`) may already help temporal/semantic — audit it.

---

## 6. Plots (in `results/plots/`)
| File | What it shows | Use for slide? |
|---|---|---|
| `knowledge_filter_gemini-3-flash.png` | Raw vs "facts it knows" per conflict type — the **93/96/95** story | ✅ **headline** |
| `knowledge_filter_qwen-2.5-72b.png` | Same, for Qwen (67/65/73) | supporting |
| `by_conflict_type_llm_judge.png` | Raw accuracy per model split by conflict type (95% CI) | ✅ headline |
| `accuracy_llm_judge.png` | Accuracy by model | ⚠️ pools mixed + per-type conditions — caption carefully |
| `accuracy_llm_judge_provenance.png` | provenance resolver | secondary |
| `accuracy_multi_agent_debate.png` | debate resolver (shows it underperforms) | supporting point |

**Regenerate all plots + reports** (this is the plotting command):
```bash
"/home/orangefull/Documents/WP2 DART/.venv/bin/python" "/home/orangefull/Documents/WP2 DART/plot_results.py"
```
It reprints the data-quality report, the knowledge-filtered report, and the per-model
summary, and rewrites every PNG in `results/plots/`.

---

## 7. How to run experiments
```bash
# key first (rotated key, in .env):   OPENROUTER_API_KEY=sk-or-...
cd "/home/orangefull/Documents/WP2 DART"

# single conflict type for one model:
~/run_dart.sh small-gemflash  gemflash_fact  fact_conflict

# knowledge probe (control) for one model:
~/run_dart.sh small-gemflash  gemflash_control  control

# full per-type + control sweep for a model:
for ct in fact_conflict temporal_conflict semantic_conflict; do
  ~/run_dart.sh mid-qwen72b  qwen72b_$ct  $ct
done
~/run_dart.sh mid-qwen72b  qwen72b_control  control

# then regenerate plots (see §6)
```
Notes: runs are deterministic (`temperature=0`, fixed seed) — re-running the same
model×type gives the same number. Reasoning models (Gemini 2.5 Pro, GPT-5, o3/o4-mini, R1)
are slow (~20–40 s/instance) and cost more.

---

## 8. Open items / next steps
- [x] Control runs for Qwen 2.5 72B and **Gemini 3 Flash** → knowledge-filtered numbers (§4c).
- [ ] Fix `config.RESOLVERS` back to `["llm_judge"]`.
- [ ] Re-run **Gemma 3 4B temporal** (corrupt — 51/100 failed calls).
- [ ] Control + per-type for **Gemini 2.5 Pro** and a top frontier model (GPT-5 / Opus) to
      see if any beats Gemini 3 Flash's knowledge-filtered 93–96%.
- [ ] Rotate the exposed OpenRouter key.
- [ ] (Optional) frontier-Western sweep: `frontier-gpt5`, `frontier-sonnet45`,
      `frontier-opus45`, `frontier-mistral`, `reason-o4mini`, `reason-o3`.

**One-line story for the deck:** *Gemini 3 Flash is the strongest conflict resolver. On the
facts it actually knows, it resists planted knowledge conflicts **93% (misinformation), 96%
(temporal), 95% (semantic)** — single conflict, per type. Raw accuracy (all facts, incl. ones
it never knew) is 77/77/73; the gap is base knowledge, not resolution skill. Two conflicts at
once (mixed) is much harder for every model (Gemini 3 Flash 72%). Self-consistency and
multi-agent debate did not help; the wins came from a clean-answer prompt, honest canonical
scoring, and the knowledge filter.*
