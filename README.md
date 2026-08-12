# WP2 DART — conflict resolution in agentic KG construction

When a knowledge graph is built from many sources, those sources **disagree**. Given a
`(subject, relation)` and several conflicting evidence passages — one correct, the rest
planted conflicts — can an LLM agent adjudicate the true value, and which models resist
being misled? Scoped to **ConflictBank**, with a synthetic **telecom conflict-injection**
transfer path for DART (OSS/BSS vs field data).

Why it's novel: KARMA has a conflict-resolution *agent* and Graphusion a fusion *module*,
but neither ships a public benchmark that scores conflict resolution during construction.
This repo reframes ConflictBank (a QA benchmark) into a KG-fusion task.

## Setup
```bash
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-...' > .env    # every model is called through OpenRouter
```
`.env` is git-ignored — never commit a key. The bundled `sample_data.jsonl` lets the
notebook and `--dry-run` work offline (`USE_LOCAL_SAMPLE=True` in `config.py`); real runs
stream ConflictBank from HuggingFace (`Warrieryes/CB_qa`).

## The two conditions
There are exactly two, and one command runs both:

| Condition | Bundle | Question it answers |
|---|---|---|
| `control` | correct evidence only | Does the model even **know** this fact? |
| `mixed` | correct evidence + `N_CONFLICT_SOURCES` planted conflicts, shuffled | Can it **resist** the conflict? |

The **knowledge filter** grades the conflict run only on the facts the model got right in
its own control run — a fact it never knew is not a conflict-resolution failure. It is
applied at *scoring* time: the mixed run covers every instance and each row carries a
`known` flag, so one file yields both the raw and the filtered accuracy. **Always report
the pair** ("74% overall; 92% on the 73% of facts it knows") — the filtered number alone
overstates.

Per-conflict-type runs (fact / temporal / semantic separately) were retired; `mixed` draws
its conflicts at random from all three types.

## First 15 minutes
1. `jupyter notebook data_vis.ipynb` — the task, the three conflict types, an evidence bundle (free).
2. `python run_experiment.py --dry-run` — load data + print one bundle, no API calls.
3. `python capture_prompt.py` — see the exact prompt a resolver sends (also free).
4. `python run_experiment.py --models small-gemflash --out gemflash -n 20` — a real, cheap run.

## Running experiments
```bash
python run_experiment.py --models small-gemflash --out gemflash    # control + mixed
python run_experiment.py --models frontier-opus5 --control-only    # knowledge probe only
python run_experiment.py --models frontier-opus5 --no-control      # conflict run only (no `known` flag)
python run_experiment.py --models small-gemflash --out smoke -n 10 # smoke test

python rescore.py --judge      # re-score every results file under one set of rules
python plot_results.py         # reports + every PNG in results/plots/
```
Useful flags: `-n/--n-instances`, `--resolvers`, `--reasoning-effort minimal` (stops a
thinking model burning its whole budget and truncating into a false abstain),
`--max-tokens`, `--no-judge` (strict string matching only — runs a point or two low).

Runs are deterministic (`temperature=0`, fixed seed): the same model × condition gives the
same number. Reasoning models are slow (~20–40 s/instance) and cost more.

## Files
| File | What it is |
|---|---|
| `data_vis.ipynb` | **Start here.** The task, the conflict types, evidence bundles, scoring, the telecom transfer. |
| `config.py` | **Every knob.** Data source, conflict composition, models, resolvers, token/reasoning budgets, priors. |
| `dataio.py` | Stream ConflictBank; assemble the shuffled, source-tagged evidence bundle; gold object. |
| `llm.py` | The single logged call site — tokens/$/latency, retries with backoff, skip-one-instance on failure. |
| `resolvers.py` | The methods: `majority_vote`, `recency`, `source_trust`, `llm_judge` (**the headline**), `llm_judge_provenance`, `multi_agent_debate`, `llm_judge_sc`. A resolver never sees gold. |
| `metrics.py` | Deterministic scoring: correct / misled / abstained, Wilson CIs, canonical (not fuzzy) matching. |
| `run_experiment.py` | The driver → `results/{scores,calls}_<tag>{,_control}.csv`. |
| `rescore.py` | Re-score stored `(pred, gold)` pairs under one rule set; `--judge` adds a cached equivalence pass (sees only the two strings, never the evidence). |
| `plot_results.py` | Data-quality report (auto-excludes cells with >10% blanks), knowledge-filtered report, charts. |
| `capture_prompt.py` | Print the exact prompt a resolver builds — no API call, no spend. |
| `telecom_inject.py` | **DART transfer.** Inject controlled conflicts into a clean telecom KG → same pipeline, perfect gold. |
| `SESSION_HANDOFF.md` | Full brief: results, methodology caveats, what changed and why. **Read before making a slide.** |
| `PIPELINE.md` / `PIPELINE_DIAGRAM.md` / `Task2_Brief.md` | Pipeline write-up, diagram, original brief. |

## Output
- `results/scores_<tag>.csv` — one row per instance: `correct` / `misled` / `abstained`, plus
  `known` (the knowledge filter) and `judged`. **This is the headline result.**
- `results/scores_<tag>_control.csv` — the knowledge probe.
- `results/calls_<tag>*.csv` — one row per LLM call. Cost/latency are logged for reference;
  they are **not** a study axis — the focus is resolution correctness.
- `results/plots/*.png` — regenerated by `plot_results.py`.

## Current numbers (`llm_judge`, post-leak-fix runs of 2026-08-12)
| Model | Knows (control) | Mixed accuracy | Knowledge-filtered | Misled |
|---|---|---|---|---|
| Claude Opus 5 | 81% | 88% | **98%** (n=81) | 1% |
| GPT-5.6 Terra | 73% | 74% | 92% (n=73) | — |
| Claude Opus 4.5 | 82% | 73% | 87% (n=233) | 8% |
| Gemini 3 Flash | 79% | 73% | — | 18% |

n=100 unless noted. ⚠️ **The 93/96/95 figures in `SESSION_HANDOFF.md` §4b/§4c are
superseded** — they were inflated by a source-tag leak and a timestamp leak, both since
fixed (`dataio` now uses ConflictBank's real per-record category; `SHOW_TIMESTAMPS=False`).
Do not put them on a slide.

## Notes
- **API-only, no GPU.** Dry-run and prompt capture are free.
- **ConflictBank** (arXiv 2408.12076, NeurIPS 2024 D&B): 7.45M claim-evidence pairs,
  Wikidata top-100 relations, 3 conflict types, CC BY-SA 4.0. ⚠️ Verify the HF license +
  full-set assembly before publishing (the HF card omits the license; the visible config is a shard).
- **Scoring is exact, not fuzzy.** A discarded probe used substring matching and scored
  "John Entwistle" = "The Who". Canonical matching plus the equivalence judge replaced it;
  neither over-credits whole-vs-part.
- `DROP_UNANSWERABLE` skips instances whose gold is a Wikidata artifact rather than an
  answer. The list is fixed, declared, and applied identically for every model before any
  call — not a post-hoc drop.
- Validate on real conflicts too: **CONFLICTS** (2506.08500, Apache-2.0) and
  **WikiContradict** (2406.13805).
