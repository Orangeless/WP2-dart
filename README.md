# T2 starter kit — conflict resolution in agentic KG construction

A ready-to-run framework for the T2/T4 direction: when multiple sources **disagree**
while building a KG, how should an agent adjudicate? Scoped to **ConflictBank**, with a
synthetic **telecom conflict-injection** transfer path for DART (OSS/BSS vs field data).

Why it's novel: KARMA has a conflict-resolution *agent* and Graphusion has a fusion
*module*, but neither ships a public benchmark that scores conflict resolution during
construction. This kit reframes ConflictBank (a QA benchmark) into a KG-fusion task.

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...          # only for paid runs, not the dry-run
```
The bundled `sample_data.jsonl` lets the notebook + dry-run work with no download. For
real runs, load ConflictBank from HuggingFace (`Warrieryes/CB_claim_evidence`) — set
`USE_LOCAL_SAMPLE=False` in `config.py`.

## First 15 minutes
1. `jupyter notebook data_vis.ipynb` — see the task, the conflict types, an evidence bundle (free).
2. `python run_experiment.py --dry-run` — load data + print one bundle, no API calls.
3. `python run_experiment.py` — run `llm_judge` on `gpt-4o-mini` over the sample (a few cents).
4. Read `EXPLORE.md`, pick a lane.

## Files
| File | What it is |
|---|---|
| `data_vis.ipynb` | **Start here.** The task, the three conflict types, evidence bundles, scoring, the telecom transfer. |
| `config.py` | **Every knob.** Data source, conflict types, bundle composition, models, resolvers, priors. |
| `dataio.py` | Load ConflictBank instances; assemble the shuffled evidence bundle; gold object. |
| `llm.py` | The single logged LLM call site (tokens/$/latency). |
| `resolvers.py` | The methods: `majority_vote`, `recency`, `source_trust`, `llm_judge`, `llm_judge_provenance`, `multi_agent_debate`. |
| `metrics.py` | Deterministic scoring: correct / misled / abstained, per conflict type. |
| `run_experiment.py` | Driver → `results/{scores.csv, calls.csv}`. |
| `telecom_inject.py` | **DART transfer.** Inject controlled conflicts into a clean telecom KG → same pipeline, perfect gold. |
| `EXPLORE.md` | **The research questions** and where to experiment. |

## Output
- `results/scores.csv` — one row per instance: correct/misled/abstained + conflict_type. **This is the headline result** (broken down by conflict type in `metrics.aggregate`).
- `results/calls.csv` — one row per LLM call. (Cost/latency are logged here for reference; they are **not** a study axis in T2 — the focus is resolution correctness.)

## Notes
- **API-only, no GPU.** Dry-run is free.
- **ConflictBank** (arXiv 2408.12076, NeurIPS 2024 D&B): 7.45M claim-evidence pairs, Wikidata top-100 relations, 3 conflict types, CC BY-SA 4.0. ⚠️ Verify HF license + full-set assembly before publishing (the HF card omits the license; visible config is a shard).
- Validate on real conflicts too: **CONFLICTS** (2506.08500, Apache-2.0) and **WikiContradict** (2406.13805) — see EXPLORE.md Lane/guardrail 4.
# wp2
