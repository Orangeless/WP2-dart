# Task 2 — Conflict Resolution in Agentic KG Construction

## Overview

When a knowledge graph (KG) is built from many sources, those sources disagree: OSS
inventory vs. live topology vs. field tickets can each assert something different about
the *same* fact.

- **Research question:** When sources conflict, how should an agent decide the correct
  triple — and which resolution strategy resists being misled?
- **Gap:** KARMA has a conflict-resolution agent and Graphusion has a fusion module, but
  neither ships a *public benchmark* that scores conflict resolution *during construction*.
- **Goal:** Reframe a QA conflict benchmark into a KG-fusion task; compare resolution
  strategies on **resolution accuracy**, **misled rate**, and **provenance**.
- **DART transfer:** Inject controlled conflicts into a clean telecom KG (perfect gold),
  then resolve OSS/BSS-vs-field disagreements with no confidential data.

---

## Benchmark and Task

**Dataset:** ConflictBank (arXiv 2408.12076, NeurIPS 2024) — 7.45M claim–evidence pairs
from Wikidata. Three conflict types:

| Conflict type   | Meaning                                   |
|-----------------|-------------------------------------------|
| Misinformation  | Wrong value                               |
| Temporal        | True only at another point in time        |
| Semantic        | Entity meaning has shifted                |

**Task:** Given a `(subject, relation)` and several *disagreeing* evidence passages,
output the adjudicated triple **plus the source trusted** — or **abstain**.

**Quality metrics** (deterministic — no LLM judge):

- Accuracy (matched the gold object)
- Misled rate (fell for the planted conflict)
- Abstain rate + provenance correctness

**Study axes:** resolution strategy × model tier; broken down by conflict type.

**Validation on real conflicts (optional):**
- CONFLICTS (arXiv 2506.08500)
- WikiContradict (arXiv 2406.13805)

---

## Code Structure — `kick_off/wp1/d4_starter/`

| File                | Purpose                                                                                          |
|---------------------|--------------------------------------------------------------------------------------------------|
| `data_vis.ipynb`    | Guided tour: the task, the three conflict types, evidence bundles, scoring, the telecom transfer.|
| `config.py`         | All knobs: data source, conflict types, bundle composition, models, resolvers, trust priors.     |
| `dataio.py`         | Load ConflictBank; assemble the shuffled, unlabelled evidence bundle; gold object.               |
| `llm.py`            | The single logged LLM call site (records tokens / $ / latency).                                  |
| `resolvers.py`      | Five methods: `majority_vote`, `recency`, `source_trust`, `llm_judge`, `multi_agent_debate`.     |
| `metrics.py`        | Deterministic scoring: correct / misled / abstained, per conflict type.                          |
| `run_experiment.py` | Driver → `results/` (`scores.csv`: each instance's outcome by conflict type).                    |
| `telecom_inject.py` | DART transfer: inject controlled conflicts into a clean telecom KG (perfect gold).               |

---

## References & Platforms

> ★ = start here (entry-level). All arXiv IDs and repos verified July 2026.

### The benchmark (read first)

- **★ Su et al.** *ConflictBank: A Benchmark for Evaluating the Influence of Knowledge
  Conflicts in LLMs.* NeurIPS 2024 D&B. arXiv:2408.12076 —
  `github.com/zhaochen0110/conflictbank`
  (data: HF `Warrieryes/CB_claim_evidence`, CC BY-SA 4.0).

### Conflict resolution during KG construction

- **★ Lu et al.** *KARMA: Multi-Agent LLMs for KG Enrichment* (has a Conflict-Resolution
  Agent). NeurIPS 2025. arXiv:2502.06472 — `github.com/YuxingLu613/KARMA`.
- **Yang et al.** *Graphusion: Global KG Fusion* (entity merge + conflict resolution).
  NLP4KGC @ WWW 2025. arXiv:2410.17600 — data: HF `li-lab/tutorqa`.

### Methods & real-conflict validation sets

- **Wang et al.** *RAMDocs + MADAM-RAG: multi-agent debate over conflicting documents.*
  COLM 2025. arXiv:2504.13079 — `github.com/HanNight/RAMDocs`.
- **DRAGged into Conflicts (CONFLICTS)** — real retrieved sources, expert conflict-type
  labels. 2025. arXiv:2506.08500 — `github.com/google-research-datasets/rag_conflicts`
  (Apache-2.0).
- **WikiContradict** — real editor-flagged Wikipedia contradictions. NeurIPS 2024 D&B.
  arXiv:2406.13805 — HF `ibm-research/Wikipedia_contradict_benchmark`.

### Platforms & tools

- **LiteLLM** — unified API for OpenAI / Anthropic / OpenRouter: `docs.litellm.ai`.
- **OpenAI API** (`gpt-4o-mini`) · **OpenRouter** (open-weight models): `openrouter.ai` ·
  **Hugging Face Datasets** (ConflictBank hosting).
