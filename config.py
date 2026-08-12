"""
config.py — every knob of the D4 conflict-resolution experiment in one place.

D4 task (conflict-aware KG construction): given a (subject, relation) and several
evidence passages that DISAGREE about the object — some correct, some planted
conflicts — an agent must output the *adjudicated* triple (the object it believes)
and say which source it trusted. Gold = the true object. We measure resolution
accuracy, broken down by conflict type and by how the sources are mixed.

Data source: ConflictBank (arXiv 2408.12076, CC BY-SA 4.0). Each instance already
carries a correct evidence + a planted-conflict evidence + the gold answer, so we
do not have to build gold labels — we reframe its multiple-choice QA into
open triple adjudication.
"""
import os

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
# ConflictBank lives on HuggingFace. Loading the full 7.45M pairs is unnecessary;
# we take a small slice. A tiny bundled sample (sample_data.jsonl) lets the
# notebook + dry-run work with no download at all.
HF_DATASET   = "Warrieryes/CB_qa"               # merged records that match our instance schema
                                                # (CB_claim_evidence is raw one-evidence-per-row; wrong shape)
USE_LOCAL_SAMPLE = False                         # True = read sample_data.jsonl (offline)
SAMPLE_PATH  = os.path.join(os.path.dirname(__file__), "sample_data.jsonl")
N_INSTANCES  = 100                      # how many instances to run (standardized)

# Skip instances whose gold is an artifact of the Wikidata relation rather than an answer
# to the question ("filing" for patent records, French INSEE occupation codes, Wikipedia
# project pages). A 300-instance control run scored 0/16 on these — they are denominator
# only, and they drag the knowledge ceiling down without measuring knowledge. The rule is
# a fixed, declared list in dataio._UNANSWERABLE_GOLD, applied before any model sees the
# data and identically for every model — not a post-hoc drop of instances anything failed.
DROP_UNANSWERABLE = True

# ----------------------------------------------------------------------------
# 2. WHICH CONFLICTS TO STUDY  (ConflictBank's three types)
# ----------------------------------------------------------------------------
# There are exactly TWO conditions in the experiment, and run_experiment.py runs both
# in one command:
#   control — correct evidence only (N_CONFLICT_SOURCES forced to 0). The knowledge
#             probe: does the model even know this fact?
#   mixed   — correct evidence + N_CONFLICT_SOURCES conflicting passage(s) drawn at
#             random from CONFLICT_TYPES below. The actual task.
# Splitting the conflict run out per type (fact / temporal / semantic separately) is
# deliberately not supported — add it back only on request.
CONFLICT_TYPES = ["fact_conflict", "temporal_conflict", "semantic_conflict"]

# Evidence-bundle composition — the "how sources disagree" axis:
INCLUDE_CORRECT_EVIDENCE = True   # include the one supported-by-truth passage?
N_CONFLICT_SOURCES       = 1      # how many conflicting passages to add (1..3).
                                  # 2 = the old, much harder 'mixed' setting; every
                                  # pre-2026-08 mixed number in the handoff used 2.
SHUFFLE_SOURCES          = True   # hide position cues (order must not leak the answer)
TAG_SOURCE_METADATA      = True   # attach the source type to passages

# Show the passage timestamp in the prompt?  OFF by design: only temporal-conflict
# passages have a date in ConflictBank, so rendering it marks the planted passage by
# presence alone (and 79/100 of those dates are in the FUTURE, which makes the planted
# passage trivially dismissible without any temporal reasoning).  The timestamp is still
# kept on the bundle so the `recency` baseline can use it.  Flip to True to measure the
# size of that leak.
SHOW_TIMESTAMPS          = False

# Neutral subject background, appended to every source. Uses the subject's own
# Wikidata description (`subject_description`) — written independently of the conflict,
# so it identifies WHO/WHAT the subject is without pointing at the correct OR the
# planted-wrong object. Never uses object/replaced/semantic descriptions (those leak).
ADD_SUBJECT_CONTEXT      = True

# ----------------------------------------------------------------------------
# 3. MODELS  (model-tier axis)
# ----------------------------------------------------------------------------
# Model strings should be provider-prefixed when you switch providers, for example:
#   - "openrouter/meta-llama/llama-3.2-3b-instruct"
#   - "openai/gpt-4o-mini"
#   - "anthropic/claude-3-5-sonnet-latest"
# If you are using a provider-specific key, set the matching environment variable.
MODELS = {
    # Small tier
    "small-gpt4omini":  "openrouter/openai/gpt-4o-mini",
    "small-haiku":      "openrouter/anthropic/claude-3-haiku",
    "small-qwen7b":     "openrouter/qwen/qwen-2.5-7b-instruct",
    "small-gemma4b":    "openrouter/google/gemma-3-4b-it",
    "small-gemflash":   "openrouter/google/gemini-3-flash-preview",
    # Mid tier
    "mid-llama70b":     "openrouter/meta-llama/llama-3.3-70b-instruct",
    "mid-qwen72b":      "openrouter/qwen/qwen-2.5-72b-instruct",
    # Frontier tier
    "frontier-sonnet":  "openrouter/anthropic/claude-sonnet-4",
    "frontier-gpt4o":   "openrouter/openai/gpt-4o",
    "frontier-gempro":  "openrouter/google/gemini-2.5-pro",
    "frontier-gemflash": "openrouter/google/gemini-3-flash-preview",
    # Frontier tier 
    "frontier-gpt5":    "openrouter/openai/gpt-5",                  # OpenAI (US)
    "frontier-sonnet45":"openrouter/anthropic/claude-sonnet-4.5",   # Anthropic (US)
    "frontier-opus45":  "openrouter/anthropic/claude-opus-4.5",     # Anthropic (US), top tier
    "frontier-mistral": "openrouter/mistralai/mistral-large-2512",  # Mistral (EU)
    # Reasoning
    "reason-r1":        "openrouter/deepseek/deepseek-r1",
    "reason-o4mini":    "openrouter/openai/o4-mini",                # OpenAI reasoning (US)
    "reason-o3":        "openrouter/openai/o3",                     # OpenAI reasoning (US)

    # -- 2026 generation (slugs verified live against the OpenRouter catalogue) --
    # The tiers above are 2024/25 models; comparing them with a 2026 model measures
    # release date as much as conflict-resolution skill. Prefer these for new sweeps.
    "frontier-qwen38max":  "openrouter/qwen/qwen3.8-max",           # Alibaba (CN), thinking
    # 3.8 ships only as -max; the cheaper non-max tiers stop at 3.7.
    "mid-qwen37plus":      "openrouter/qwen/qwen3.7-plus",
    "frontier-opus5":      "openrouter/anthropic/claude-opus-5",    # Anthropic (US)
    "frontier-sonnet5":    "openrouter/anthropic/claude-sonnet-5",  # Anthropic (US)
    "frontier-gpt56terra": "openrouter/openai/gpt-5.6-terra",       # OpenAI (US)
    "frontier-gem31pro":   "openrouter/google/gemini-3.1-pro-preview",
    # Same-generation small tier — the fair like-for-like comparison.
    "small-haiku45":       "openrouter/anthropic/claude-haiku-4.5",
    "small-gem36flash":    "openrouter/google/gemini-3.6-flash",
    "small-gpt54mini":     "openrouter/openai/gpt-5.4-mini",
    "small-qwen37flash":   "openrouter/qwen/qwen3.7-flash",
    # Open-weight — deployable on-prem, which is what the telecom transfer needs.
    "open-qwen35397b":     "openrouter/qwen/qwen3.5-397b-a17b",
    "open-glm52":          "openrouter/z-ai/glm-5.2",
    "open-deepseekv4pro":  "openrouter/deepseek/deepseek-v4-pro",
}

# ----------------------------------------------------------------------------
# 4. RESOLUTION STRATEGIES  (the "method" axis — see resolvers.py)
# ----------------------------------------------------------------------------
RESOLVERS = ["llm_judge"]

# Prior reliability of each source type, used by the "source_trust" resolver.
# (news < book < wikipedia is a starting guess — an intern should study this.)
SOURCE_TRUST_PRIOR = {"wikipedia": 0.9, "book": 0.6, "news": 0.5, "unknown": 0.5}

# ----------------------------------------------------------------------------
# 5. LLM CALL HYPER-PARAMETERS
# ----------------------------------------------------------------------------
TEMPERATURE = 0.0

# The budget must cover a thinking model's hidden chain-of-thought PLUS the final answer.
# Truncating mid-thought emits nothing, which scores as a (false) abstain — so this cap
# is a correctness setting, not a cost setting. Non-reasoning models stop after ~20-50
# tokens and never approach it, so raising it costs them nothing.
#   400   too low for any thinking model (Gemini 2.5 Pro read 37/47/53 -> 68/74/65 once
#         it could finish)
#   2000  enough for most; the default here
#   8000  needed by the heaviest thinkers (Qwen3.8 Max), which is better handled with
#         REASONING_EFFORT below than by paying for 8000 tokens on every instance
MAX_TOKENS  = 2000
REQUEST_TIMEOUT = 120.0   # seconds; a truncating thinking model can take 45s+

# Reasoning effort for thinking models: None leaves the model's default alone, otherwise
# "minimal" / "low" / "medium" / "high" (OpenRouter's unified parameter).
#
# Adjudicating a bundle is one decisive extraction, not a multi-step problem. Qwen3.8 Max
# burned 8000 tokens / 179 s / $0.05 on a single instance at its default; it rejects
# reasoning={"enabled": false} outright ("Reasoning is mandatory for this endpoint") but
# accepts effort="minimal". Comparing minimal vs default is a legitimate study axis, not
# only a cost knob.
REASONING_EFFORT = None
SEED        = 7
ALLOW_ABSTAIN = False   # may the resolver answer "cannot determine"? (studies precision/coverage)

# ----------------------------------------------------------------------------
# 6. OUTPUT
# ----------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 7. RE-SCORING  (see rescore.py)
# ----------------------------------------------------------------------------
# Model used by `rescore.py --judge` to decide whether a rejected answer actually
# denotes the same entity as gold. It only ever sees the two strings, never the
# evidence, so it cannot leak the answer into the task.
EQUIVALENCE_MODEL = "openrouter/google/gemini-3.1-flash-lite"

# ----------------------------------------------------------------------------
# 8. THE KNOWLEDGE FILTER
# ----------------------------------------------------------------------------
# Only grade on the instances the model demonstrably knows: if it cannot give the correct
# answer WITHOUT a conflict present, getting it wrong WITH one is not a conflict-resolution
# failure. That is what the `control` condition establishes.
#
# The filter is applied at SCORING time, not before the calls: the mixed run covers every
# instance and each row carries a `known` flag, so one file yields both the raw accuracy
# and the filtered accuracy. Report the PAIR — "83% overall; 91% on the 74% of facts the
# model knows" — because the filtered number alone, unqualified, overstates.