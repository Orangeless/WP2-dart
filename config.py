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

# ----------------------------------------------------------------------------
# 2. WHICH CONFLICTS TO STUDY  (ConflictBank's three types)
# ----------------------------------------------------------------------------
# For each instance we build an evidence bundle from these passages.
CONFLICT_TYPES = ["fact_conflict", "temporal_conflict", "semantic_conflict"]

# Evidence-bundle composition — the "how sources disagree" axis:
INCLUDE_CORRECT_EVIDENCE = True   # include the one supported-by-truth passage?
N_CONFLICT_SOURCES       = 2      # how many conflicting passages to add (1..3)
SHUFFLE_SOURCES          = True   # hide position cues (order must not leak the answer)
TAG_SOURCE_METADATA      = True   # attach source type + timestamp to passages

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
    # Reasoning
    "reason-r1":        "openrouter/deepseek/deepseek-r1",
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
MAX_TOKENS  = 400
SEED        = 7
ALLOW_ABSTAIN = False   # may the resolver answer "cannot determine"? (studies precision/coverage)

# ----------------------------------------------------------------------------
# 6. OUTPUT
# ----------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

#. 7 Only grade on the instances that we know the model "knows", if the mdoel does not know the correct answer without conflicts, how can it know it with conflicts.