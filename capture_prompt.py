"""
capture_prompt.py — print the exact string a resolver sends to the model.

Monkeypatches llm.chat so nothing is actually called and nothing is spent, then runs
the real resolver over a real bundle and prints the prompt it built.

    python capture_prompt.py                  # first sample record, llm_judge
    python capture_prompt.py --index 1        # a different record
    python capture_prompt.py --resolver llm_judge_provenance
    python capture_prompt.py --conflict-type temporal_conflict
    python capture_prompt.py --hf             # use the real CB_qa data instead of the sample
"""
import argparse, json, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--resolver", default="llm_judge")
ap.add_argument("--conflict-type", default=None,
                choices=["fact_conflict", "temporal_conflict", "semantic_conflict"])
ap.add_argument("--n-conflicts", type=int, default=None)
ap.add_argument("--hf", action="store_true", help="stream the real CB_qa instead of sample_data.jsonl")
args = ap.parse_args()

import config
config.USE_LOCAL_SAMPLE = not args.hf
if args.conflict_type:
    config.CONFLICT_TYPES = [args.conflict_type]
    config.N_CONFLICT_SOURCES = 1
if args.n_conflicts is not None:
    config.N_CONFLICT_SOURCES = args.n_conflicts

import llm
CAPTURED = []
def fake_chat(prompt, model, **kw):
    CAPTURED.append((prompt, kw))
    return '{"object": "<call intercepted, nothing sent>"}'
llm.chat = fake_chat

import resolvers, dataio, run_experiment
resolvers.chat = fake_chat

insts = dataio.load_instances()
if args.conflict_type:
    insts = [i for i in insts if i.get(args.conflict_type + "_evidence")]
inst = insts[args.index]

bundle = dataio.build_bundle(inst, random.Random(config.SEED + args.index))
query = dataio.query_phrase(inst)

print("=" * 78, "\nCONFIG\n", "=" * 78, sep="")
for k in ("USE_LOCAL_SAMPLE", "CONFLICT_TYPES", "INCLUDE_CORRECT_EVIDENCE",
          "N_CONFLICT_SOURCES", "SHUFFLE_SOURCES", "TAG_SOURCE_METADATA",
          "ADD_SUBJECT_CONTEXT", "ALLOW_ABSTAIN", "TEMPERATURE", "MAX_TOKENS", "SEED"):
    print(f"  {k:26} = {getattr(config, k)}")

print("\n" + "=" * 78, "\nBUNDLE OBJECT  (return value of dataio.build_bundle)\n", "=" * 78, sep="")
print(json.dumps(bundle, indent=2))

print("\n" + "=" * 78, "\nHELD OUT, NEVER IN THE PROMPT\n", "=" * 78, sep="")
print("  gold_object     =", dataio.gold_object(inst))
print("  conflict_object =", run_experiment._conflict_object(inst))

resolvers.get_resolver(args.resolver)(
    inst.get("subject", ""), query, bundle,
    "openrouter/google/gemini-3-flash-preview", str(inst.get("id", inst.get("subject"))), [])

for n, (prompt, kw) in enumerate(CAPTURED, 1):
    print("\n" + "=" * 78,
          f"\nEXACT STRING SENT TO THE MODEL  [call {n} of {len(CAPTURED)}, "
          f"resolver={kw.get('resolver')}, stage={kw.get('stage')}]\n", "=" * 78, sep="")
    print(prompt)
print("\n" + "=" * 78)
print("call kwargs:", {k: v for k, v in CAPTURED[0][1].items() if k != "log"})
