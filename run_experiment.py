"""
run_experiment.py — driver for the D4 conflict-resolution study.

For every (resolver x model x instance): assemble the conflicting evidence bundle,
run the resolver, score the adjudicated object against gold, and log cost. Writes:

  results/scores_<tag>.csv   # one row per instance: correct/misled/abstained + conflict_type + $
  results/calls_<tag>.csv    # one row per LLM call (cost attribution)

Usage:
  python run_experiment.py                              # runs config.RESOLVERS x config.MODELS
  python run_experiment.py --dry-run                     # no API calls: check data loading + bundle assembly
  python run_experiment.py --models small-gpt4omini,small-haiku --resolvers llm_judge,llm_judge_provenance --out small_batch1
  python run_experiment.py --models frontier-sonnet --resolvers multi_agent_debate --out sonnet_debate

When --out is provided, output files are results/scores_<tag>.csv and results/calls_<tag>.csv
instead of the default results/scores.csv and results/calls.csv. This lets you run
multiple instances in parallel terminals without file collisions.
"""
import os, csv, json, random, argparse
from dotenv import load_dotenv
import config, dataio, metrics
from resolvers import get_resolver
from llm import CallLog

load_dotenv()  # load API keys from .env into the environment before any LLM/HF call


def _conflict_object(inst):
    """The planted wrong object, if we can recover it (for the 'misled' metric)."""
    opts, ri = inst.get("options"), inst.get("replaced_option")
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ri, ri)
    if opts and idx is not None and idx < len(opts):
        return opts[idx]
    return inst.get("replaced_object")


def run(models_override=None, resolvers_override=None, out_tag=None, conflict_type=None,
        control=False):
    # Determine which models and resolvers to use
    models = models_override if models_override else config.MODELS
    resolvers_list = resolvers_override if resolvers_override else config.RESOLVERS

    # Control mode: correct evidence ONLY, no conflicting passage. This measures what the
    # model KNOWS (its knowledge ceiling) so a later analysis can grade conflict runs on
    # just the facts it gets right conflict-free (see the knowledge filter in plot_results).
    #
    # Single-conflict-type mode: restrict every bundle to (correct + exactly one
    # conflicting passage of the requested type) and tag each row with that type, so
    # results can be broken down per conflict type instead of the default 'mixed'
    # bundle (which draws 2 of the 3 types and is therefore unlabelled).
    if control:
        config.N_CONFLICT_SOURCES = 0
    elif conflict_type:
        config.CONFLICT_TYPES = [conflict_type]
        config.N_CONFLICT_SOURCES = 1

    rng = random.Random(config.SEED)
    insts = dataio.load_instances()
    # When studying one conflict type, only score instances that actually carry that
    # conflict's evidence — others would be conflict-free and inflate accuracy.
    if conflict_type and not control:
        field = conflict_type + "_evidence"
        insts = [i for i in insts if i.get(field)]
        print(f"[conflict-type={conflict_type}] {len(insts)} instances carry this conflict")
    if control:
        print(f"[control] correct-evidence-only knowledge probe over {len(insts)} instances")
    call_log, rows = [], []

    # Pre-build bundles once per instance so every resolver sees the same evidence.
    # Each instance gets its own per-instance rng seeded from config.SEED + instance index,
    # so bundles are deterministic but not shared between instances.
    instance_bundles = {}
    for idx, inst in enumerate(insts):
        iid = str(inst.get("id", inst.get("subject", "?")))
        inst_rng = random.Random(config.SEED + idx)
        instance_bundles[iid] = dataio.build_bundle(inst, inst_rng)

    for tier, model in models.items():
        for rname in resolvers_list:
            resolver = get_resolver(rname)
            for inst in insts:
                iid = str(inst.get("id", inst.get("subject", "?")))
                subject = inst.get("subject", "")
                query = dataio.query_phrase(inst)
                bundle = instance_bundles[iid]
                pred = resolver(subject, query, bundle, model, iid, call_log)
                sc = metrics.score_instance(pred, dataio.gold_object(inst), _conflict_object(inst))
                ct = "control" if control else (conflict_type or inst.get("conflict_type", "") or "mixed")
                sc.update({"resolver": rname, "model_tier": tier, "model": model,
                           "inst_id": iid, "conflict_type": ct,
                           "n_sources": len(bundle), "chosen_source": pred.get("chosen_source")})
                rows.append(sc)
            print(f"[done] {rname} x {tier}: {len(insts)} instances")
    _write_calls(call_log, out_tag)
    _write_scores(rows, call_log, out_tag)
    print("\nAggregate:", json.dumps(metrics.aggregate(rows), indent=2))


def dry_run():
    """No API calls. Load data, assemble bundles, print one example + the label stats."""
    rng = random.Random(config.SEED)
    insts = dataio.load_instances()
    print(f"loaded {len(insts)} instances from "
          f"{'local sample' if config.USE_LOCAL_SAMPLE else config.HF_DATASET}\n")
    ex = insts[0]
    bundle = dataio.build_bundle(ex, rng)
    print(f"subject : {ex.get('subject')}")
    print(f"relation: {ex.get('relation')}")
    print(f"query   : {dataio.query_phrase(ex)}")
    print(f"gold    : {dataio.gold_object(ex)}")
    print(f"conflict: {_conflict_object(ex)}")
    print(f"\nEVIDENCE BUNDLE the agent must adjudicate ({len(bundle)} sources):")
    print(dataio.render_bundle(bundle))
    from collections import Counter
    print("\nconflict_type distribution:",
          dict(Counter(i.get("conflict_type", "mixed") for i in insts)))


def _write_calls(call_log, out_tag=None):
    fname = f"calls_{out_tag}.csv" if out_tag else "calls.csv"
    with open(os.path.join(config.RESULTS_DIR, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resolver", "stage", "model", "inst_id", "prompt_tokens",
                    "completion_tokens", "usd", "latency_ms"])
        for c in call_log:
            w.writerow([c.resolver, c.stage, c.model, c.inst_id, c.prompt_tokens,
                        c.completion_tokens, f"{c.usd:.6f}", f"{c.latency_ms:.1f}"])


def _write_scores(rows, call_log, out_tag=None):
    cost = {}
    for c in call_log:
        cost[(c.resolver, c.model, c.inst_id)] = cost.get((c.resolver, c.model, c.inst_id), 0.0) + c.usd
    cols = ["resolver", "model_tier", "model", "inst_id", "conflict_type", "n_sources",
            "correct", "misled", "abstained", "chosen_source", "pred_object", "gold_object", "usd"]
    fname = f"scores_{out_tag}.csv" if out_tag else "scores.csv"
    with open(os.path.join(config.RESULTS_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["usd"] = cost.get((r["resolver"], r["model"], r["inst_id"]), 0.0)
            w.writerow({k: r.get(k, "") for k in cols})


def _parse_models(model_str):
    """Parse comma-separated model tier keys, filtering to those in config.MODELS."""
    if not model_str:
        return None
    keys = [k.strip() for k in model_str.split(",")]
    invalid = [k for k in keys if k not in config.MODELS]
    if invalid:
        print(f"ERROR: Unknown model keys: {invalid}")
        print(f"Available: {list(config.MODELS.keys())}")
        exit(1)
    return {k: config.MODELS[k] for k in keys}


def _parse_resolvers(resolver_str):
    """Parse comma-separated resolver names."""
    if not resolver_str:
        return None
    names = [r.strip() for r in resolver_str.split(",")]
    for r in names:
        try:
            get_resolver(r)
        except KeyError:
            print(f"ERROR: Unknown resolver: {r}")
            print(f"Available: {list(config.RESOLVERS)}")
            exit(1)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WP2 DART conflict-resolution experiment")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; verify data loading + bundle assembly")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model tier keys (e.g. small-gpt4omini,small-haiku)")
    ap.add_argument("--resolvers", type=str, default=None,
                    help="comma-separated resolver names (e.g. llm_judge,multi_agent_debate)")
    ap.add_argument("--out", type=str, default=None,
                    help="output tag: writes results/scores_<tag>.csv and results/calls_<tag>.csv")
    ap.add_argument("--conflict-type", type=str, default=None,
                    choices=["fact_conflict", "temporal_conflict", "semantic_conflict"],
                    help="restrict bundles to a single conflict type and tag rows with it "
                         "(default: a 'mixed' bundle of 2 types). Run once per type to get a "
                         "per-conflict-type breakdown in the graph.")
    ap.add_argument("--control", action="store_true",
                    help="correct-evidence-only run (no conflict) — the knowledge probe. Tag "
                         "'control'. Grade conflict runs on facts a model gets right here.")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
    else:
        models = _parse_models(args.models)
        resolvers = _parse_resolvers(args.resolvers)
        run(models_override=models, resolvers_override=resolvers, out_tag=args.out,
            conflict_type=args.conflict_type, control=args.control)
