"""
run_experiment.py — driver for the D4 conflict-resolution study.

There are exactly TWO conditions, and one command runs both:

  control  — correct evidence only, no conflict. The knowledge probe: does the model
             even know this fact? Bundle = 1 passage.
  mixed    — correct evidence + config.N_CONFLICT_SOURCES conflicting passage(s) drawn
             at random from config.CONFLICT_TYPES. The actual task.

Per-conflict-type runs (fact / temporal / semantic separately) are deliberately NOT
here. Add them back only on request.

The knowledge filter is applied at SCORING time, not before the calls: the mixed run
covers every instance, and each row carries a `known` flag (1 = the model got this fact
right in its own control run). That way one file yields both the raw accuracy and the
knowledge-filtered accuracy, and the pair can be reported honestly. Filtering before the
calls — what this used to do — makes the "raw" column a copy of the filtered one.

Writes:
  results/scores_<tag>_control.csv  # the knowledge probe
  results/scores_<tag>.csv          # the mixed conflict run, with the `known` column
  results/calls_<tag>*.csv          # one row per LLM call (cost attribution)

Usage:
  python run_experiment.py                                   # config.MODELS x config.RESOLVERS
  python run_experiment.py --models small-gemflash --out gemflash
  python run_experiment.py --models frontier-opus45 --control-only   # probe only
  python run_experiment.py --models frontier-opus45 --no-control     # conflict run only
  python run_experiment.py --dry-run                         # no API calls; check bundles
"""
import os, csv, json, random, argparse
from datetime import datetime, timezone
from dotenv import load_dotenv
import config, dataio, metrics
from resolvers import get_resolver
from llm import CallLog

# Load API keys before any LLM/HF call. Bare load_dotenv() searches upward from the
# CURRENT directory, so launching from anywhere but this one found no .env and failed
# later with a bare "Missing OPENROUTER_API_KEY". Try the caller's location first (so a
# deliberate .env elsewhere still wins), then this file's own directory. python-dotenv
# does not overwrite variables that are already set, so the second call is a no-op when
# the first succeeded, and a real environment variable still beats both.
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _known_set(control_rows):
    """{model: {inst_id it answered correctly with no conflict}} — the knowledge filter.

    Per model on purpose: a fact one model knows another may not, so the filter cannot
    be a single shared instance list.
    """
    known = {}
    for r in control_rows:
        known.setdefault(r["model"], set())
        if r["correct"]:
            known[r["model"]].add(r["inst_id"])
    return known


def _conflict_object(inst):
    """The planted wrong object, if we can recover it (for the 'misled' metric)."""
    opts, ri = inst.get("options"), inst.get("replaced_option")
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ri, ri)
    if opts and idx is not None and idx < len(opts):
        return opts[idx]
    return inst.get("replaced_object")


def _judge(rows, call_log):
    """Run the equivalence pass inline, so a finished run reports its final number.

    Exact matching rejects answers that name the same entity in different words
    ("Prague" / "Prag V", "United States" / "United States of America"). Left uncredited
    those look like knowledge failures, which both understates accuracy and — on the
    control run — wrongly shrinks the set of facts the model is judged to know.

    Verdicts are cached in results/equivalence_cache.json, so re-runs are free.
    """
    import rescore
    cache = json.load(open(rescore.CACHE_PATH)) if os.path.exists(rescore.CACHE_PATH) else {}
    flipped = rescore.judge_rows(rows, cache, call_log)
    with open(rescore.CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    return flipped


def run(models_override=None, resolvers_override=None, out_tag=None,
        control=False, known=None, judge=True):
    """One condition over one model set. `control` picks the knowledge probe; `known`
    is the per-model known-fact map used to ANNOTATE (never to filter) the rows.

    Control mode assigns to module-level config, so running both conditions in one
    process must not let the control setting leak into the conflict run: snapshot
    and restore.
    """
    models = models_override if models_override else config.MODELS
    resolvers_list = resolvers_override if resolvers_override else config.RESOLVERS
    saved = config.N_CONFLICT_SOURCES
    try:
        return _run_one(models, resolvers_list, out_tag, control, known, judge)
    finally:
        config.N_CONFLICT_SOURCES = saved


def _run_one(models, resolvers_list, out_tag, control, known=None, judge=True):
    if control:
        config.N_CONFLICT_SOURCES = 0

    insts = dataio.load_instances()
    label = "control" if control else "mixed"
    print(f"[{label}] {len(insts)} instances, "
          f"{config.N_CONFLICT_SOURCES} conflicting passage(s) per bundle", flush=True)
    call_log, rows = [], []

    # Pre-build bundles once per instance so every resolver and every model sees the same
    # evidence. Each instance gets its own rng seeded from config.SEED + instance index,
    # so bundles are deterministic but not shared between instances.
    instance_bundles = {}
    for idx, inst in enumerate(insts):
        iid = str(inst.get("id", inst.get("subject", "?")))
        inst_rng = random.Random(config.SEED + idx)
        instance_bundles[iid] = dataio.build_bundle(inst, inst_rng)

    for tier, model in models.items():
        keep = known.get(model) if known else None
        for rname in resolvers_list:
            resolver = get_resolver(rname)
            for inst in insts:
                iid = str(inst.get("id", inst.get("subject", "?")))
                subject = inst.get("subject", "")
                query = dataio.query_phrase(inst)
                bundle = instance_bundles[iid]
                pred = resolver(subject, query, bundle, model, iid, call_log)
                sc = metrics.score_instance(pred, dataio.gold_object(inst), _conflict_object(inst))
                sc.update({"resolver": rname, "model_tier": tier, "model": model,
                           "inst_id": iid, "conflict_type": label,
                           "conflict_object": _conflict_object(inst),
                           "n_sources": len(bundle), "chosen_source": pred.get("chosen_source"),
                           "known": "" if keep is None else int(iid in keep)})
                rows.append(sc)
            print(f"  {tier} / {rname}: {len(insts)} done", flush=True)
    if judge:
        _judge(rows, call_log)
    _write_calls(call_log, out_tag)
    _write_scores(rows, call_log, out_tag)
    return rows


def _headline(rows):
    """The one number this experiment produces: knowledge-filtered accuracy, post-judge.

    Only the facts the model got right in its own control run count. Failing a fact the
    model never knew is not a conflict-resolution failure, so grading on it measures base
    knowledge instead of the thing under study.
    """
    print()
    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        kk = [r for r in mr if r["known"] == 1]
        if not kk:
            print(f"{model}\n  no control run — cannot be knowledge-filtered")
            continue
        k = sum(r["correct"] for r in kk)
        lo, hi = metrics.wilson_ci(k, len(kk))
        # Never print the confidence level ("95% CI") on the same line as the accuracy:
        # two percentages side by side get read as one number, and the 95 gets mistaken
        # for the result. Label the accuracy, and spell the interval out in words.
        print(f"{model}")
        print(f"  ACCURACY  {100 * k / len(kk):.1f}%   ({k} of {len(kk)} known facts)")
        print(f"  plausible range {100*lo:.1f} to {100*hi:.1f} at 95% confidence")
        print(f"  knows {100*len(kk)/len(mr):.0f}% of the {len(mr)} facts tested")


def dry_run():
    """No API calls. Load data, assemble bundles, print one example."""
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


def _write_calls(call_log, out_tag=None):
    fname = f"calls_{out_tag}.csv" if out_tag else "calls.csv"
    with open(os.path.join(config.RESULTS_DIR, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resolver", "stage", "model", "inst_id", "prompt_tokens",
                    "completion_tokens", "usd", "latency_ms", "provider"])
        for c in call_log:
            w.writerow([c.resolver, c.stage, c.model, c.inst_id, c.prompt_tokens,
                        c.completion_tokens, f"{c.usd:.6f}", f"{c.latency_ms:.1f}",
                        c.provider])


def _write_scores(rows, call_log, out_tag=None):
    cost = {}
    for c in call_log:
        cost[(c.resolver, c.model, c.inst_id)] = cost.get((c.resolver, c.model, c.inst_id), 0.0) + c.usd
    # pred/gold/conflict_object are the raw record; correct/misled/abstained are derived
    # from them and can be recomputed at any time by rescore.py. `known` is the knowledge
    # filter, recorded per row so raw and filtered accuracy both come out of this file.
    # run_utc records when this run happened, in the file itself. File mtime is not a
    # reliable substitute: rescore.py rewrites every scores file in one pass, and anything
    # that touches them flattens the ordering plot_results needs to tell which of two runs
    # of the same cell is the current one.
    cols = ["resolver", "model_tier", "model", "inst_id", "conflict_type", "n_sources",
            "known", "correct", "misled", "abstained", "judged", "chosen_source",
            "pred_object", "gold_object", "conflict_object", "usd", "run_utc"]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fname = f"scores_{out_tag}.csv" if out_tag else "scores.csv"
    with open(os.path.join(config.RESULTS_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["usd"] = cost.get((r["resolver"], r["model"], r["inst_id"]), 0.0)
            r["run_utc"] = stamp
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
    ap = argparse.ArgumentParser(description="WP2 DART conflict-resolution experiment "
                                             "(control probe + mixed conflict run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; verify data loading + bundle assembly")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model tier keys (e.g. small-gemflash,frontier-opus45)")
    ap.add_argument("--resolvers", type=str, default=None,
                    help="comma-separated resolver names (default config.RESOLVERS)")
    ap.add_argument("--out", type=str, default=None,
                    help="output tag: writes results/scores_<tag>.csv (mixed) and "
                         "results/scores_<tag>_control.csv (probe)")
    ap.add_argument("--control-only", action="store_true",
                    help="run only the knowledge probe (correct evidence, no conflict)")
    ap.add_argument("--no-control", action="store_true",
                    help="run only the mixed conflict run. Rows get no `known` flag, so "
                         "they cannot be knowledge-filtered later — use only when you "
                         "already have a control file for this model.")
    ap.add_argument("-n", "--n-instances", type=int, default=None,
                    help=f"how many instances to run (default {config.N_INSTANCES}); "
                         f"use a small number for a smoke test")
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="reasoning effort for thinking models. 'minimal' stops a model "
                         "burning its whole token budget thinking and truncating into a "
                         "false abstain (see config.REASONING_EFFORT)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"override config.MAX_TOKENS (default {config.MAX_TOKENS})")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the inline equivalence pass. The reported accuracy is then "
                         "strict string matching only, and runs a point or two low.")
    args = ap.parse_args()

    if args.n_instances:
        config.N_INSTANCES = args.n_instances
    if args.reasoning_effort:
        config.REASONING_EFFORT = args.reasoning_effort
    if args.max_tokens:
        config.MAX_TOKENS = args.max_tokens

    if args.dry_run:
        dry_run()
        raise SystemExit
    if args.control_only and args.no_control:
        raise SystemExit("--control-only and --no-control are mutually exclusive")

    models = _parse_models(args.models)
    resolvers = _parse_resolvers(args.resolvers)
    ctrl_tag = f"{args.out}_control" if args.out else "control"
    judge = not args.no_judge

    # Control FIRST: it establishes which facts each model actually knows, which is the
    # only thing that lets the conflict run be graded on "facts the model knows". It is
    # judged for equivalence before `known` is derived from it — an answer rejected only
    # for its wording would otherwise drop that fact out of the filter entirely.
    known = None
    if not args.no_control:
        known = _known_set(run(models, resolvers, ctrl_tag, control=True, judge=judge))
    if args.control_only:
        print()
        for m, ids in sorted(known.items()):
            print(f"{m}\n  knows {100*len(ids)/config.N_INSTANCES:.0f}% "
                  f"({len(ids)}/{config.N_INSTANCES})")
        raise SystemExit

    _headline(run(models, resolvers, args.out, control=False, known=known, judge=judge))
