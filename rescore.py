"""
rescore.py — recompute correct/misled/abstained from the stored (pred, gold) pairs,
without re-running a single resolver.

Two reasons this exists.

1. **Uniformity.** results/scores_*.csv were written at different times with different
   versions of metrics.match(), so the stored columns are not comparable across files —
   re-scoring some models by +5 points and others by +1. Scoring belongs in a pass over
   the raw record, not baked in at run time. Run this and every file is scored by the
   same rules.

2. **Answers that are right but worded differently.** Exact matching rejects
   "Police and Crime Commissioner for Wiltshire" vs "Wiltshire Police and Crime
   Commissioner", or "Paracelsus Medal" vs "Paracelsus Medal of the German Medical
   Association". Roughly a third of the "neither correct nor misled" bucket is this, and
   it shows up even in control runs where there is no conflict at all — so it deflates
   the knowledge ceiling that the knowledge filter divides by. `--judge` adds a second
   pass that asks a cheap model whether two strings denote the same entity.

   The judge sees ONLY the two strings — never the evidence, never the bundle — so it
   cannot leak the answer into the task. It runs after the fact, on stored output.
   Verdicts are cached in results/equivalence_cache.json, so re-running is free and
   the scores stay reproducible.

    python rescore.py                    # deterministic re-score only (no API calls)
    python rescore.py --judge            # + equivalence pass over the disagreements
    python rescore.py --judge --dry-run  # report what would change, write nothing
"""
import argparse, csv, json, os, re, sys
import config, metrics

# llm/dotenv are imported lazily inside the judge so the deterministic re-score — the
# part you run most often — needs nothing beyond the standard library.
CACHE_PATH = os.path.join(config.RESULTS_DIR, "equivalence_cache.json")

# Strict on purpose. The failure mode to avoid is over-crediting: a previous probe used
# substring matching, scored "John Entwistle" as "The Who", and had to be thrown away.
# Whole-vs-part is the specific trap here, so it is called out explicitly.
_PROMPT = (
    "Do these two strings refer to the SAME entity or value?\n\n"
    "A: {a}\n"
    "B: {b}\n\n"
    "Answer YES only if they denote the same thing. Spelling, word order, "
    "abbreviation, article, punctuation and language variants are all still the same "
    "thing.\n"
    "Answer NO if one is merely a parent, part, superset or subset of the other — a "
    "university is not its faculty, a country is not its capital, a band is not its "
    "member — or if they are only related.\n\n"
    "Reply with one word: YES or NO."
)


def _equivalent(a: str, b: str, cache: dict, log: list) -> bool:
    """Cached one-call equivalence check. Cache key is the raw pair, so a verdict is
    reused across every file and every future run."""
    key = json.dumps([str(a), str(b)])   # unambiguous, and survives the JSON cache file
    if key in cache:
        return cache[key]
    from llm import chat
    out = chat(_PROMPT.format(a=a, b=b), config.EQUIVALENCE_MODEL,
               resolver="rescore", stage="equivalence", inst_id="-",
               temperature=0.0, max_tokens=200, seed=config.SEED, log=log)
    # take the last YES/NO token so a model that thinks out loud still parses
    hits = re.findall(r"\b(YES|NO)\b", (out or "").upper())
    if not hits:
        # A failed call returns "", which would parse as NO and cache a fake verdict —
        # an outage would look like a working-but-strict judge. Refuse instead.
        raise RuntimeError(
            f"equivalence judge {config.EQUIVALENCE_MODEL} returned no YES/NO "
            f"(got {out[:120]!r}). Check OPENROUTER_API_KEY / model availability.")
    cache[key] = hits[-1] == "YES"
    return cache[key]


def judge_rows(rows: list, cache: dict, log: list) -> int:
    """Credit rows whose answer denotes the same entity as gold in different words.

    Shared by the inline pass in run_experiment (so a finished run reports its final
    number) and by rescore.py's whole-directory pass. Rows are mutated in place; the
    count of newly credited rows is returned.

    A judge outage must never lose a completed experiment, so a failed verdict leaves
    the row exactly as scored rather than propagating the exception.
    """
    judged = 0
    for r in rows:
        if int(r.get("correct") or 0) or int(r.get("abstained") or 0):
            continue
        pred, gold = r.get("pred_object"), r.get("gold_object")
        conflict = (r.get("conflict_object") or "").strip()
        try:
            if _equivalent(pred, gold, cache, log):
                r["correct"], r["judged"], judged = 1, 1, judged + 1
            elif conflict and not int(r.get("misled") or 0) and _equivalent(pred, conflict, cache, log):
                r["misled"], r["judged"], judged = 1, 1, judged + 1
        except Exception as exc:
            print(f"  ! equivalence judge failed ({exc}); row left as scored")
    return judged


def rescore_file(path: str, cache: dict, log: list, use_judge: bool) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows, cols = list(reader), list(reader.fieldnames or [])
    if not rows:
        return {}
    before = sum(int(r.get("correct") or 0) for r in rows)
    for r in rows:
        conflict = (r.get("conflict_object") or "").strip()
        # A row already credited by the equivalence judge keeps its credit. Deterministic
        # matching rejected it by definition — that is WHY the judge was asked — so
        # recomputing would silently strip the credit and lower every number. Runs are now
        # judged inline, so a bare `rescore.py` would otherwise undo the last run's result.
        if int(r.get("judged") or 0):
            r.setdefault("judged", 1)
            continue
        sc = metrics.score_instance({"object": r.get("pred_object")},
                                    r.get("gold_object"), conflict or None)
        r["correct"], r["abstained"] = sc["correct"], sc["abstained"]
        # Older files predate the conflict_object column; without it misled is not
        # recomputable, so keep whatever was stored rather than zeroing real data.
        if conflict:
            r["misled"] = sc["misled"]
        r.setdefault("judged", 0)
    judged = judge_rows(rows, cache, log) if use_judge else 0
    after = sum(int(r.get("correct") or 0) for r in rows)
    if "judged" not in cols:
        cols.append("judged")
    return {"rows": rows, "cols": cols, "n": len(rows),
            "before": before, "after": after, "judged": judged}


def main() -> None:
    ap = argparse.ArgumentParser(description="re-score stored runs from pred/gold")
    ap.add_argument("--judge", action="store_true",
                    help="also ask an LLM whether rejected answers mean the same as gold")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    if args.judge:
        from dotenv import load_dotenv
        load_dotenv()   # OPENROUTER_API_KEY for the equivalence calls

    files = sorted(f for f in os.listdir(config.RESULTS_DIR)
                   if f.startswith("scores_") and f.endswith(".csv"))
    if not files:
        sys.exit(f"no scores_*.csv in {config.RESULTS_DIR}")

    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    n_cached, log = len(cache), []
    print(f"re-scoring {len(files)} file(s)"
          f"{' with equivalence judge ' + config.EQUIVALENCE_MODEL if args.judge else ''}"
          f"{'  [dry run]' if args.dry_run else ''}\n")

    tot_n = tot_before = tot_after = 0
    for name in files:
        path = os.path.join(config.RESULTS_DIR, name)
        res = rescore_file(path, cache, log, args.judge)
        if not res:
            continue
        tot_n += res["n"]; tot_before += res["before"]; tot_after += res["after"]
        delta = res["after"] - res["before"]
        mark = f"  {delta:+d}" if delta else ""
        print(f"  {name:46} {res['before']/res['n']:.2f} -> {res['after']/res['n']:.2f}"
              f"{mark}{'  (' + str(res['judged']) + ' by judge)' if res['judged'] else ''}")
        if not args.dry_run:
            # Preserve the file's mtime. Re-scoring rewrites EVERY scores file in one go,
            # which otherwise stamps them all with the same timestamp and destroys the
            # only record of which run is newer — plot_results uses that to decide which
            # of two runs of the same cell supersedes the other.
            stat = os.stat(path)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=res["cols"])
                w.writeheader()
                for r in res["rows"]:
                    w.writerow({k: r.get(k, "") for k in res["cols"]})
            os.utime(path, (stat.st_atime, stat.st_mtime))

    print(f"\noverall {tot_before/tot_n:.3f} -> {tot_after/tot_n:.3f} "
          f"over {tot_n} rows ({tot_after - tot_before:+d})")
    if args.judge:
        spend = sum(c.usd for c in log)
        print(f"judge: {len(cache) - n_cached} new verdicts, "
              f"{n_cached} from cache, ${spend:.4f}")
        if not args.dry_run:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
