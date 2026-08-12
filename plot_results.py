"""Plot accuracy-vs-model bar charts, one figure per resolver.

Reads every results/scores_<tier>.csv run, computes per-model accuracy for each
resolver, and writes one PNG per resolver to results/plots/.

There are two conditions and only two: `control` (the no-conflict knowledge probe) and
`mixed` (correct evidence + planted conflict). Rows tagged with any other conflict_type
are left over from the retired per-conflict-type runs and are dropped on load.

Usage:  .venv/bin/python plot_results.py
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from metrics import wilson_ci

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Display name, parameter count, and tier for each model id found in the CSVs.
# Parameter counts are only stated where the vendor has published them;
# closed models are marked undisclosed (estimates in the report, not here).
MODELS = {
    # Small tier
    "openrouter/openai/gpt-4o-mini": ("GPT-4o mini", "params undisclosed", "small"),
    "openrouter/anthropic/claude-3-haiku": ("Claude 3 Haiku", "params undisclosed", "small"),
    "openrouter/qwen/qwen-2.5-7b-instruct": ("Qwen 2.5 7B", "7.6B params", "small"),
    "openrouter/google/gemma-3-4b-it": ("Gemma 3 4B", "4B params", "small"),
    "openrouter/google/gemini-3-flash-preview": ("Gemini 3 Flash", "params undisclosed", "small"),
    # Mid tier
    "openrouter/meta-llama/llama-3.3-70b-instruct": ("Llama 3.3 70B", "70B params", "mid"),
    "openrouter/qwen/qwen-2.5-72b-instruct": ("Qwen 2.5 72B", "72B params", "mid"),
    # Frontier tier
    "openrouter/anthropic/claude-sonnet-4": ("Claude Sonnet 4", "params undisclosed", "frontier"),
    "openrouter/openai/gpt-4o": ("GPT-4o", "params undisclosed", "frontier"),
    "openrouter/google/gemini-2.5-pro": ("Gemini 2.5 Pro", "params undisclosed", "frontier"),
    "openrouter/openai/gpt-5": ("GPT-5", "params undisclosed", "frontier"),
    "openrouter/anthropic/claude-sonnet-4.5": ("Claude Sonnet 4.5", "params undisclosed", "frontier"),
    "openrouter/anthropic/claude-opus-4.5": ("Claude Opus 4.5", "params undisclosed", "frontier"),
    "openrouter/mistralai/mistral-large-2512": ("Mistral Large", "params undisclosed", "frontier"),
    # Reasoning tier
    "openrouter/deepseek/deepseek-r1": ("DeepSeek R1", "671B (37B active)", "reasoning"),
    "openrouter/openai/o4-mini": ("o4-mini", "params undisclosed", "reasoning"),
    "openrouter/openai/o3": ("o3", "params undisclosed", "reasoning"),
    # 2026 generation — same-generation comparison (see config.MODELS)
    "openrouter/qwen/qwen3.7-flash": ("Qwen3.7 Flash", "params undisclosed", "small"),
    "openrouter/anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", "params undisclosed", "small"),
    "openrouter/google/gemini-3.6-flash": ("Gemini 3.6 Flash", "params undisclosed", "small"),
    "openrouter/openai/gpt-5.4-mini": ("GPT-5.4 mini", "params undisclosed", "small"),
    "openrouter/z-ai/glm-5.2": ("GLM-5.2", "open weights", "mid"),
    "openrouter/qwen/qwen3.5-397b-a17b": ("Qwen3.5 397B", "397B (17B active)", "mid"),
    "openrouter/deepseek/deepseek-v4-pro": ("DeepSeek V4 Pro", "open weights", "mid"),
    "openrouter/qwen/qwen3.8-max": ("Qwen3.8 Max", "params undisclosed", "frontier"),
    "openrouter/qwen/qwen3.7-plus": ("Qwen3.7 Plus", "params undisclosed", "mid"),
    "openrouter/anthropic/claude-opus-5": ("Claude Opus 5", "params undisclosed", "frontier"),
    "openrouter/anthropic/claude-sonnet-5": ("Claude Sonnet 5", "params undisclosed", "frontier"),
    "openrouter/openai/gpt-5.6-terra": ("GPT-5.6 Terra", "params undisclosed", "frontier"),
    "openrouter/google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", "params undisclosed", "frontier"),
}
MODEL_ORDER = list(MODELS)  # small -> mid -> frontier -> reasoning, as declared above

RESOLVER_TITLES = {
    "llm_judge": "LLM Judge",
    "llm_judge_sc": "LLM Judge (self-consistency)",
    "llm_judge_provenance": "LLM Judge + Provenance",
    "multi_agent_debate": "Multi-Agent Debate",
}

# Reference categorical palette (light mode), slots 1-4 in fixed order.
TIER_COLORS = {"small": "#2a78d6", "mid": "#008300", "frontier": "#e87ba4", "reasoning": "#8a63d2"}

# The only two conditions. Anything else in a scores file predates the single-condition
# refactor (the per-conflict-type runs) and is dropped by load_scores().
CONTROL, CONFLICT = "control", "mixed"
CONDITIONS = (CONTROL, CONFLICT)

# With ALLOW_ABSTAIN=False the resolver is forced to answer, so genuine abstains are ~0.
# A high abstain rate therefore means failed API calls (rate-limits/provider errors) that
# were skipped and scored as blanks — a corrupted run. Cells above this are flagged and
# kept OUT of the chart so bad data never reaches a slide; re-run them.
MAX_ABSTAIN_OK = 0.10
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load_scores() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("scores_*.csv"))
    if not files:
        raise SystemExit(f"no scores_*.csv files found in {RESULTS_DIR}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["_file"] = f.name
        # Prefer the run timestamp stored in the file: mtime is destroyed by anything that
        # rewrites the results (rescore.py rewrites all of them at once). Files written
        # before run_utc existed fall back to mtime, with the filename as a deterministic
        # tiebreak so the choice is at least reproducible.
        stamped = d["run_utc"].dropna().iloc[:1] if "run_utc" in d.columns else []
        d["_when"] = (str(stamped.iloc[0]) if len(stamped)
                      else datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
                                   .isoformat(timespec="seconds"))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["_when", "_file"], kind="stable").reset_index(drop=True)

    retired = sorted(set(df["conflict_type"]) - set(CONDITIONS))
    if retired:
        n = int((~df["conflict_type"].isin(CONDITIONS)).sum())
        print(f"ignoring {n} row(s) from retired conditions {retired} — "
              f"only {CONTROL!r} and {CONFLICT!r} are plotted\n")
        df = df[df["conflict_type"].isin(CONDITIONS)]
    if df.empty:
        raise SystemExit(f"no {CONTROL}/{CONFLICT} rows in {RESULTS_DIR}")

    # Re-running a cell writes a NEW file rather than overwriting the old one, and the old
    # run may have used a different config (2 conflicting passages instead of 1, a leakier
    # source tag, ...). Pooling both averages two different experiments together, and for
    # control rows it inflates the denominator the knowledge filter divides by.
    #
    # Supersede whole cells, not individual rows: keep only the newest file that covers a
    # (resolver, model, condition). Row-level dedupe would keep the new run's rows AND the
    # instances only the old run happened to cover, splicing two configs into one number.
    cell = ["resolver", "model", "conflict_type"]
    newest = df.groupby(cell)["_file"].transform("last")   # files were read oldest-first
    stale = df["_file"] != newest
    if stale.any():
        for keys, grp in df[stale].groupby(cell):
            res, m, ct = keys
            won = newest[grp.index].iloc[0]
            print(f"superseded: {MODELS.get(m, (m,))[0]} / {ct} [{res}] — using {won} "
                  f"({int((df['_file'] == won).sum())} rows), ignoring "
                  f"{', '.join(sorted(set(grp['_file'])))}")
        df = df[~stale]
        print()
    df = df.drop(columns=["_file", "_when"])
    unknown = set(df["model"]) - set(MODELS)
    if unknown:
        raise SystemExit(f"models missing from MODELS mapping: {unknown}")
    return df


def known_ids(df: pd.DataFrame, model: str) -> set:
    """The facts `model` got right in its own control run — the knowledge filter.

    Read off the control rows rather than the `known` column so it works for files
    written before that column existed. The column is what run_experiment records at
    run time; this recomputes the same thing at plot time.
    """
    cm = df[(df["model"] == model) & (df["conflict_type"] == CONTROL)]
    return set(cm[cm["correct"] == 1]["inst_id"])


def plot_resolver(df: pd.DataFrame, resolver: str) -> Path:
    sub = df[df["resolver"] == resolver]
    models = [m for m in MODEL_ORDER if m in set(sub["model"])]

    accs, los, his, colors, labels = [], [], [], [], []
    for m in models:
        rows = sub[sub["model"] == m]
        n, k = len(rows), int(rows["correct"].sum())
        lo, hi = wilson_ci(k, n)
        accs.append(100 * k / n)
        los.append(100 * lo)
        his.append(100 * hi)
        name, params, tier = MODELS[m]
        colors.append(TIER_COLORS[tier])
        labels.append(f"{name}\n{params}")

    fig, ax = plt.subplots(figsize=(11.8, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(models))
    ax.bar(x, accs, width=0.62, color=colors, zorder=3)
    ax.errorbar(
        x, accs,
        yerr=[[a - lo for a, lo in zip(accs, los)], [hi - a for a, hi in zip(accs, his)]],
        fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, capthick=1, zorder=4,
    )
    for xi, acc, hi in zip(x, accs, his):
        ax.text(xi, hi + 2.2, f"{acc:.0f}%", ha="center", va="bottom",
                fontsize=10, color=INK_2, fontweight="bold")

    n_per_model = sub.groupby("model").size()
    n_txt = (f"n = {n_per_model.iloc[0]}" if n_per_model.nunique() == 1
             else f"n = {n_per_model.min()}–{n_per_model.max()}")
    ax.set_title(f"Accuracy by model — {RESOLVER_TITLES.get(resolver, resolver)}",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.015, f"% of conflict instances resolved to the gold answer "
                      f"({n_txt} per model, 95% Wilson CI)",
            transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=10)
    ax.set_xticks(list(x), labels)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    tiers_present = [t for t in TIER_COLORS if any(MODELS[m][2] == t for m in models)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in tiers_present]
    ax.legend(handles, [f"{t} tier" for t in tiers_present], loc="upper left",
              frameon=False, fontsize=9, labelcolor=INK_2)

    fig.tight_layout()
    out = PLOTS_DIR / f"accuracy_{resolver}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_knowledge_filter(df: pd.DataFrame, resolver: str) -> Path | None:
    """Grouped bars: per model, conflict accuracy over ALL facts vs over only the facts
    that model gets right conflict-free. This is the headline chart — it shows the
    filtered number honestly, next to the raw one it is derived from.

    Needs a control run per model; models without one are skipped (there is no way to
    know what they know). Returns None when no model qualifies."""
    sub = df[(df["resolver"] == resolver) & (df["conflict_type"] == CONFLICT)]
    models = [m for m in MODEL_ORDER
              if m in set(sub["model"]) and known_ids(df, m)]
    if not models:
        return None

    cells = []  # (model, raw, filt, n_raw, n_filt, knows_pct)
    for m in models:
        rows = sub[sub["model"] == m]
        n = len(rows)
        ab = int(rows["abstained"].sum())
        if ab / n > MAX_ABSTAIN_OK:  # corrupted run (failed calls) — leave a gap, don't plot
            print(f"  [skip] {MODELS[m][0]}: {ab}/{n} blank (failed calls) — "
                  f"excluded from chart, re-run this cell")
            continue
        known = known_ids(df, m)
        kk = rows[rows["inst_id"].isin(known)]
        if kk.empty:
            continue
        n_ctrl = len(df[(df["model"] == m) & (df["conflict_type"] == CONTROL)])
        cells.append((m, 100 * rows["correct"].mean(), 100 * kk["correct"].mean(),
                      n, len(kk), 100 * len(known) / n_ctrl))
    if not cells:
        return None

    fig, ax = plt.subplots(figsize=(max(9.5, 1.9 * len(cells) + 3), 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = list(range(len(cells)))
    bw = 0.38
    raw_vals = [c[1] for c in cells]
    filt_vals = [c[2] for c in cells]
    ax.bar([x - bw / 2 for x in xs], raw_vals, bw, color="#c3c2b7", zorder=3,
           label="All facts (raw)")
    ax.bar([x + bw / 2 for x in xs], filt_vals, bw, color="#2a78d6", zorder=3,
           label="Only facts the model knows")
    # Wilson CI on the filtered bar only — it is the one carrying the claim, and it has
    # the smaller n, so its uncertainty is what a reader needs to see.
    lo_err, hi_err = [], []
    for _, _, filt, _, n_f, _ in cells:
        lo, hi = wilson_ci(round(filt * n_f / 100), n_f)
        lo_err.append(filt - 100 * lo)
        hi_err.append(100 * hi - filt)
    ax.errorbar([x + bw / 2 for x in xs], filt_vals, yerr=[lo_err, hi_err],
                fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, capthick=1, zorder=4)
    for x, v in zip(xs, raw_vals):
        ax.text(x - bw / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                fontsize=9, color=MUTED)
    for x, v, e in zip(xs, filt_vals, hi_err):
        ax.text(x + bw / 2, v + e + 1.5, f"{v:.0f}", ha="center", va="bottom",
                fontsize=9.5, color=INK_2, fontweight="bold")

    ax.set_title(f"Knowledge filter — {RESOLVER_TITLES.get(resolver, resolver)}",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.015, "conflict-resolution accuracy on ALL facts vs only the facts each "
                      "model gets right with no conflict present (95% Wilson CI)",
            transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=10)
    ax.set_xticks(xs, [f"{MODELS[c[0]][0]}\nknows {c[5]:.0f}%  (n={c[4]})" for c in cells])
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
              fontsize=9.5, labelcolor=INK_2, ncol=2)

    fig.tight_layout()
    out = PLOTS_DIR / f"knowledge_filter_{resolver}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def _quality_report(df: pd.DataFrame) -> None:
    """Flag any resolver×model×conflict-type cell whose blank rate is too high to trust
    (with forced answers, blanks == failed API calls). These need re-running."""
    g = (df.groupby(["resolver", "model", "conflict_type"])
           .agg(n=("abstained", "size"), blank=("abstained", "sum")).reset_index())
    bad = g[g["blank"] / g["n"] > MAX_ABSTAIN_OK]
    if bad.empty:
        print("data quality: OK — every cell has an acceptable blank rate\n")
        return
    print("⚠ DATA QUALITY — cells with failed calls (scored as blanks); RE-RUN these:")
    for _, r in bad.iterrows():
        name = MODELS.get(r["model"], (r["model"],))[0]
        print(f"   {name} / {r['conflict_type']} [{r['resolver']}]: "
              f"{int(r['blank'])}/{int(r['n'])} blank ({100*r['blank']/r['n']:.0f}%)")
    print()


def _knowledge_filtered_report(df: pd.DataFrame) -> None:
    """The knowledge filter: grade each model's conflict run only on the facts it gets
    right conflict-free (its 'control' run). Failing a fact the model never knew isn't a
    resolution error, so this isolates conflict-resolution skill. No-op without control data."""
    ctrl = df[df["conflict_type"] == CONTROL]
    if ctrl.empty:
        print("knowledge filter: no control runs found — every run_experiment.py "
              "invocation writes one unless you pass --no-control\n")
        return
    conflict = df[df["conflict_type"] == CONFLICT]
    missing = sorted(set(conflict["model"]) - set(ctrl["model"]))
    print("KNOWLEDGE-FILTERED accuracy (conflict run graded only on facts the model knows):")
    for model in sorted(set(ctrl["model"])):
        known = known_ids(df, model)
        n_ctrl = len(ctrl[ctrl["model"] == model])
        name = MODELS.get(model, (model,))[0]
        ceil = 100 * len(known) / n_ctrl if n_ctrl else 0
        parts = []
        for res in sorted(set(conflict[conflict["model"] == model]["resolver"])):
            cc = conflict[(conflict["model"] == model) & (conflict["resolver"] == res)]
            kk = cc[cc["inst_id"].isin(known)]
            filt = f"{100 * kk['correct'].mean():.0f}%" if len(kk) else "n/a"
            parts.append(f"{res} {100 * cc['correct'].mean():.0f}%→{filt} (n={len(kk)})")
        tail = " | ".join(parts) if parts else "(no conflict run yet)"
        print(f"   {name}: knows {ceil:.0f}% of {n_ctrl}  |  {tail}")
    if missing:
        print("   no control run (cannot be knowledge-filtered — re-run without --no-control):")
        for m in missing:
            print(f"      {MODELS.get(m, (m,))[0]}")
    print("   report the pair: filtered alone overstates. 'graded on the N% it knows'.")
    print()


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    df = load_scores()
    _quality_report(df)
    _knowledge_filtered_report(df)

    # 'control' is the no-conflict knowledge probe, not a conflict condition — keep it out
    # of the summary and the accuracy chart (it belongs only in the knowledge-filter
    # report and chart, which take the full df because they need the control rows).
    viz = df[df["conflict_type"] == CONFLICT]

    summary = (
        viz.groupby(["resolver", "model"])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             misled=("misled", "mean"), abstained=("abstained", "mean"),
             usd=("usd", "sum"))
        .reset_index()
    )
    summary["params"] = summary["model"].map(lambda m: MODELS[m][1])
    for col in ("accuracy", "misled", "abstained"):
        summary[col] = (100 * summary[col]).round(1)
    print(summary.to_string(index=False))

    for resolver in sorted(viz["resolver"].unique()):
        out = plot_resolver(viz, resolver)
        print(f"wrote {out.relative_to(RESULTS_DIR.parent)}")
        kf = plot_knowledge_filter(df, resolver)  # full df — it needs the control rows
        if kf:
            print(f"wrote {kf.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
