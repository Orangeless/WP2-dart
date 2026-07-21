"""Plot accuracy-vs-model bar charts, one figure per resolver.

Reads every results/scores_<tier>.csv run, computes per-model accuracy for each
resolver, and writes one PNG per resolver to results/plots/.

Usage:  .venv/bin/python plot_results.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

# Per-conflict-type labels + palette (for the conflict-type breakdown chart).
CONFLICT_LABELS = {"fact_conflict": "Misinformation", "temporal_conflict": "Temporal",
                   "semantic_conflict": "Semantic"}
CONFLICT_COLORS = {"fact_conflict": "#2a78d6", "temporal_conflict": "#008300",
                   "semantic_conflict": "#e87ba4"}

# With ALLOW_ABSTAIN=False the resolver is forced to answer, so genuine abstains are ~0.
# A high abstain rate therefore means failed API calls (rate-limits/provider errors) that
# were skipped and scored as blanks — a corrupted run. Cells above this are flagged and
# kept OUT of the chart so bad data never reaches a slide; re-run them.
MAX_ABSTAIN_OK = 0.10
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_scores() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("scores_*.csv"))
    if not files:
        raise SystemExit(f"no scores_*.csv files found in {RESULTS_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    unknown = set(df["model"]) - set(MODELS)
    if unknown:
        raise SystemExit(f"models missing from MODELS mapping: {unknown}")
    return df


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


def plot_by_conflict_type(df: pd.DataFrame, resolver: str) -> Path | None:
    """Grouped bars: accuracy per model, split by conflict type (Misinformation /
    Temporal / Semantic). Only rendered when the data carries ≥2 typed runs — i.e.
    after running run_experiment.py with --conflict-type. Returns None otherwise."""
    sub = df[(df["resolver"] == resolver) & (df["conflict_type"].isin(CONFLICT_LABELS))]
    models = [m for m in MODEL_ORDER if m in set(sub["model"])]
    types = [t for t in CONFLICT_LABELS if t in set(sub["conflict_type"])]
    if len(types) < 2 or not models:
        return None

    fig, ax = plt.subplots(figsize=(max(11.8, 1.6 * len(models) + 3), 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = list(range(len(models)))
    group_w, bar_w = 0.8, 0.8 / len(types)
    n_lo, n_hi = 10**9, 0
    for j, ct in enumerate(types):
        # collect only model×type cells that were actually run — a missing run must
        # leave a gap, never render as a fake 0% bar.
        xpos, accs, lo_err, hi_err = [], [], [], []
        for i, m in enumerate(models):
            rows = sub[(sub["model"] == m) & (sub["conflict_type"] == ct)]
            n, k = len(rows), int(rows["correct"].sum())
            if n == 0:
                continue
            ab = int(rows["abstained"].sum())
            if ab / n > MAX_ABSTAIN_OK:  # corrupted run (failed calls) — leave a gap, don't plot
                print(f"  [skip] {MODELS[m][0]} / {CONFLICT_LABELS[ct]}: {ab}/{n} blank "
                      f"(failed calls) — excluded from chart, re-run this cell")
                continue
            n_lo, n_hi = min(n_lo, n), max(n_hi, n)
            lo, hi = wilson_ci(k, n)
            acc = 100 * k / n
            xpos.append(i - group_w / 2 + bar_w * (j + 0.5))
            accs.append(acc)
            lo_err.append(acc - 100 * lo)
            hi_err.append(100 * hi - acc)
        if not xpos:
            continue
        ax.bar(xpos, accs, width=bar_w * 0.9, color=CONFLICT_COLORS[ct], zorder=3,
               label=CONFLICT_LABELS[ct])
        for xi, acc in zip(xpos, accs):
            ax.text(xi, acc + 1.5, f"{acc:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)
        ax.errorbar(xpos, accs, yerr=[lo_err, hi_err],
                    fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, capthick=1, zorder=4)

    n_txt = f"n = {n_lo}" if n_lo == n_hi else f"n = {n_lo}–{n_hi}"
    ax.set_title(f"Accuracy by conflict type — {RESOLVER_TITLES.get(resolver, resolver)}",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.015, f"% resolved to the gold answer, split by planted conflict type "
                      f"({n_txt} per model×type, 95% Wilson CI)",
            transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=10)
    ax.set_xticks(xs, [MODELS[m][0] for m in models])
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2, ncol=len(types))

    fig.tight_layout()
    out = PLOTS_DIR / f"by_conflict_type_{resolver}.png"
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


def plot_knowledge_filter(df: pd.DataFrame) -> list:
    """One chart per model that has a control run: conflict-resolution accuracy on ALL facts
    (raw) vs only the facts the model gets right conflict-free ('facts it knows'). This is
    the headline slide — it shows the 90s number honestly, side-by-side with the raw one."""
    lj = df[df["resolver"] == "llm_judge"]
    ctrl = lj[lj["conflict_type"] == "control"]
    outs = []
    for model in sorted(set(ctrl["model"])):
        cm = ctrl[ctrl["model"] == model]
        known = set(cm[cm["correct"] == 1]["inst_id"])
        knows_pct = 100 * len(known) / len(cm) if len(cm) else 0
        sub = lj[(lj["model"] == model) & (lj["conflict_type"].isin(CONFLICT_LABELS))]
        types, raw_vals, filt_vals = [], [], []
        for t in CONFLICT_LABELS:
            cc = sub[sub["conflict_type"] == t]
            if cc.empty or cc["abstained"].sum() / len(cc) > MAX_ABSTAIN_OK:
                continue  # missing or corrupted cell — skip
            kk = cc[cc["inst_id"].isin(known)]
            if not len(kk):
                continue
            types.append(t)
            raw_vals.append(100 * cc["correct"].mean())
            filt_vals.append(100 * kk["correct"].mean())
        if not types:
            continue
        name = MODELS.get(model, (model,))[0]

        fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        xs = list(range(len(types)))
        bw = 0.38
        ax.bar([x - bw / 2 for x in xs], raw_vals, bw, color="#c3c2b7", zorder=3,
               label="All facts (raw)")
        ax.bar([x + bw / 2 for x in xs], filt_vals, bw, color="#2a78d6", zorder=3,
               label="Only facts the model knows")
        for x, v in zip(xs, raw_vals):
            ax.text(x - bw / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=9, color=MUTED)
        for x, v in zip(xs, filt_vals):
            ax.text(x + bw / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=9.5, color=INK_2, fontweight="bold")

        ax.set_title(f"Knowledge filter — {name}", color=INK, fontsize=14,
                     fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.015, f"conflict-resolution accuracy on ALL facts vs only the "
                          f"{knows_pct:.0f}% this model gets right with no conflict present",
                transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=10)
        ax.set_xticks(xs, [CONFLICT_LABELS[t] for t in types])
        ax.tick_params(colors=MUTED, labelsize=9.5)
        for tick in ax.get_xticklabels():
            tick.set_color(INK_2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False,
                  fontsize=9.5, labelcolor=INK_2, ncol=2)

        fig.tight_layout()
        safe = name.lower().replace(" ", "-").replace(".", "").replace("(", "").replace(")", "")
        out = PLOTS_DIR / f"knowledge_filter_{safe}.png"
        fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        outs.append(out)
    return outs


def _knowledge_filtered_report(df: pd.DataFrame) -> None:
    """The knowledge filter: grade each model's conflict runs only on the facts it gets
    right conflict-free (its 'control' run). Failing a fact the model never knew isn't a
    resolution error, so this isolates conflict-resolution skill. No-op without control data."""
    ctrl = df[df["conflict_type"] == "control"]
    if ctrl.empty:
        print("knowledge filter: no control runs found — run `--control` per model to enable\n")
        return
    print("KNOWLEDGE-FILTERED accuracy (conflict runs graded only on facts the model knows):")
    for model in sorted(set(ctrl["model"])):
        cm = ctrl[ctrl["model"] == model]
        known = set(cm[cm["correct"] == 1]["inst_id"])
        name = MODELS.get(model, (model,))[0]
        ceil = 100 * len(known) / len(cm) if len(cm) else 0
        sub = df[(df["model"] == model) & (df["conflict_type"].isin(CONFLICT_LABELS))]
        parts = []
        for ct in CONFLICT_LABELS:
            cc = sub[sub["conflict_type"] == ct]
            if cc.empty:
                continue
            kk = cc[cc["inst_id"].isin(known)]
            raw = 100 * cc["correct"].mean()
            filt = 100 * kk["correct"].mean() if len(kk) else float("nan")
            parts.append(f"{CONFLICT_LABELS[ct]} {raw:.0f}%→{filt:.0f}%")
        tail = " | ".join(parts) if parts else "(no per-type conflict runs yet)"
        print(f"   {name}: knows {ceil:.0f}%  |  {tail}")
    print()


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    df = load_scores()
    _quality_report(df)
    _knowledge_filtered_report(df)

    # 'control' is the no-conflict knowledge probe, not a conflict condition — keep it out
    # of the summary and the charts (it belongs only in the knowledge-filter report above).
    viz = df[df["conflict_type"] != "control"]

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
        ct_out = plot_by_conflict_type(viz, resolver)
        if ct_out:
            print(f"wrote {ct_out.relative_to(RESULTS_DIR.parent)}")

    for kf in plot_knowledge_filter(df):  # uses full df (needs the control rows)
        print(f"wrote {kf.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
