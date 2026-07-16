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
    "openrouter/openai/gpt-4o-mini": ("GPT-4o mini", "params undisclosed", "small"),
    "openrouter/anthropic/claude-3-haiku": ("Claude 3 Haiku", "params undisclosed", "small"),
    "openrouter/qwen/qwen-2.5-7b-instruct": ("Qwen 2.5 7B", "7.6B params", "small"),
    "openrouter/google/gemini-3-flash-preview": ("Gemini 3 Flash", "params undisclosed", "small"),
    "openrouter/meta-llama/llama-3.3-70b-instruct": ("Llama 3.3 70B", "70B params", "mid"),
    "openrouter/anthropic/claude-sonnet-4": ("Claude Sonnet 4", "params undisclosed", "frontier"),
    "openrouter/openai/gpt-4o": ("GPT-4o", "params undisclosed", "frontier"),
}
MODEL_ORDER = list(MODELS)  # small -> mid -> frontier, as declared above

RESOLVER_TITLES = {
    "llm_judge": "LLM Judge",
    "llm_judge_provenance": "LLM Judge + Provenance",
    "multi_agent_debate": "Multi-Agent Debate",
}

# Reference categorical palette (light mode), slots 1-3 in fixed order.
TIER_COLORS = {"small": "#2a78d6", "mid": "#008300", "frontier": "#e87ba4"}
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

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in TIER_COLORS]
    ax.legend(handles, [f"{t} tier" for t in TIER_COLORS], loc="upper left",
              frameon=False, fontsize=9, labelcolor=INK_2)

    fig.tight_layout()
    out = PLOTS_DIR / f"accuracy_{resolver}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    df = load_scores()

    summary = (
        df.groupby(["resolver", "model"])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"),
             misled=("misled", "mean"), abstained=("abstained", "mean"),
             usd=("usd", "sum"))
        .reset_index()
    )
    summary["params"] = summary["model"].map(lambda m: MODELS[m][1])
    for col in ("accuracy", "misled", "abstained"):
        summary[col] = (100 * summary[col]).round(1)
    print(summary.to_string(index=False))

    for resolver in sorted(df["resolver"].unique()):
        out = plot_resolver(df, resolver)
        print(f"wrote {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
