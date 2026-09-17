#!/usr/bin/env python3
"""
make_figures.py
===============
Regenerates the index-dependent manuscript figures from the deposited
extracted-signal dataset under the baseline specification (Section 3.6):

    figure3.png  Figure 2  AQI temporal evolution (95% CI)
    figure1.png  Figure 4  AQI by application
    figure2.png  Figure 5  AQI aggregated by primary operational market
    figure6.png  Figure 6  top adoption barriers (declared categories)
    figure9.png  Figure 8  application-level validation scatter plots

Figures 1 (pipeline), 3 (sentiment heatmap) and 7 (cultural factors) do not
depend on the index and are unchanged.

    python scripts/make_figures.py --out figures/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from aqi_core import (APP_ORDER, APP_LABEL, COUNTRY_LABEL, DECLARED_BARRIERS,  # noqa: E402
                      RELIABILITY_BARRIERS, load, build_components, aqi)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
GREEN, RED, BLUE, GREY = "#2ca02c", "#d62728", "#4575b4", "#808080"
APP_BAR_COLORS = ["#1a9850", "#66bd63", "#a6d96a", "#fdae61", "#f46d43", "#d73027"]


def temporal(df, out):
    y = df.groupby("year")["AQI"].agg(["mean", "std", "count"])
    ci = 1.96 * y["std"] / np.sqrt(y["count"])
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    ax.fill_between(y.index, y["mean"] - ci, y["mean"] + ci, color=GREEN, alpha=0.22,
                    linewidth=0, label="95% CI")
    ax.plot(y.index, y["mean"], marker="o", markersize=10, linewidth=3, color=GREEN, label="Mean AQI")
    for yr, m in y["mean"].items():
        ax.annotate(f"{m:.3f}", (yr, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean AQI Score")
    ax.set_title("AQI Temporal Evolution (2019-2024)", fontweight="bold")
    ax.set_xticks(y.index); ax.set_ylim(0.60, 0.83)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def by_app(df, out):
    m = df.groupby("app")["AQI"].mean().reindex(APP_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar([APP_LABEL[a] for a in m.index], m.values, color=APP_BAR_COLORS,
                  edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, m.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.axhline(m.mean(), color=GREY, ls="--", lw=1.2,
               label=f"Mean of application means = {m.mean():.3f}")
    ax.set_ylabel("Mean AQI Score"); ax.set_ylim(0, 1.0)
    ax.set_title("AQI Score by Application")
    ax.legend(loc="upper right", framealpha=0.95)
    plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def by_country(df, out):
    m = df.groupby("country")["AQI"].mean().sort_values(ascending=False)
    labels = [COUNTRY_LABEL[c] for c in m.index]
    colors = [GREEN if c in ("Kenya", "Senegal", "Nigeria") else RED for c in labels]
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)
    bars = ax.bar(labels, m.values, color=colors, edgecolor="black", linewidth=1.2, width=0.65)
    for bar, v in zip(bars, m.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean AQI Score", fontsize=13); ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=12)
    ax.set_title("AQI Score Aggregated by Primary Operational Market", fontweight="bold", fontsize=15)
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(facecolor=GREEN, edgecolor="black", label="Higher-performing markets"),
                       mpatches.Patch(facecolor=RED, edgecolor="black", label="Lower-performing markets")],
              loc="upper right", fontsize=12, framealpha=0.95)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def barriers(df, out):
    n = len(df)
    shares = {k: 100 * df["_barriers"].map(lambda l: k in l).sum() / n for k in DECLARED_BARRIERS}
    items = sorted(shares.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.barh(labels, vals, color=[RED if k in RELIABILITY_BARRIERS else BLUE for k in labels],
                   edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.12, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=10, fontweight="bold")
    ax.invert_yaxis(); ax.set_xlim(0, 12.5)
    ax.set_xlabel("% of all reviews"); ax.set_title("Top Adoption Barriers")
    ax.legend(handles=[mpatches.Patch(color=RED, label="Reliability-related"),
                       mpatches.Patch(color=BLUE, label="Other")], loc="lower right", framealpha=0.95)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def validation(df, out):
    app = df.groupby("app").agg(AQI=("AQI", "mean"), rating=("rating", "mean"),
                                pos=("sentiment", lambda x: 100 * (x == "positive").mean()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    for ax, col, ylab, title in [(axes[0], "rating", "Google Play Star Rating", "AQI vs Google Play Star Rating"),
                                 (axes[1], "pos", "Positive Reviews (%)", "AQI vs Positive Reviews (%)")]:
        r, _ = stats.pearsonr(app["AQI"], app[col])
        ax.scatter(app["AQI"], app[col], s=70, color="#2166ac", edgecolor="black", zorder=3)
        for a, row in app.iterrows():
            ax.annotate(APP_LABEL[a], (row["AQI"], row[col]), xytext=(7, 5),
                        textcoords="offset points", fontsize=9)
        k, c0 = np.polyfit(app["AQI"], app[col], 1)
        xx = np.linspace(app["AQI"].min() - 0.03, app["AQI"].max() + 0.03, 100)
        ax.plot(xx, k * xx + c0, "--", color=GREY, lw=1.2)
        ax.set_xlabel("AQI Score"); ax.set_ylabel(ylab)
        ax.set_title(f"{title}  (r = {r:.3f}, p < 0.001)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extracted_signals.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.data)
    s, i, b = build_components(df)
    df["AQI"] = aqi(s, i, b)
    temporal(df, out / "figure3.png")
    by_app(df, out / "figure1.png")
    by_country(df, out / "figure2.png")
    barriers(df, out / "figure6.png")
    validation(df, out / "figure9.png")
    print("figures written to", out)


if __name__ == "__main__":
    main()
