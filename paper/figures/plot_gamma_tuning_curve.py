"""
Figure 5.4 — Gamma tuning of the M2-robust hybrid loss (1×3 line plot).

Three panels, line_training_curve style:
  (a) Mean annualised Sharpe vs gamma
  (b) Cross-seed CV vs gamma
  (c) Mean monthly long-short return std vs gamma   ← new: variance evidence

Highlights:
  gamma07 — red star, "Recommended trade-off"
  gamma10 — orange triangle, "Higher Sharpe, higher variance"

Data: doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv
Output: paper/figures/fig5_2_gamma_tuning_curve.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import apply_paper_style, style_framed_axes

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv"
OUT = ROOT / "paper/figures/fig5_2_gamma_tuning_curve.png"

GAMMA_ORDER = [
    "m2_robust_gamma03",
    "m2_robust_gamma05",
    "m2_robust_gamma07",
    "m2_robust_gamma10",
    "m2_robust_gamma15",
]
GAMMA_VAL = [0.03, 0.05, 0.07, 0.10, 0.15]
GAMMA_LABEL = [r"$\gamma$=0.03", r"$\gamma$=0.05", r"$\gamma$=0.07",
               r"$\gamma$=0.10", r"$\gamma$=0.15"]
BEST = "m2_robust_gamma07"
ALT = "m2_robust_gamma10"

C_LINE = "#1B3D6E"       # deep navy main line
C_BEST = "#C0392B"       # red star
C_ALT = "#E67E22"        # orange triangle
C_BAND = "#A8C8E8"       # light blue band


def _add_markers(ax, x_vals, y_vals, gamma_names):
    for i, name in enumerate(gamma_names):
        if name == BEST:
            ax.plot(x_vals[i], y_vals[i], marker="*", markersize=14,
                    color=C_BEST, zorder=5, markeredgecolor="black", markeredgewidth=0.6)
        elif name == ALT:
            ax.plot(x_vals[i], y_vals[i], marker="^", markersize=10,
                    color=C_ALT, zorder=5, markeredgecolor="black", markeredgewidth=0.6)
        else:
            ax.plot(x_vals[i], y_vals[i], marker="o", markersize=7,
                    color=C_LINE, zorder=4, markeredgecolor="white", markeredgewidth=0.5)


def panel_sharpe(ax, agg):
    means = agg["sharpe_mean"].to_numpy()
    stds = agg["sharpe_std"].to_numpy()

    ax.fill_between(GAMMA_VAL, means - stds, means + stds,
                    color=C_BAND, alpha=0.45, zorder=1)
    ax.plot(GAMMA_VAL, means, color=C_LINE, lw=2.0, zorder=3)
    _add_markers(ax, GAMMA_VAL, means, GAMMA_ORDER)

    # Annotations
    best_idx = GAMMA_ORDER.index(BEST)
    alt_idx = GAMMA_ORDER.index(ALT)
    ax.annotate("Recommended\ntrade-off",
                xy=(GAMMA_VAL[best_idx], means[best_idx]),
                xytext=(GAMMA_VAL[best_idx] - 0.025, means[best_idx] + 0.18),
                fontsize=8.5, color=C_BEST, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.9))
    ax.annotate("Higher Sharpe,\nhigher variance",
                xy=(GAMMA_VAL[alt_idx], means[alt_idx]),
                xytext=(GAMMA_VAL[alt_idx] + 0.018, means[alt_idx] - 0.22),
                fontsize=8.5, color=C_ALT, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_ALT, lw=0.9))

    ax.set_xticks(GAMMA_VAL)
    ax.set_xticklabels(GAMMA_LABEL, fontsize=9.5)
    ax.set_ylabel(r"Mean Sharpe  ($\sqrt{12}\cdot\bar r/\sigma_r$)", fontsize=10.5, fontweight="bold")
    ax.set_title("(a) Mean Sharpe vs $\\gamma$\n(band = ±1 std across seeds)", fontsize=10.5, pad=5)
    ax.set_ylim(bottom=0)
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9)


def panel_cv(ax, agg):
    cvs = agg["sharpe_cv"].to_numpy()

    ax.plot(GAMMA_VAL, cvs, color=C_LINE, lw=2.0, zorder=3)
    _add_markers(ax, GAMMA_VAL, cvs, GAMMA_ORDER)

    # Preferred band
    ax.axhspan(0, 0.30, color="#D5F5E3", alpha=0.40, zorder=0)
    ax.text(0.155, 0.28, "Preferred\nstability zone", fontsize=8, color="#1E8449",
            ha="right", va="top")

    ax.set_xticks(GAMMA_VAL)
    ax.set_xticklabels(GAMMA_LABEL, fontsize=9.5)
    ax.set_ylabel(r"CV = $\sigma_S / |\mu_S|$  (lower = more stable)", fontsize=10.5, fontweight="bold")
    ax.set_title("(b) Seed-stability (CV) vs $\\gamma$", fontsize=10.5, pad=5)
    ax.set_ylim(bottom=0)
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9)


def panel_ls_std(ax, agg):
    """Mean monthly long-short return std — direct variance evidence."""
    stds = agg["ls_std_mean"].to_numpy()

    ax.plot(GAMMA_VAL, stds, color=C_LINE, lw=2.0, zorder=3)
    _add_markers(ax, GAMMA_VAL, stds, GAMMA_ORDER)

    best_idx = GAMMA_ORDER.index(BEST)
    ax.annotate(f"{stds[best_idx]*100:.2f}%",
                xy=(GAMMA_VAL[best_idx], stds[best_idx]),
                xytext=(GAMMA_VAL[best_idx] - 0.025, stds[best_idx] + 0.003),
                fontsize=8.5, color=C_BEST, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.9))

    ax.set_xticks(GAMMA_VAL)
    ax.set_xticklabels(GAMMA_LABEL, fontsize=9.5)
    ax.set_ylabel("Mean monthly LS return std", fontsize=10.5, fontweight="bold")
    ax.set_title("(c) Portfolio return volatility vs $\\gamma$\n(lower = less volatile)", fontsize=10.5, pad=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_ylim(bottom=0)
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9)


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(SRC)
    df = df[df["cap_tag"] == "cap05"].copy()

    agg = (
        df.groupby("loss")
        .agg(
            sharpe_mean=("long_short_sharpe", "mean"),
            sharpe_std=("long_short_sharpe", "std"),
            ls_std_mean=("long_short_std", "mean"),
        )
        .reindex(GAMMA_ORDER)
        .reset_index()
    )
    agg["sharpe_cv"] = agg["sharpe_std"] / agg["sharpe_mean"].abs()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.15, top=0.83, wspace=0.30)

    panel_sharpe(axes[0], agg)
    panel_cv(axes[1], agg)
    panel_ls_std(axes[2], agg)

    fig.suptitle(
        "Figure 5.4 — Gamma tuning of the M2-robust hybrid loss  "
        "(cap05 · 3 seeds · test 1995-01..1996-12)",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
