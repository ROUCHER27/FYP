"""
Figure 5.3 — Phase 2.2 γ refinement (multi-seed), paper-plot-skills
`bar_grouped_hatch` ablation style.

Single row, two panels:
(a) Mean Sharpe with ±1 std error bars + min/max range whiskers; γ07 bar uses
    `//` white-hatch on deep red to mark the recommended variant.
(b) Coefficient of variation (CV) as a bar chart on the same γ axis; γ07 bar
    again hatched red. Value labels above each bar (bold red for best).

Inputs : doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv
Output : paper/figures/fig5_3_gamma_refinement.png (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import (
    BEST_VALUE_COLOR,
    C_ABL,
    HATCH_BEST,
    apply_paper_style,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv"
OUT = ROOT / "paper/figures/fig5_3_gamma_refinement.png"

GAMMA_ORDER = [
    "m2_robust_gamma03",
    "m2_robust_gamma05",
    "m2_robust_gamma07",
    "m2_robust_gamma10",
    "m2_robust_gamma15",
]
GAMMA_LABEL = {
    "m2_robust_gamma03": r"$\gamma$=0.3",
    "m2_robust_gamma05": r"$\gamma$=0.5",
    "m2_robust_gamma07": r"$\gamma$=0.7",
    "m2_robust_gamma10": r"$\gamma$=1.0",
    "m2_robust_gamma15": r"$\gamma$=1.5",
}
BEST = "m2_robust_gamma07"


def draw_sharpe_panel(ax, agg: pd.DataFrame) -> None:
    x = np.arange(len(GAMMA_ORDER))
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()
    mins = agg["min"].to_numpy()
    maxs = agg["max"].to_numpy()

    # Three-tone warm gradient along gamma axis + red for best.
    colors = [C_ABL[0], C_ABL[1], C_ABL[2], C_ABL[1], C_ABL[0]]
    hatches = ["", "", HATCH_BEST, "", ""]

    for i, (m, c, h) in enumerate(zip(means, colors, hatches)):
        ax.bar(
            x[i],
            m,
            width=0.62,
            color=c,
            hatch=h,
            edgecolor="white",
            linewidth=0.9,
            zorder=2,
        )
    # Error bars (std) overlaid on bars
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#222",
        elinewidth=1.0,
        capsize=4.5,
        capthick=1.0,
        zorder=3,
    )
    # Min-max whiskers (thin vertical)
    for i in range(len(x)):
        ax.vlines(
            x[i],
            mins[i],
            maxs[i],
            color="#888",
            linewidth=1.4,
            alpha=0.55,
            zorder=2,
        )
    # Value labels above bars; red+bold for best.
    for i, v in enumerate(means):
        is_best = GAMMA_ORDER[i] == BEST
        top = max(maxs[i], v + stds[i]) + 0.05
        ax.text(
            x[i],
            top,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold" if is_best else "normal",
            color=BEST_VALUE_COLOR if is_best else "black",
        )

    ax.axhline(0, color="#666", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([GAMMA_LABEL[n] for n in GAMMA_ORDER], fontsize=10.5)
    ax.set_xlabel("Robustness parameter in M2-robust loss", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Annualised Sharpe  ($\sqrt{12}\cdot \bar r/\sigma_r$)", fontsize=11, fontweight="bold")
    ax.set_title("(a) Mean Sharpe across 3 seeds (error bars ±1 std; whiskers min–max)", fontsize=12, pad=6)
    ax.set_ylim(-0.2, 2.0)
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(length=0, labelsize=10)


def draw_cv_panel(ax, agg: pd.DataFrame) -> None:
    x = np.arange(len(GAMMA_ORDER))
    cvs = agg["cv"].to_numpy()
    colors = [C_ABL[0], C_ABL[1], C_ABL[2], C_ABL[1], C_ABL[0]]
    hatches = ["", "", HATCH_BEST, "", ""]

    for i, (v, c, h) in enumerate(zip(cvs, colors, hatches)):
        ax.bar(
            x[i],
            v,
            width=0.62,
            color=c,
            hatch=h,
            edgecolor="white",
            linewidth=0.9,
            zorder=2,
        )

    for i, v in enumerate(cvs):
        is_best = GAMMA_ORDER[i] == BEST
        ax.text(
            x[i],
            v + 0.035,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold" if is_best else "normal",
            color=BEST_VALUE_COLOR if is_best else "black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([GAMMA_LABEL[n] for n in GAMMA_ORDER], fontsize=10.5)
    ax.set_xlabel("Robustness parameter in M2-robust loss", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"CV = $\sigma_S / |\mu_S|$", fontsize=11, fontweight="bold")
    ax.set_title("(b) Seed-stability (lower = more stable)", fontsize=12, pad=6)
    ax.set_ylim(0, max(cvs) * 1.3)
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(length=0, labelsize=10)


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(SRC)
    df = df[df["cap_tag"] == "cap05"].copy()
    agg = (
        df.groupby("loss")["long_short_sharpe"]
        .agg(["mean", "std", "min", "max", "count"])
        .reindex(GAMMA_ORDER)
    )
    agg["cv"] = agg["std"] / agg["mean"].abs()
    assert (agg["count"] == 3).all(), "Expected 3 seeds per gamma"

    fig, (ax_s, ax_cv) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.86, wspace=0.24)

    draw_sharpe_panel(ax_s, agg)
    draw_cv_panel(ax_cv, agg)

    fig.suptitle(
        "Figure 5.3 — Phase 2.2 γ refinement  (cap05 · 3 seeds · test 1995-01..1996-12)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
