"""
Figure 5.7 — Loss-component normalisation probe (Phase 2.2-fix1).

Paper-plot-skills style: `bar_paired_delta` (paired bars + arrow + red delta
annotation + horizontal reference dashed line at the baseline).

Left panel: grouped paired bars (original vs normalised) for the three
            candidates. Arrow from baseline top to method top; red
            percentage-change annotation.
Right panel: per-seed normalised Sharpes as dots (same candidate column)
            to expose the dispersion that collapses the mean for gamma10
            and alpha06. Open-axis style.

Inputs : doc/phase2-fix/phase2.2-fix1/phase2_summary.json
Output : paper/figures/fig5_7_normalisation_probe.png (dpi=300)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import (
    C_BASELINE,
    C_DELTA,
    C_METHOD,
    apply_paper_style,
    style_framed_axes,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc/phase2-fix/phase2.2-fix1/phase2_summary.json"
OUT = ROOT / "paper/figures/fig5_7_normalisation_probe.png"

ORDER = ["m2_robust_gamma07", "m2_robust_gamma10", "imadl_m2_alpha06"]
LABEL = {
    "m2_robust_gamma07": r"m2-robust $\gamma$=0.7",
    "m2_robust_gamma10": r"m2-robust $\gamma$=1.0",
    "imadl_m2_alpha06": r"imadl-m2 $\alpha$=0.6",
}


def main() -> None:
    apply_paper_style()
    data = json.loads(SRC.read_text())

    original = np.array([data[n]["original_sharpe"] for n in ORDER])
    normalised = np.array([data[n]["avg_normalized_sharpe"] for n in ORDER])
    per_seed = {n: data[n]["normalized_sharpes"] for n in ORDER}

    # Delta strings: relative change, with sign.
    delta = []
    for o, nn in zip(original, normalised):
        pct = (nn - o) / abs(o) * 100.0
        sign = "+" if pct > 0 else ""
        delta.append(f"{sign}{pct:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.88, wspace=0.28)

    # --- Left: bar_paired_delta ---
    n = len(ORDER)
    x = np.arange(n)
    BAR_W = 0.32
    GAP = 0.02

    # Bars
    ax1.bar(
        x - (BAR_W + GAP) / 2,
        original,
        width=BAR_W,
        color=C_BASELINE,
        zorder=3,
        label="original",
    )
    ax1.bar(
        x + (BAR_W + GAP) / 2,
        normalised,
        width=BAR_W,
        color=C_METHOD,
        zorder=3,
        label="normalised",
    )

    # Reference dashed line at each baseline top + arrow + red delta label.
    for i, (bl, me, d) in enumerate(zip(original, normalised, delta)):
        # Horizontal dashed line from baseline bar to method bar top.
        ax1.plot(
            [x[i] - BAR_W, x[i] + BAR_W + GAP / 2],
            [bl, bl],
            color="black",
            lw=0.9,
            ls="--",
            zorder=4,
        )
        # Only draw a directed arrow when the change is material (≥ 0.06 in
        # absolute Sharpe). Below that, the tiny delta produces messy glyphs.
        if abs(me - bl) >= 0.06:
            y_start = bl + (-0.02 if me < bl else 0.02)
            y_end = me + (0.02 if me < bl else -0.02)
            ax1.annotate(
                "",
                xy=(x[i] + (BAR_W + GAP) / 2, y_end),
                xytext=(x[i] + (BAR_W + GAP) / 2, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                zorder=5,
            )
        # Red delta label
        y_label = max(bl, me) + 0.06
        ax1.text(
            x[i] + (BAR_W + GAP) / 2,
            y_label,
            d,
            color=C_DELTA,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        # Underlying value labels (original above bar, normalised inside if tall
        # enough, below the tiny negative bar otherwise).
        ax1.text(
            x[i] - (BAR_W + GAP) / 2,
            bl + (0.03 if bl >= 0 else -0.05),
            f"{bl:.3f}",
            ha="center",
            va="bottom" if bl >= 0 else "top",
            fontsize=8.5,
            color="black",
        )
        ax1.text(
            x[i] + (BAR_W + GAP) / 2,
            me + (0.03 if me >= 0.04 else -0.05),
            f"{me:.3f}",
            ha="center",
            va="bottom" if me >= 0.04 else "top",
            fontsize=8.5,
            color="white" if me >= 0.25 else "black",
            fontweight="bold",
        )

    ax1.axhline(0, color="#666", linewidth=0.8, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABEL[n] for n in ORDER], fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Mean annualised Sharpe", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(-0.25, 1.3)
    ax1.set_title("(a) Original vs normalised mean Sharpe", fontsize=12, pad=6)
    style_framed_axes(ax1)
    ax1.tick_params(length=0, labelsize=9.5)
    ax1.set_axisbelow(True)
    ax1.legend(
        fontsize=9,
        loc="upper right",
        framealpha=1.0,
        edgecolor="#C8C8C8",
        fancybox=False,
    )

    # --- Right: per-seed normalised dots with bucket mean ---
    jitter = np.linspace(-0.12, 0.12, 3)
    for i, name in enumerate(ORDER):
        vals = per_seed[name]
        ax2.scatter(
            [i] * len(vals) + jitter,
            vals,
            s=60,
            color=C_METHOD,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax2.hlines(
            normalised[i],
            i - 0.22,
            i + 0.22,
            color=C_DELTA,
            linewidth=2.2,
            zorder=4,
            label="normalised mean" if i == 0 else None,
        )
        for j, v in zip(jitter, vals):
            ax2.text(
                i + j,
                v + (0.06 if v >= 0 else -0.06),
                f"{v:.3f}",
                fontsize=7.8,
                ha="center",
                va="bottom" if v >= 0 else "top",
                color="#444",
            )

    ax2.axhline(0, color="#666", linewidth=0.8)
    ax2.set_xticks(np.arange(len(ORDER)))
    ax2.set_xticklabels([LABEL[n] for n in ORDER], fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Per-seed normalised annualised Sharpe", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Per-seed dispersion under normalisation", fontsize=12, pad=6)
    style_open_axes(ax2)
    _style.y_grid_only(ax2)
    ax2.tick_params(labelsize=9.5)
    ax2.set_ylim(-1.1, 1.7)
    ax2.legend(
        fontsize=9,
        loc="upper right",
        framealpha=1.0,
        edgecolor="#C8C8C8",
        fancybox=False,
    )

    fig.suptitle(
        "Figure 5.7 — Phase 2.2-fix1 normalisation probe (cap05, 3 seeds per row)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
