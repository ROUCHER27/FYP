"""
Figure 5.1 — Baseline loss comparison (seed 42, 24 months).

Two-panel paper-plot-skills figure combining the `line_confidence_band`
(Type A continuous) spine style on the left and `bar_grouped_hatch` (ablation
variant) on the right. M1 is the best single-seed baseline Sharpe and is
hatched on the right panel.

Left  : cumulative long-short return curve per loss over 1995-01..1996-12,
        seven coloured line series, zero reference dashed line.
Right : annualised Sharpe as horizontal bars, sorted descending, with M1
        hatched + dark-red bold value label.

Inputs : doc/final_report_all_24m_evidence/results/baseline/{loss}/
           sanity_metrics_{loss}.csv     (monthly long_short_return)
         doc/final_report_all_24m_evidence/results/baseline/{loss}/
           sanity_summary_{loss}.json    (long_short_sharpe)
Output : paper/figures/fig5_1_baseline_comparison.png  (dpi=300)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import (
    BEST_VALUE_COLOR,
    HATCH_BEST,
    apply_paper_style,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "doc/final_report_all_24m_evidence/results/baseline"
OUT = ROOT / "paper/figures/fig5_1_baseline_comparison.png"

LOSS_ORDER = ["mse", "medse", "madl", "gmadl", "imadl", "hybrid_mul_m1", "hybrid_mul_m2"]
LOSS_LABEL = {
    "mse": "MSE",
    "medse": "MedSE",
    "madl": "MADL",
    "gmadl": "GMADL",
    "imadl": "IMADL",
    "hybrid_mul_m1": "hybrid_mul_M1",
    "hybrid_mul_m2": "hybrid_mul_M2",
}
# 7-colour palette — deep blue for M1 (best), muted gradient for the rest.
LOSS_COLOR = {
    "mse": "#8C8C8C",
    "medse": "#5B7DB1",
    "madl": "#8A6FBF",
    "gmadl": "#5BBCCA",
    "imadl": "#D49A4F",
    "hybrid_mul_m1": "#1B3D6E",   # deep navy — best
    "hybrid_mul_m2": "#A8C8E8",
}
BEST = "hybrid_mul_m1"


def load_monthly_returns(loss: str) -> pd.Series:
    csv = SRC_ROOT / loss / f"sanity_metrics_{loss}.csv"
    df = pd.read_csv(csv, parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    return df.set_index("month")["long_short_return"]


def load_sharpe(loss: str) -> float:
    js = SRC_ROOT / loss / f"sanity_summary_{loss}.json"
    return float(json.loads(js.read_text())["long_short_sharpe"])


def draw_cumret(ax) -> None:
    for loss in LOSS_ORDER:
        r = load_monthly_returns(loss)
        cum = (1.0 + r.fillna(0.0)).cumprod() - 1.0
        is_best = loss == BEST
        ax.plot(
            cum.index,
            cum.values,
            color=LOSS_COLOR[loss],
            linewidth=2.2 if is_best else 1.6,
            label=LOSS_LABEL[loss],
            marker="o" if is_best else None,
            markersize=3.5,
            zorder=4 if is_best else 3,
        )

    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative long-short return", fontsize=11, fontweight="bold")
    ax.set_title("(a) Baseline cumulative long-short return", fontsize=12, pad=6)
    style_open_axes(ax)
    ax.tick_params(labelsize=9.5)
    # X-axis ticks: every 4 months.
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    # Legend to the right, no frame.
    leg = ax.legend(
        fontsize=8.5,
        loc="upper left",
        frameon=False,
        ncol=2,
        handlelength=1.5,
    )
    for text in leg.get_texts():
        if text.get_text() == LOSS_LABEL[BEST]:
            text.set_fontweight("bold")


def draw_sharpe(ax) -> None:
    # Sort descending by Sharpe.
    losses_sorted = sorted(LOSS_ORDER, key=lambda l: load_sharpe(l), reverse=True)
    values = [load_sharpe(l) for l in losses_sorted]
    labels = [LOSS_LABEL[l] for l in losses_sorted]
    colors = [LOSS_COLOR[l] for l in losses_sorted]
    hatches = [HATCH_BEST if l == BEST else "" for l in losses_sorted]
    y = np.arange(len(losses_sorted))[::-1]   # top item first

    for yi, v, c, h, l in zip(y, values, colors, hatches, losses_sorted):
        ax.barh(
            yi,
            v,
            color=c,
            hatch=h,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            height=0.65,
        )
        is_best = l == BEST
        ha = "left" if v >= 0 else "right"
        dx = 0.015 if v >= 0 else -0.015
        ax.text(
            v + dx,
            yi,
            f"{v:.4f}",
            va="center",
            ha=ha,
            fontsize=9.5,
            color=BEST_VALUE_COLOR if is_best else "black",
            fontweight="bold" if is_best else "normal",
        )

    ax.axvline(0, color="#555", linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Annualised Sharpe (seed 42)", fontsize=11, fontweight="bold")
    ax.set_title("(b) Baseline Sharpe, sorted descending", fontsize=12, pad=6)
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(length=0, labelsize=9.5)
    ax.set_xlim(min(values) * 1.3 - 0.05, max(values) * 1.25)


def main() -> None:
    apply_paper_style()
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.86, wspace=0.22)

    draw_cumret(ax_line)
    draw_sharpe(ax_bar)

    fig.suptitle(
        "Figure 5.1 — Baseline loss comparison  (seed 42 · cap05 · test 1995-01..1996-12)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
