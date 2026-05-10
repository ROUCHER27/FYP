"""
Figure 5.2 — Phase 1.5 additive (A1–A5) and multiplicative (M1–M4) variants.

Paper-plot-skills style: `bar_grouped_hatch` (comparison variant, horizontal
orientation). Two grouped horizontal bar sets: additive A-series (warm palette)
and multiplicative M-series (cool palette). Seed-42 peaks A3 and M1 are
hatched + dark-red bold value labels.

Inputs : doc/final_report_all_24m_evidence/results/phase15/{A1..A5,M1..M4}/
           sanity_summary_*.json    (field: long_short_sharpe)
Output : paper/figures/fig5_2_phase15_variants.png (dpi=300)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import (
    BEST_VALUE_COLOR,
    HATCH_BEST,
    apply_paper_style,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "doc/final_report_all_24m_evidence/results/phase15"
OUT = ROOT / "paper/figures/fig5_2_phase15_variants.png"

ADDITIVE = {
    "A1": "hybrid_add_a1",
    "A2": "hybrid_add_a2",
    "A3": "hybrid_add_a3",
    "A4": "hybrid_add_a4",
    "A5": "hybrid_add_a5",
}
MULTIPLICATIVE = {
    "M1": "hybrid_mul_m1",
    "M2": "hybrid_mul_m2",
    "M3": "hybrid_mul_m3",
    "M4": "hybrid_mul_m4",
}

BEST_A = "A3"
BEST_M = "M1"

# Warm palette for additive (lighter → deeper), cool palette for multiplicative.
COLOR_A = ["#F5C5A3", "#E8845A", "#C0392B", "#EF9170", "#D17860"]   # peak at A3
COLOR_M = ["#1B3D6E", "#5B7DB1", "#A8C8E8", "#C8DCEF"]               # peak at M1


def sharpe_for(label: str, artifact: str, group_root: str) -> float:
    js = SRC_ROOT / label / f"sanity_summary_{artifact}.json"
    return float(json.loads(js.read_text())["long_short_sharpe"])


def draw_group(
    ax,
    title: str,
    variants: dict,
    colors: list,
    best_key: str,
    label_prefix: str = "",
) -> None:
    labels = list(variants.keys())
    values = [sharpe_for(k, variants[k], label_prefix) for k in labels]
    y = np.arange(len(labels))[::-1]   # first item at top
    hatches = [HATCH_BEST if k == best_key else "" for k in labels]

    for yi, v, c, h, k in zip(y, values, colors, hatches, labels):
        ax.barh(
            yi,
            v,
            color=c,
            hatch=h,
            edgecolor="white",
            linewidth=0.9,
            height=0.6,
            zorder=3,
        )
        is_best = k == best_key
        ha = "left" if v >= 0 else "right"
        dx = 0.008 if v >= 0 else -0.008
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
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Annualised Sharpe (seed 42)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlim(min(values + [0]) * 1.25 - 0.05, max(values + [0]) * 1.3 + 0.05)
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(length=0, labelsize=9.5)


def main() -> None:
    apply_paper_style()
    fig, (ax_a, ax_m) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.86, wspace=0.22)

    draw_group(
        ax_a,
        "(a) Additive hybrids (A-series)",
        ADDITIVE,
        COLOR_A,
        BEST_A,
    )
    draw_group(
        ax_m,
        "(b) Multiplicative hybrids (M-series)",
        MULTIPLICATIVE,
        COLOR_M,
        BEST_M,
    )

    fig.suptitle(
        "Figure 5.2 — Phase 1.5 hybrid variants  (seed 42 · cap05 · test 1995-01..1996-12)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
