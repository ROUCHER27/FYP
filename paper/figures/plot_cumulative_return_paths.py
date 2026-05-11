"""
Figure 5.9 — Cumulative long-short return paths.

Two panels (1×2), line_confidence_band style:
  Left  (a): Phase 1 baselines — seed-42 paths for MSE, MedSE, MADL, GMADL,
             IMADL, hybrid_mul_m1 (A3 also included as best additive).
  Right (b): Phase 2 gamma robustness — mean ± seed envelope for
             gamma03/05/07/10/15 (3 seeds each).

Data:
  Baselines : doc/final_report_all_24m_evidence/results/baseline/*/sanity_metrics_*.csv
  Phase 1.5 : doc/final_report_all_24m_evidence/results/phase15/*/sanity_metrics_*.csv
  Gamma     : doc/phase2-fix/phase2_2/gamma_refinement/results/*/sanity_metrics_*.csv

Output: paper/figures/fig5_9_cumulative_return_paths.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import apply_paper_style, style_open_axes

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "paper/figures/fig5_9_cumulative_return_paths.png"

# ── colour palette ────────────────────────────────────────────────────────────
BASELINE_COLORS = {
    "MSE":           "#A8C8E8",
    "MedSE":         "#5BBCCA",
    "MADL":          "#8B1A4A",
    "GMADL":         "#7D3C98",
    "IMADL":         "#E67E22",
    "hybrid_mul_m1": "#1B3D6E",
    "hybrid_add_a3": "#27AE60",
}
GAMMA_COLORS = {
    "gamma03": "#A8C8E8",
    "gamma05": "#5BBCCA",
    "gamma07": "#C0392B",   # recommended — red
    "gamma10": "#E67E22",   # orange
    "gamma15": "#95A5A6",
}
GAMMA_LW = {"gamma03": 1.4, "gamma05": 1.4, "gamma07": 2.4, "gamma10": 1.8, "gamma15": 1.4}


def load_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["month"])
    return df.set_index("month")["cumulative_long_short_return"]


# ── Panel A: baselines (seed 42) ──────────────────────────────────────────────
BASELINE_PATHS = {
    "MSE":           ROOT / "doc/final_report_all_24m_evidence/results/baseline/mse/sanity_metrics_mse.csv",
    "MedSE":         ROOT / "doc/final_report_all_24m_evidence/results/baseline/medse/sanity_metrics_medse.csv",
    "MADL":          ROOT / "doc/final_report_all_24m_evidence/results/baseline/madl/sanity_metrics_madl.csv",
    "GMADL":         ROOT / "doc/final_report_all_24m_evidence/results/baseline/gmadl/sanity_metrics_gmadl.csv",
    "IMADL":         ROOT / "doc/final_report_all_24m_evidence/results/baseline/imadl/sanity_metrics_imadl.csv",
    "hybrid_mul_m1": ROOT / "doc/final_report_all_24m_evidence/results/baseline/hybrid_mul_m1/sanity_metrics_hybrid_mul_m1.csv",
    "hybrid_add_a3": ROOT / "doc/final_report_all_24m_evidence/results/phase15/A3/sanity_metrics_hybrid_add_a3.csv",
}
BASELINE_LABELS = {
    "MSE": "MSE", "MedSE": "MedSE", "MADL": "MADL", "GMADL": "GMADL",
    "IMADL": "IMADL", "hybrid_mul_m1": "Hybrid M1", "hybrid_add_a3": "Hybrid A3",
}

# ── Panel B: gamma robustness (3 seeds) ───────────────────────────────────────
GAMMA_SEEDS = [42, 52, 62]
GAMMA_NAMES = ["gamma03", "gamma05", "gamma07", "gamma10", "gamma15"]
GAMMA_LABELS = {
    "gamma03": r"$\gamma$=0.03", "gamma05": r"$\gamma$=0.05",
    "gamma07": r"$\gamma$=0.07 (Rec.)", "gamma10": r"$\gamma$=0.10",
    "gamma15": r"$\gamma$=0.15",
}


def load_gamma_seeds(gamma: str) -> pd.DataFrame:
    frames = []
    for seed in GAMMA_SEEDS:
        p = ROOT / f"doc/phase2-fix/phase2_2/gamma_refinement/results/m2_robust_{gamma}_seed{seed}_cap05/sanity_metrics_m2_robust_{gamma}.csv"
        if p.exists():
            frames.append(load_csv(p))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def draw_baselines(ax: plt.Axes) -> None:
    for key, path in BASELINE_PATHS.items():
        if not path.exists():
            continue
        s = load_csv(path)
        lw = 2.2 if "hybrid" in key else 1.6
        ls = "-" if "hybrid" in key else "--" if key in ("MADL", "GMADL", "IMADL") else "-"
        ax.plot(s.index, s.values, color=BASELINE_COLORS[key],
                lw=lw, ls=ls, label=BASELINE_LABELS[key], zorder=3)

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":", zorder=1)
    ax.set_title("(a) Phase 1 baselines  (seed 42)", fontsize=11, pad=5)
    ax.set_ylabel("Cumulative long-short return", fontsize=10.5, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=8.5, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#CCCCCC", ncol=1)
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)


def draw_gamma(ax: plt.Axes) -> None:
    for gamma in GAMMA_NAMES:
        df = load_gamma_seeds(gamma)
        if df.empty:
            continue
        mean = df.mean(axis=1)
        lo   = df.min(axis=1)
        hi   = df.max(axis=1)
        c  = GAMMA_COLORS[gamma]
        lw = GAMMA_LW[gamma]
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=c, alpha=0.12, zorder=1)
        ax.plot(mean.index, mean.values, color=c, lw=lw,
                label=GAMMA_LABELS[gamma], zorder=3)

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":", zorder=1)
    ax.set_title(r"(b) Phase 2 $\gamma$ robustness  (mean ± seed range, 3 seeds)",
                 fontsize=11, pad=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=8.5, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_open_axes(ax)
    _style.y_grid_only(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)


def main() -> None:
    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.16, top=0.86, wspace=0.26)

    draw_baselines(ax1)
    draw_gamma(ax2)

    fig.suptitle(
        "Figure 5.9 — Cumulative long-short return paths  "
        "(train 1990-01..1994-12 · test 1995-01..1996-12 · cap05)",
        fontsize=10.5, fontweight="bold", y=0.99,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
