"""
Figure 5.6 — Sharpe-stability frontier across multi-seed hybrid families.

Scatter plot (scatter_broken_axis style):
  x-axis: Cross-seed CV (lower = more stable)
  y-axis: Mean annualised Sharpe
  marker size: Mean cumulative return (scaled)
  colour: loss family

Highlights:
  gamma07  — red star   "Recommended"
  gamma10  — orange ▲   "High-return, unstable"
  alpha06  — green ◆    "Stable fallback"

Extreme beta rows (CV > 10) are excluded from the main panel and noted in caption.

Data sources:
  - Gamma rows: doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv
  - Integrated rows: embedded from results_source_of_truth.md §4
    (git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv)

Output: paper/figures/fig5_3_sharpe_cv_frontier.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import _style
from _style import apply_paper_style, style_open_axes

ROOT = Path(__file__).resolve().parents[2]
GAMMA_SRC = ROOT / "doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv"
OUT = ROOT / "paper/figures/fig5_3_sharpe_cv_frontier.png"

# ── Integrated rows from phase2.2-fix branch (results_source_of_truth.md §4) ─
# Embedded to avoid branch switching at runtime.
INTEGRATED_ROWS = [
    # loss, sharpe_mean, sharpe_cv, cumulative_return_mean
    ("adaptive_lambda10",   0.4938, 1.5426, 0.0618),
    ("adaptive_lambda50",   0.2763, 0.5780, 0.0817),
    ("adaptive_lambda100",  0.0955, 0.3591, 0.0021),
    ("imadl_gmadl_beta03", -0.0345, 4.0084, -0.0424),
    ("imadl_gmadl_beta05",  0.0406, 10.133, -0.0138),
    # beta07 CV=139 excluded (extreme)
    ("imadl_m2_alpha02",    0.1788, 6.4735, 0.2225),
    ("imadl_m2_alpha03",    0.2159, 0.9310, 0.0588),
    ("imadl_m2_alpha04",    0.3540, 0.1853, 0.0962),
    ("imadl_m2_alpha05",    0.5822, 0.5484, 0.2465),
    ("imadl_m2_alpha06",    0.6895, 0.2443, 0.3042),
    ("imadl_m2_alpha07",    0.4024, 0.6128, 0.1036),
    ("imadl_m2_alpha08",    0.5683, 0.7267, 0.2071),
    ("m2_robust_gamma001",  0.6919, 1.1936, 0.1705),
    ("m2_robust_gamma01",   0.7470, 0.5270, 0.2718),
    ("m2_robust_gamma10",   1.0043, 0.5613, 0.2368),
]

# ── Family classification ─────────────────────────────────────────────────────
def classify(name: str) -> str:
    if name.startswith("m2_robust"):
        return "M2-robust gamma"
    if name.startswith("imadl_m2"):
        return "IMADL-m2 alpha"
    if name.startswith("adaptive"):
        return "Adaptive lambda"
    if name.startswith("imadl_gmadl"):
        return "IMADL-GMADL beta"
    return "Other"


FAMILY_COLOR = {
    "M2-robust gamma":  "#1B3D6E",   # deep navy
    "IMADL-m2 alpha":   "#27AE60",   # green
    "Adaptive lambda":  "#95A5A6",   # grey
    "IMADL-GMADL beta": "#E67E22",   # muted orange
}
FAMILY_MARKER = {
    "M2-robust gamma":  "o",
    "IMADL-m2 alpha":   "s",
    "Adaptive lambda":  "D",
    "IMADL-GMADL beta": "v",
}

BEST = "m2_robust_gamma07"
ALT  = "m2_robust_gamma10"
FALLBACK = "imadl_m2_alpha06"

CV_CUTOFF = 8.0   # exclude extreme-CV rows from main panel


def build_df() -> pd.DataFrame:
    # Gamma rows from local CSV
    gdf = pd.read_csv(GAMMA_SRC)
    gdf = gdf[gdf["cap_tag"] == "cap05"][
        ["loss", "sharpe_mean", "sharpe_cv", "cumulative_return_mean"]
    ].copy()

    # Integrated rows (embedded)
    idf = pd.DataFrame(INTEGRATED_ROWS,
                       columns=["loss", "sharpe_mean", "sharpe_cv", "cumulative_return_mean"])

    # Merge: gamma rows take precedence (they are the authoritative local source)
    df = pd.concat([gdf, idf], ignore_index=True)
    df = df.drop_duplicates(subset="loss", keep="first")
    df["family"] = df["loss"].map(classify)
    return df


def main() -> None:
    apply_paper_style()
    df = build_df()

    # Split: main panel (CV ≤ cutoff) vs excluded
    main_df = df[df["sharpe_cv"] <= CV_CUTOFF].copy()
    excl_df = df[df["sharpe_cv"] > CV_CUTOFF].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.88)

    # Preferred region shading — label placed after axes limits are known (deferred below)
    ax.axvspan(0, 0.35, ymin=0, ymax=1, color="#EBF5FB", alpha=0.55, zorder=0)

    # Plot each family
    for family, fdf in main_df.groupby("family"):
        # Exclude the three highlighted points from the bulk scatter
        bulk = fdf[~fdf["loss"].isin([BEST, ALT, FALLBACK])]
        size = np.clip(np.abs(bulk["cumulative_return_mean"]) * 800, 30, 300)
        ax.scatter(
            bulk["sharpe_cv"], bulk["sharpe_mean"],
            s=size,
            color=FAMILY_COLOR[family],
            marker=FAMILY_MARKER[family],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=family,
        )

    # ── Highlighted points ────────────────────────────────────────────────────
    def _highlight(loss, marker, color, edge, size, label, text, text_offset,
                   fontsize=9, curved=False):
        row = main_df[main_df["loss"] == loss]
        if row.empty:
            return
        x, y = float(row["sharpe_cv"].iloc[0]), float(row["sharpe_mean"].iloc[0])
        ax.scatter(x, y, s=size, marker=marker, color=color,
                   edgecolors=edge, linewidths=1.0, zorder=6, label=label)
        arrowprops = dict(
            arrowstyle="->", color=color, lw=0.9,
            connectionstyle="arc3,rad=0.35" if curved else "arc3,rad=0.0",
        )
        ax.annotate(text, xy=(x, y),
                    xytext=(x + text_offset[0], y + text_offset[1]),
                    fontsize=fontsize, color=color, fontweight="bold",
                    arrowprops=arrowprops)

    _highlight(BEST,     "*", "#C0392B", "black", 280,
               r"$\gamma$07 (Recommended)",
               "Recommended\n" + r"$\gamma$=0.07",
               (-0.12, 0.13))
    _highlight(ALT,      "^", "#E67E22", "black", 160,
               r"$\gamma$10 (High-return, unstable)",
               "High-return,\nunstable",
               (0.18, -0.18))                        # moved down
    _highlight(FALLBACK, "D", "#27AE60", "black", 140,
               r"$\alpha$06 (Stable fallback)",
               "Stable\nfallback",
               (0.55, -0.05),                        # moved right
               fontsize=8, curved=True)              # smaller font, curved arrow

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"Cross-seed CV = $\sigma_S / |\mu_S|$  (lower = more stable)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Mean annualised Sharpe  ($\sqrt{12}\cdot\bar r/\sigma_r$)",
                  fontsize=11, fontweight="bold")
    ax.axhline(0, color="#888", lw=0.8, ls=":", zorder=1)

    # Preferred region label — placed now that ylim is determined
    ylo, yhi = ax.get_ylim()
    ax.text(0.175, yhi - 0.04 * (yhi - ylo),
            "Preferred\nregion", ha="center", va="top", fontsize=9,
            color="#1A5276", style="italic", zorder=5)

    # Note excluded rows
    if not excl_df.empty:
        names = ", ".join(excl_df["loss"].tolist())
        ax.text(0.99, 0.02,
                f"Excluded (CV > {CV_CUTOFF}): {names}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#888", style="italic")

    # Legend: families + size note
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    size_patch = mpatches.Patch(color="none",
                                label="Marker size ∝ mean cumulative return")
    ax.legend(handles=h2 + [size_patch], labels=l2 + ["Marker size ∝ mean cum. return"],
              loc="upper right", fontsize=8.5, frameon=True,
              facecolor="white", edgecolor="#CCCCCC")

    style_open_axes(ax)
    ax.tick_params(direction="out", labelsize=9.5)

    fig.suptitle(
        "Figure 5.6 — Sharpe-stability frontier across multi-seed hybrid families  "
        "(cap05 · 3 seeds · test 1995-01..1996-12)",
        fontsize=10.5,
        fontweight="bold",
        y=0.97,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
