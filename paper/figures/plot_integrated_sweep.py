"""
Figure 5.5 — IMADL-m2 α sweep vs γ reference.

Paper-plot-skills style: `line_confidence_band` (Type B, discrete scaling
curve). Equi-spaced x positions with manual tick labels, marker-per-point,
open-axis style (only left/bottom spines visible), STIX serif, framealpha=0
legend.

Left  : α sweep mean annualised Sharpe with ±1 std shaded band and markers;
        dashed horizontal reference lines for γ07 and γ10 (distinct colours).
Right : coefficient of variation on a log y-axis for the same α values with
        γ07 / γ10 reference lines, exposing the wide CV range of β / λ-style
        families (here: only the α sweep + γ refs to keep the chart focused).

Inputs : integrated summary `phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`
         and local `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`
         (for γ07 reference row).
Output : paper/figures/fig5_5_imadl_alpha_sweep.png (dpi=300)
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import (
    C_LINE_ALT,
    C_LINE_BASE,
    C_LINE_MAIN,
    C_REF_HLINE,
    apply_paper_style,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig5_5_imadl_alpha_sweep.png"

ALPHA_ORDER = [
    "imadl_m2_alpha02",
    "imadl_m2_alpha03",
    "imadl_m2_alpha04",
    "imadl_m2_alpha05",
    "imadl_m2_alpha06",
    "imadl_m2_alpha07",
    "imadl_m2_alpha08",
]
ALPHA_TICK = [r"$\alpha$=0.2", r"$\alpha$=0.3", r"$\alpha$=0.4",
              r"$\alpha$=0.5", r"$\alpha$=0.6", r"$\alpha$=0.7", r"$\alpha$=0.8"]

GAMMA_REF = ["m2_robust_gamma07", "m2_robust_gamma10"]
GAMMA_LABEL = {
    "m2_robust_gamma07": r"m2-robust $\gamma$=0.7 (ref)",
    "m2_robust_gamma10": r"m2-robust $\gamma$=1.0 (ref)",
}
COLOURS_REF = {"m2_robust_gamma07": C_REF_HLINE, "m2_robust_gamma10": "#E58C00"}


def load_data() -> pd.DataFrame:
    out = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            "phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv",
        ],
        text=True,
    )
    integrated = pd.read_csv(io.StringIO(out))
    local = pd.read_csv(
        ROOT / "doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv"
    )
    df = pd.concat([integrated, local], ignore_index=True)
    df = df[df["cap_tag"] == "cap05"].drop_duplicates(
        subset=["loss", "cap_tag"], keep="last"
    )
    keep = ALPHA_ORDER + GAMMA_REF
    return df[df["loss"].isin(keep)].set_index("loss").reindex(keep)


def draw_sharpe(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(ALPHA_ORDER))
    means = df.loc[ALPHA_ORDER, "sharpe_mean"].to_numpy()
    stds = df.loc[ALPHA_ORDER, "sharpe_std"].to_numpy()

    # Confidence band + main curve (main = alpha sweep, green)
    ax.fill_between(
        x,
        means - stds,
        means + stds,
        color=C_LINE_MAIN,
        alpha=0.15,
        zorder=2,
    )
    ax.plot(
        x,
        means,
        marker="o",
        markersize=7,
        color=C_LINE_MAIN,
        linewidth=1.8,
        zorder=3,
        label=r"IMADL-m2 $\alpha$ sweep (mean ± 1 std, 3 seeds)",
    )

    # Reference horizontal lines for gamma07 / gamma10 mean Sharpe
    for name in GAMMA_REF:
        val = float(df.loc[name, "sharpe_mean"])
        ax.axhline(
            val,
            color=COLOURS_REF[name],
            lw=1.4,
            ls="--",
            zorder=2,
            label=f"{GAMMA_LABEL[name]}  = {val:.3f}",
        )

    # Highlight alpha06 with a ring and a bold label
    idx_06 = ALPHA_ORDER.index("imadl_m2_alpha06")
    ax.scatter(
        [x[idx_06]],
        [means[idx_06]],
        s=190,
        facecolor="none",
        edgecolor="#8B0000",
        linewidth=1.8,
        zorder=5,
    )
    # Annotation placed clear of the red ring, connected by an arrow.
    ax.annotate(
        f"peak  α = 0.6\nmean = {means[idx_06]:.3f}\nCV = {float(df.loc['imadl_m2_alpha06', 'sharpe_cv']):.3f}",
        xy=(x[idx_06] + 0.08, means[idx_06] + 0.04),
        xytext=(x[idx_06] - 1.35, means[idx_06] - 0.55),
        fontsize=8.8,
        color="#8B0000",
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#8B0000",
            linewidth=0.9,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="#8B0000",
            linewidth=1.1,
            connectionstyle="arc3,rad=-0.2",
        ),
        zorder=6,
    )

    ax.axhline(0, color="#888", lw=0.7, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ALPHA_TICK, fontsize=10)
    ax.set_xlabel(r"IMADL-m2 directional-rebalancing weight $\alpha$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualised Sharpe", fontsize=11, fontweight="bold")
    ax.set_title(r"(a) IMADL-m2 $\alpha$ sweep vs $\gamma$ reference", fontsize=12, pad=6)
    style_open_axes(ax)
    leg = ax.legend(
        fontsize=9,
        loc="upper left",
        frameon=False,
    )
    for text in leg.get_texts():
        if r"$\alpha$ sweep" in text.get_text():
            text.set_fontweight("bold")
    ax.tick_params(labelsize=9.5)


def draw_cv(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(ALPHA_ORDER))
    cvs = df.loc[ALPHA_ORDER, "sharpe_cv"].to_numpy()

    ax.plot(
        x,
        cvs,
        marker="s",
        markersize=6,
        color=C_LINE_ALT,
        linewidth=1.8,
        zorder=3,
        label=r"IMADL-m2 $\alpha$ sweep CV",
    )

    for name in GAMMA_REF:
        val = float(df.loc[name, "sharpe_cv"])
        ax.axhline(
            val,
            color=COLOURS_REF[name],
            lw=1.4,
            ls="--",
            zorder=2,
            label=f"{GAMMA_LABEL[name]}  = {val:.3f}",
        )

    idx_06 = ALPHA_ORDER.index("imadl_m2_alpha06")
    ax.scatter(
        [x[idx_06]],
        [cvs[idx_06]],
        s=170,
        facecolor="none",
        edgecolor="#8B0000",
        linewidth=1.8,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ALPHA_TICK, fontsize=10)
    ax.set_xlabel(r"IMADL-m2 directional-rebalancing weight $\alpha$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"CV = $\sigma_S / |\mu_S|$  (log scale)", fontsize=11, fontweight="bold")
    ax.set_title(r"(b) Seed-stability of the IMADL-m2 $\alpha$ sweep", fontsize=12, pad=6)
    ax.set_yscale("log")
    ax.set_ylim(0.1, max(cvs) * 2.2)
    style_open_axes(ax)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9, loc="upper left", frameon=False)


def main() -> None:
    apply_paper_style()
    df = load_data()
    assert (df["runs"] == 3).all(), "Expected 3 seeds per row"

    fig, (ax_s, ax_cv) = plt.subplots(1, 2, figsize=(12.0, 4.7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.86, wspace=0.28)

    draw_sharpe(ax_s, df)
    draw_cv(ax_cv, df)

    fig.suptitle(
        "Figure 5.5 — IMADL-m2 α sweep (Phase 2 integrated summary · cap05 · 3 seeds per row)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
