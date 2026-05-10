"""
Figure 5.3 — Phase 2.2 γ refinement (multi-seed).

Inputs : doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv
Output : paper/figures/fig5_3_gamma_refinement.png

Left panel : mean annualised Sharpe (sqrt(12)-scaled monthly) with ±1 std
             error bars across three seeds, and a min–max range line.
Right panel: coefficient of variation  CV = std / |mean|.

Evaluation window 1995-01..1996-12, cap05 portfolio, MLP[64,32,16]+ReLU+dropout 0.2,
feature set X1, batch 1024, 20 epochs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "m2_robust_gamma03": r"$\gamma$=0.03",
    "m2_robust_gamma05": r"$\gamma$=0.05",
    "m2_robust_gamma07": r"$\gamma$=0.07",
    "m2_robust_gamma10": r"$\gamma$=0.10",
    "m2_robust_gamma15": r"$\gamma$=0.15",
}


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["cap_tag"] == "cap05"].copy()

    # Aggregate across 3 seeds.
    agg = (
        df.groupby("loss")["long_short_sharpe"]
        .agg(["mean", "std", "min", "max", "count"])
        .reindex(GAMMA_ORDER)
    )
    agg["cv"] = agg["std"] / agg["mean"].abs()

    assert (agg["count"] == 3).all(), "Expected 3 seeds per gamma"

    x_pos = np.arange(len(GAMMA_ORDER))
    labels = [GAMMA_LABEL[name] for name in GAMMA_ORDER]

    fig, (ax_sharpe, ax_cv) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), constrained_layout=True
    )

    # --- Left: mean Sharpe with ±1 std error bars + min–max range ---
    ax_sharpe.errorbar(
        x_pos,
        agg["mean"],
        yerr=agg["std"],
        fmt="o",
        color="#1f4e79",
        ecolor="#1f4e79",
        elinewidth=1.4,
        capsize=5,
        markersize=7,
        label="mean ± 1 std (3 seeds)",
        zorder=3,
    )
    # Min–max range as a thin vertical line.
    for i, (lo, hi) in enumerate(zip(agg["min"], agg["max"])):
        ax_sharpe.vlines(x_pos[i], lo, hi, color="#9ab7d9", linewidth=3, alpha=0.55, zorder=2)

    # Highlight gamma07.
    idx_07 = GAMMA_ORDER.index("m2_robust_gamma07")
    ax_sharpe.scatter(
        [x_pos[idx_07]],
        [agg["mean"].iloc[idx_07]],
        s=160,
        facecolor="none",
        edgecolor="#c0392b",
        linewidth=1.8,
        zorder=4,
        label=r"recommended ($\gamma$=0.07)",
    )

    ax_sharpe.axhline(0, color="#666", linewidth=0.8, linestyle=":")
    ax_sharpe.set_xticks(x_pos)
    ax_sharpe.set_xticklabels(labels)
    ax_sharpe.set_xlabel("Robustness parameter in M2-robust loss")
    ax_sharpe.set_ylabel(r"Annualised Sharpe ($\sqrt{12}\cdot \bar r/\sigma_r$)")
    ax_sharpe.set_title("Mean Sharpe across seeds")
    ax_sharpe.legend(frameon=False, loc="lower right", fontsize=9)
    ax_sharpe.grid(alpha=0.25)

    # --- Right: coefficient of variation bar chart ---
    bar_colors = ["#b5b5b5"] * len(GAMMA_ORDER)
    bar_colors[idx_07] = "#c0392b"
    ax_cv.bar(x_pos, agg["cv"], color=bar_colors, edgecolor="#333", linewidth=0.6)
    for i, v in enumerate(agg["cv"]):
        ax_cv.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax_cv.set_xticks(x_pos)
    ax_cv.set_xticklabels(labels)
    ax_cv.set_xlabel("Robustness parameter in M2-robust loss")
    ax_cv.set_ylabel(r"CV = $\sigma_S / |\mu_S|$")
    ax_cv.set_title("Seed-stability (lower = more stable)")
    ax_cv.set_ylim(0, max(agg["cv"]) * 1.25)
    ax_cv.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Figure 5.3 — Phase 2.2 γ refinement: multi-seed Sharpe and stability "
        "(test 1995-01..1996-12, cap05, 3 seeds per γ)",
        fontsize=11,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
