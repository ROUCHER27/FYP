"""
Figure 5.4 — IMADL-m2 α sweep (multi-seed, integrated Phase 2 summary).

Inputs : `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`
         (read via subprocess because the file lives on the `phase2.2-fix` branch).

Output : paper/figures/fig5_4_imadl_alpha_sweep.png

Left panel  : mean annualised Sharpe with ±1 std error bars across 3 seeds for
              α ∈ {0.2, 0.3, ..., 0.8}. gamma07 and gamma10 plotted for reference.
Right panel : coefficient of variation (log y-axis to compare strong vs unstable rows).

Evaluation window 1995-01..1996-12, cap05 portfolio, 3 seeds per row.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig5_4_imadl_alpha_sweep.png"

ALPHA_ORDER = [
    "imadl_m2_alpha02",
    "imadl_m2_alpha03",
    "imadl_m2_alpha04",
    "imadl_m2_alpha05",
    "imadl_m2_alpha06",
    "imadl_m2_alpha07",
    "imadl_m2_alpha08",
]
ALPHA_LABEL = {
    f"imadl_m2_alpha0{d}": rf"$\alpha$=0.{d}" for d in range(2, 9)
}

GAMMA_REF = ["m2_robust_gamma07", "m2_robust_gamma10"]
GAMMA_LABEL = {
    "m2_robust_gamma07": r"$\gamma$=0.07 (ref)",
    "m2_robust_gamma10": r"$\gamma$=0.10 (ref)",
}


def load_branch_csv() -> pd.DataFrame:
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
    return pd.read_csv(io.StringIO(out))


def load_local_gamma() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv"
    )


def main() -> None:
    df_int = load_branch_csv()
    df_int = df_int[df_int["cap_tag"] == "cap05"]
    df_loc = load_local_gamma()
    df_loc = df_loc[df_loc["cap_tag"] == "cap05"]
    df = pd.concat([df_int, df_loc], ignore_index=True)
    # Deduplicate on (loss, cap_tag); local gamma07 wins over any same-loss row.
    df = df.drop_duplicates(subset=["loss", "cap_tag"], keep="last")

    keep = ALPHA_ORDER + GAMMA_REF
    df = df[df["loss"].isin(keep)].set_index("loss").reindex(keep)
    assert (df["runs"] == 3).all(), f"Expected 3 seeds per row, got {df['runs'].to_dict()}"

    x_alpha = np.arange(len(ALPHA_ORDER))

    fig, (ax_s, ax_cv) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # --- Left: Sharpe mean ± std ---
    means = df.loc[ALPHA_ORDER, "sharpe_mean"].to_numpy()
    stds = df.loc[ALPHA_ORDER, "sharpe_std"].to_numpy()
    mins = df.loc[ALPHA_ORDER, "sharpe_min"].to_numpy()
    maxs = df.loc[ALPHA_ORDER, "sharpe_max"].to_numpy()

    ax_s.errorbar(
        x_alpha,
        means,
        yerr=stds,
        fmt="o",
        color="#1f4e79",
        ecolor="#1f4e79",
        capsize=5,
        elinewidth=1.4,
        markersize=7,
        label="IMADL-m2 α sweep (mean ± 1 std)",
        zorder=3,
    )
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        ax_s.vlines(x_alpha[i], lo, hi, color="#9ab7d9", linewidth=3, alpha=0.55, zorder=2)

    # Reference horizontal lines for gamma07 / gamma10 mean Sharpe.
    colours = {"m2_robust_gamma07": "#c0392b", "m2_robust_gamma10": "#f39c12"}
    for name in GAMMA_REF:
        val = df.loc[name, "sharpe_mean"]
        ax_s.axhline(
            val,
            color=colours[name],
            linewidth=1.1,
            linestyle="--",
            label=f"{GAMMA_LABEL[name]} = {val:.3f}",
        )

    # Highlight alpha06.
    idx_06 = ALPHA_ORDER.index("imadl_m2_alpha06")
    ax_s.scatter(
        [x_alpha[idx_06]],
        [means[idx_06]],
        s=170,
        facecolor="none",
        edgecolor="#c0392b",
        linewidth=1.8,
        zorder=4,
        label=r"fallback pick ($\alpha$=0.6)",
    )

    ax_s.axhline(0, color="#666", linewidth=0.8, linestyle=":")
    ax_s.set_xticks(x_alpha)
    ax_s.set_xticklabels([ALPHA_LABEL[n] for n in ALPHA_ORDER])
    ax_s.set_xlabel("IMADL-m2 α (directional rebalancing weight)")
    ax_s.set_ylabel(r"Annualised Sharpe")
    ax_s.set_title("IMADL-m2 α sweep vs γ reference")
    ax_s.grid(alpha=0.25)
    ax_s.legend(frameon=False, loc="lower right", fontsize=8)

    # --- Right: CV bars on log scale (some α rows have huge CVs) ---
    cvs = df.loc[ALPHA_ORDER, "sharpe_cv"].to_numpy()
    bar_colors = ["#b5b5b5"] * len(ALPHA_ORDER)
    bar_colors[idx_06] = "#c0392b"
    ax_cv.bar(x_alpha, cvs, color=bar_colors, edgecolor="#333", linewidth=0.6)
    for i, v in enumerate(cvs):
        ax_cv.text(i, v * 1.08, f"{v:.3f}", ha="center", fontsize=8.5)
    # Reference CVs for gamma07 / gamma10.
    for name in GAMMA_REF:
        ax_cv.axhline(
            df.loc[name, "sharpe_cv"],
            color=colours[name],
            linestyle="--",
            linewidth=1.1,
            label=f"{GAMMA_LABEL[name]} CV = {df.loc[name, 'sharpe_cv']:.3f}",
        )
    ax_cv.set_xticks(x_alpha)
    ax_cv.set_xticklabels([ALPHA_LABEL[n] for n in ALPHA_ORDER])
    ax_cv.set_xlabel("IMADL-m2 α")
    ax_cv.set_ylabel(r"CV = $\sigma_S / |\mu_S|$  (log scale)")
    ax_cv.set_yscale("log")
    ax_cv.set_ylim(0.1, max(cvs) * 2.0)
    ax_cv.set_title("Seed-stability of the IMADL-m2 α sweep")
    ax_cv.grid(axis="y", alpha=0.25, which="both")
    ax_cv.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle(
        "Figure 5.4 — IMADL-m2 α sweep (Phase 2 integrated summary; 3 seeds per row, cap05)",
        fontsize=11,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
