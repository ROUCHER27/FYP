"""
Figure 5.5 — Loss-component normalisation probe (Phase 2.2-fix1).

Inputs : doc/phase2-fix/phase2.2-fix1/phase2_summary.json
Output : paper/figures/fig5_5_normalisation_probe.png

Left panel : grouped bar — original vs normalised mean Sharpe per candidate.
Right panel: per-seed normalised Sharpes (3 dots per candidate) so the reader
             can see the dispersion that collapses the mean for gamma10 / alpha06.

Candidates: m2_robust_gamma07, m2_robust_gamma10, imadl_m2_alpha06.
Evaluation window 1995-01..1996-12, cap05 portfolio, 3 seeds per row.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc/phase2-fix/phase2.2-fix1/phase2_summary.json"
OUT = ROOT / "paper/figures/fig5_5_normalisation_probe.png"

ORDER = ["m2_robust_gamma07", "m2_robust_gamma10", "imadl_m2_alpha06"]
LABEL = {
    "m2_robust_gamma07": r"m2-robust $\gamma$=0.07",
    "m2_robust_gamma10": r"m2-robust $\gamma$=0.10",
    "imadl_m2_alpha06": r"imadl-m2 $\alpha$=0.6",
}


def main() -> None:
    data = json.loads(SRC.read_text())

    x = np.arange(len(ORDER))
    width = 0.36

    original = np.array([data[name]["original_sharpe"] for name in ORDER])
    normalised = np.array([data[name]["avg_normalized_sharpe"] for name in ORDER])
    per_seed = {name: data[name]["normalized_sharpes"] for name in ORDER}

    fig, (ax_bar, ax_dot) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), constrained_layout=True
    )

    # --- Left: grouped bar (original vs normalised) ---
    ax_bar.bar(
        x - width / 2,
        original,
        width,
        color="#1f4e79",
        edgecolor="#111",
        linewidth=0.6,
        label="original (no loss-component norm.)",
    )
    ax_bar.bar(
        x + width / 2,
        normalised,
        width,
        color="#c0392b",
        edgecolor="#111",
        linewidth=0.6,
        label="normalised",
    )

    for xi, v in zip(x - width / 2, original):
        ax_bar.text(xi, v + 0.03 if v > 0 else v - 0.05, f"{v:.3f}",
                    ha="center", fontsize=8.5)
    for xi, v in zip(x + width / 2, normalised):
        ax_bar.text(xi, v + 0.03 if v > 0 else v - 0.05, f"{v:.3f}",
                    ha="center", fontsize=8.5)

    ax_bar.axhline(0, color="#666", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABEL[n] for n in ORDER])
    ax_bar.set_ylabel(r"Mean annualised Sharpe (3 seeds, cap05)")
    ax_bar.set_title("Loss-component normalisation: mean Sharpe")
    ax_bar.legend(frameon=False, fontsize=9, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.25)

    # --- Right: per-seed normalised Sharpes (dots) ---
    jitter = np.linspace(-0.12, 0.12, 3)
    for i, name in enumerate(ORDER):
        vals = per_seed[name]
        ax_dot.scatter(
            [i] * len(vals) + jitter,
            vals,
            s=55,
            color="#c0392b",
            edgecolor="#333",
            linewidth=0.6,
            zorder=3,
        )
        ax_dot.hlines(
            normalised[i],
            i - 0.22,
            i + 0.22,
            color="#c0392b",
            linewidth=2.2,
            zorder=4,
        )
        for j, v in zip(jitter, vals):
            ax_dot.text(i + j, v + 0.06, f"{v:.3f}", fontsize=8, ha="center")

    ax_dot.axhline(0, color="#666", linewidth=0.8)
    ax_dot.set_xticks(np.arange(len(ORDER)))
    ax_dot.set_xticklabels([LABEL[n] for n in ORDER])
    ax_dot.set_ylabel("Normalised annualised Sharpe (per seed)")
    ax_dot.set_title("Per-seed dispersion under normalisation")
    ax_dot.grid(axis="y", alpha=0.25)

    note = (
        "Scale ratios (diagnostics-estimated): gamma07 ≈ 113, gamma10 ≈ 113, alpha06 ≈ 34.\n"
        "Only gamma07 is approximately flat under normalisation; gamma10 and alpha06 degrade."
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=8.8, color="#333")

    fig.suptitle(
        "Figure 5.5 — Phase 2.2-fix1 normalisation probe (test 1995-01..1996-12, 3 seeds per row)",
        fontsize=11,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
