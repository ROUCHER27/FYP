"""
Figure 5.8 — Sharpe retention under diagnostic component normalisation (slope plot).

Slope chart: two x-positions (Original, Normalised), y = mean annualised Sharpe.
Three series: gamma07, gamma10, alpha06.
Annotations: retention percentage on the right end of each slope line.

Data: doc/phase2-fix/phase2.2-fix1/phase2_summary.json
Output: paper/figures/fig5_4_normalisation_retention.png  (dpi=300)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import apply_paper_style, style_open_axes

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc/phase2-fix/phase2.2-fix1/phase2_summary.json"
OUT = ROOT / "paper/figures/fig5_4_normalisation_retention.png"

ORDER = ["m2_robust_gamma07", "m2_robust_gamma10", "imadl_m2_alpha06"]
LABEL = {
    "m2_robust_gamma07": r"$\gamma$07",
    "m2_robust_gamma10": r"$\gamma$10",
    "imadl_m2_alpha06":  r"$\alpha$06",
}
COLOR = {
    "m2_robust_gamma07": "#C0392B",   # red — recommended
    "m2_robust_gamma10": "#E67E22",   # orange
    "imadl_m2_alpha06":  "#27AE60",   # green — stable fallback
}
MARKER = {
    "m2_robust_gamma07": "*",
    "m2_robust_gamma10": "^",
    "imadl_m2_alpha06":  "D",
}
MARKER_SIZE = {
    "m2_robust_gamma07": 14,
    "m2_robust_gamma10": 10,
    "imadl_m2_alpha06":  9,
}

X_ORIG, X_NORM = 0.0, 1.0
X_LABELS = ["Original", "Normalised"]


def retention_str(orig: float, norm: float) -> str:
    if abs(orig) < 1e-9:
        return "n/a"
    pct = norm / orig * 100.0
    return f"{pct:.1f}%"


def main() -> None:
    apply_paper_style()
    data = json.loads(SRC.read_text())

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    fig.subplots_adjust(left=0.12, right=0.72, bottom=0.12, top=0.82)

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":", zorder=0)

    for name in ORDER:
        orig = data[name]["original_sharpe"]
        norm = data[name]["avg_normalized_sharpe"]
        c = COLOR[name]
        ms = MARKER_SIZE[name]
        mk = MARKER[name]

        # Slope line
        ax.plot([X_ORIG, X_NORM], [orig, norm],
                color=c, lw=2.0, zorder=3)
        # Markers
        ax.plot(X_ORIG, orig, marker=mk, markersize=ms, color=c,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.plot(X_NORM, norm, marker=mk, markersize=ms, color=c,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4,
                label=LABEL[name])

        # Value labels
        ax.text(X_ORIG - 0.04, orig, f"{orig:.3f}",
                ha="right", va="center", fontsize=9, color=c, fontweight="bold")
        ax.text(X_NORM + 0.04, norm, f"{norm:.3f}",
                ha="left", va="center", fontsize=9, color=c, fontweight="bold")

        # Retention annotation on the right
        ret = retention_str(orig, norm)
        ax.text(X_NORM + 0.22, (orig + norm) / 2,
                f"retention\n{ret}",
                ha="left", va="center", fontsize=8.5, color=c)

    ax.set_xticks([X_ORIG, X_NORM])
    ax.set_xticklabels(X_LABELS, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(r"Mean annualised Sharpe  ($\sqrt{12}\cdot\bar r/\sigma_r$)",
                  fontsize=10.5, fontweight="bold")
    ax.set_title("", pad=0)  # title carried by suptitle

    ax.legend(loc="upper left", fontsize=9.5, frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_open_axes(ax)
    ax.tick_params(direction="out", labelsize=9.5)

    fig.suptitle(
        "Figure 5.8 — Normalisation retention of leading candidates\n"
        "(cap05 · 3 seeds · test 1995-01..1996-12)",
        fontsize=10, fontweight="bold", y=0.99,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
