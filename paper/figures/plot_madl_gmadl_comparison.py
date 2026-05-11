"""
Figure 2.2 — MADL vs GMADL response for small and large realised returns.

1×2 line plot, line_training_curve style:
  Left  panel: y = +0.01  (small positive return)
  Right panel: y = -0.80  (large negative return)

Each panel shows MADL and GMADL loss as a function of predicted return ŷ ∈ [-1, 1].
Vertical dashed line at ŷ = 0 marks the sign-switching point.
Horizontal dashed lines mark the saturation levels.

Illustrative; closed-form components from §3.3 / Appendix A; no training data.
Output: paper/figures/fig2_2_madl_gmadl_comparison.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import apply_paper_style, style_framed_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig2_2_madl_gmadl_comparison.png"

# ── loss parameters (match losses.py) ────────────────────────────────────────
TEMP = 25.0   # MADL temperature
A, B = 100.0, 2.0  # GMADL

YHAT = np.linspace(-1.0, 1.0, 800)

C_MADL  = "#8B1A4A"   # dark crimson — matches interim report
C_GMADL = "#2E75B6"   # steel blue
C_SIGN  = "#8B1A4A"   # same as MADL for the sign-switch line


def madl(y: float, yhat: np.ndarray) -> np.ndarray:
    return -np.tanh(TEMP * y * yhat) * abs(y)


def gmadl(y: float, yhat: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(A * y * yhat, -60, 60)))
    return -(sig - 0.5) * abs(y) ** B


def draw_panel(ax: plt.Axes, y: float, title: str, annot: str) -> None:
    ml = madl(y, YHAT)
    gl = gmadl(y, YHAT)

    # Sign-correct / sign-wrong shading
    if y > 0:
        ax.axvspan(-1.0, 0.0, color="#FDEBD0", alpha=0.45, zorder=0)
        ax.axvspan(0.0,  1.0, color="#D6EAF8", alpha=0.45, zorder=0)
    else:
        ax.axvspan(-1.0, 0.0, color="#D6EAF8", alpha=0.45, zorder=0)
        ax.axvspan(0.0,  1.0, color="#FDEBD0", alpha=0.45, zorder=0)

    # Saturation reference lines (dashed, same colour as curve)
    ax.axhline(ml.max(), color=C_MADL,  lw=1.0, ls="--", alpha=0.7)
    ax.axhline(ml.min(), color=C_MADL,  lw=1.0, ls="--", alpha=0.7)
    ax.axhline(gl.max(), color=C_GMADL, lw=1.0, ls="--", alpha=0.7)
    ax.axhline(gl.min(), color=C_GMADL, lw=1.0, ls="--", alpha=0.7)

    # Sign-switch vertical line
    ax.axvline(0.0, color=C_SIGN, lw=1.2, ls="--", alpha=0.85)
    ax.text(0.015, ax.get_ylim()[0] * 0.05 if y < 0 else ax.get_ylim()[1] * 0.92,
            "sign\nswitch", ha="left", va="top" if y > 0 else "bottom",
            fontsize=8, color=C_SIGN, style="italic")

    # Main curves
    ax.plot(YHAT, ml, color=C_MADL,  lw=2.0, marker="o", markevery=80,
            markersize=4, label="MADL", zorder=3)
    ax.plot(YHAT, gl, color=C_GMADL, lw=2.0, marker="o", markevery=80,
            markersize=4, label="GMADL", zorder=3)

    # Annotation box
    ax.text(0.97, 0.05, annot, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#AAAAAA", alpha=0.9))

    ax.set_xlabel(r"Predicted return $\hat{y}$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss value", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, pad=5)
    ax.set_xlim(-1.0, 1.0)
    ax.legend(fontsize=9.5, loc="upper center", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9)
    _style.y_grid_only(ax)


def main() -> None:
    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.84, wspace=0.26)

    draw_panel(ax1, y=+0.01,
               title=r"(a) Small positive return  ($y = +0.01$)",
               annot="GMADL reward\nsmoother near zero")
    draw_panel(ax2, y=-0.80,
               title=r"(b) Large negative return  ($y = -0.80$)",
               annot="GMADL penalty\n≈ 2× MADL at saturation")

    fig.suptitle(
        r"Figure 2.2 — MADL vs GMADL response  ($a$=100, $b$=2, temperature=25)"
        "\nIllustrative; closed-form components from §3.3; no training data",
        fontsize=10.5, fontweight="bold", y=0.99,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
