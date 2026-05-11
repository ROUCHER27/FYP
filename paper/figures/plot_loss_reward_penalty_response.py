"""
Figure 3.1 — Reward and penalty logic of the hybrid loss components.

2×2 panels, one per realised-return value:
  (a) y = -10%   (b) y = -2%
  (c) y = +2%    (d) y = +10%

Each panel shows, as ŷ sweeps [-20%, +20%]:
  - Directional gate  D(y, ŷ)
  - Multiplicative gate  1 + λ·D(y, ŷ)   (λ=5, matching hybrid_mul_m2)
  - GMADL signed score  -(σ(a·y·ŷ) - 0.5)·|y|^b  (dashed reference)

Shading: sign-correct region (blue), sign-wrong region (orange).
Vertical line at ŷ=0 labelled "sign boundary".

Illustrative; closed-form components from §3.3; no training data.
Output: paper/figures/fig3_1_loss_reward_penalty_response.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import apply_paper_style, style_framed_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig3_1_loss_reward_penalty_response.png"

A, B, LAMBDA_DIR = 100.0, 2.0, 5.0
YHAT = np.linspace(-0.20, 0.20, 500)

C_DIR = "#1B3D6E"       # deep navy — directional gate
C_MUL = "#C0392B"       # deep red  — multiplicative gate (recommended)
C_GMADL = "#7D3C98"     # purple    — GMADL reference (dashed)
C_OK = "#D6EAF8"        # sign-correct shading
C_BAD = "#FDEBD0"       # sign-wrong shading


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def dir_gate(y: float, yhat: np.ndarray) -> np.ndarray:
    penalty = 1.0 - sigmoid(A * y * yhat)
    weight = abs(y) ** B
    return penalty * (weight / (weight + 1e-8))


def mul_gate(y: float, yhat: np.ndarray) -> np.ndarray:
    return 1.0 + LAMBDA_DIR * dir_gate(y, yhat)


def gmadl_score(y: float, yhat: np.ndarray) -> np.ndarray:
    """Signed GMADL score (negative = reward, positive = penalty)."""
    return -(sigmoid(A * y * yhat) - 0.5) * abs(y) ** B


def draw_panel(ax: plt.Axes, y: float, label: str) -> None:
    sign_boundary = 0.0

    # Shading: sign-correct where y*ŷ > 0
    if y > 0:
        ax.axvspan(sign_boundary, 0.20, color=C_OK, alpha=0.55, zorder=0)
        ax.axvspan(-0.20, sign_boundary, color=C_BAD, alpha=0.55, zorder=0)
        ok_x, bad_x = 0.10, -0.10
    else:
        ax.axvspan(-0.20, sign_boundary, color=C_OK, alpha=0.55, zorder=0)
        ax.axvspan(sign_boundary, 0.20, color=C_BAD, alpha=0.55, zorder=0)
        ok_x, bad_x = -0.10, 0.10

    ax.axvline(sign_boundary, color="#666", lw=1.0, ls="--", zorder=1)

    dg = dir_gate(y, YHAT)
    mg = mul_gate(y, YHAT)
    gs = gmadl_score(y, YHAT)

    ax.plot(YHAT, dg, color=C_DIR, lw=1.8, label="Directional gate $D$", zorder=3)
    ax.plot(YHAT, mg, color=C_MUL, lw=2.2, label=r"Multiplicative gate $1+\lambda D$", zorder=4)
    ax.plot(YHAT, gs, color=C_GMADL, lw=1.4, ls="--",
            label="GMADL signed score", zorder=2)

    # Region labels
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.5
    ax.text(ok_x, 0.05, "Sign-correct", ha="center", va="bottom",
            fontsize=8, color="#1A5276", style="italic")
    ax.text(bad_x, 0.05, "Sign-wrong", ha="center", va="bottom",
            fontsize=8, color="#784212", style="italic")
    ax.text(sign_boundary + 0.005, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1.3,
            "sign\nboundary", ha="left", va="top", fontsize=7.5, color="#555")

    ax.set_title(label, fontsize=11, pad=4)
    ax.set_xlabel(r"Prediction $\hat{y}$", fontsize=10)
    ax.set_ylabel("Relative response", fontsize=10)
    ax.set_xlim(-0.20, 0.20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=8.5)


def main() -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.88,
                        hspace=0.42, wspace=0.28)

    cases = [
        (-0.10, r"(a) Realised return $y = -10\%$"),
        (-0.02, r"(b) Realised return $y = -2\%$"),
        (+0.02, r"(c) Realised return $y = +2\%$"),
        (+0.10, r"(d) Realised return $y = +10\%$"),
    ]
    for ax, (y, lbl) in zip(axes.flat, cases):
        draw_panel(ax, y, lbl)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=9.5, frameon=True, facecolor="white",
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.97))

    fig.suptitle(
        r"Figure 3.1 — Reward and penalty logic of the hybrid loss components"
        "\n"
        r"($a$=100, $b$=2, $\lambda$=5)  |  Illustrative; no training data",
        fontsize=10.5, fontweight="bold", y=1.00,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
