"""
Figure 2.1 — Conceptual loss-function shape comparison.

Paper-plot-skills style: `line_training_curve` (four-sided frame, tick_out,
serif family). No empirical data — evaluates closed-form loss expressions
from Chapter 3 §3.3 at a fixed realised return y = 0.05 over a dense ŷ grid.

Panels:
(a) MSE vs Huber(δ=0.01)  — quadratic-vs-quadratic-linear comparison
(b) MedSE illustrative shape (per-observation squared; median over batch)
(c) MADL (tanh-based) vs GMADL (sigmoid-based, |y|^b)
(d) hybrid_mul_m1 (λ_dir=2) — multiplicative hybrid vs Huber backbone

Inputs : none (synthetic)
Output : paper/figures/fig2_1_loss_shapes.png (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import (
    C_DYN,
    C_LINE_ALT,
    C_LINE_BASE,
    C_LINE_MAIN,
    C_NODYN,
    apply_paper_style,
    style_framed_axes,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig2_1_loss_shapes.png"

Y_FIXED = 0.05   # fixed realised return
DELTA_HUBER = 0.01
A_MADL = 25.0
A_GMADL = 100.0
B_GMADL = 2.0
LAMBDA_DIR_M1 = 2.0


def huber(e: np.ndarray, delta: float = DELTA_HUBER) -> np.ndarray:
    a = np.abs(e)
    return np.where(a <= delta, 0.5 * e ** 2, delta * (a - 0.5 * delta))


def mse(e: np.ndarray) -> np.ndarray:
    return e ** 2


def medse_curve(e: np.ndarray) -> np.ndarray:
    # Per-observation profile only — the batch-level median is a reduction.
    return e ** 2


def madl(y: float, yhat: np.ndarray) -> np.ndarray:
    return -np.tanh(A_MADL * y * yhat) * abs(y)


def gmadl(y: float, yhat: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-A_GMADL * y * yhat))
    return -(s - 0.5) * abs(y) ** B_GMADL


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mul_hybrid(y: float, yhat: np.ndarray, lambda_dir: float = LAMBDA_DIR_M1) -> np.ndarray:
    # Direction factor as in hybrid_mul_m1: D = 1 - sigmoid(a*y*yhat) * |y|^b / mean-equivalent.
    # For a pointwise sketch we drop the batch-mean factor (constant across ŷ).
    d = (1.0 - sigmoid(A_GMADL * y * yhat)) * abs(y) ** B_GMADL
    return (1.0 + lambda_dir * d) * huber(y - yhat)


def draw_panel(
    ax,
    title: str,
    yhat: np.ndarray,
    curves: list,
    highlight_zero_band: bool = False,
) -> None:
    for y_vals, label, color, lw, dashed in curves:
        ax.plot(
            yhat,
            y_vals,
            color=color,
            linewidth=lw,
            label=label,
            linestyle="--" if dashed else "-",
            zorder=3,
        )
    if highlight_zero_band:
        ax.axvspan(-0.005, 0.005, color="#FFE8B0", alpha=0.6, zorder=1,
                   label="weak-gradient band around $\\hat y=0$")

    ax.axvline(Y_FIXED, color="#999", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlabel(r"Prediction $\hat y$ (realised $y = 0.05$)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=10.5, fontweight="bold")
    ax.set_title(title, fontsize=11.5, pad=5)
    style_framed_axes(ax)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=9)
    ax.legend(fontsize=8.6, loc="best", frameon=True,
              framealpha=1.0, edgecolor="#C8C8C8", fancybox=False)


def main() -> None:
    apply_paper_style()
    yhat = np.linspace(-0.3, 0.3, 1200)
    y = Y_FIXED

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.4))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.92, hspace=0.35, wspace=0.25)

    # (a) MSE vs Huber
    e = y - yhat
    draw_panel(
        axs[0, 0],
        r"(a) MSE vs Huber ($\delta=0.01$)",
        yhat,
        [
            (mse(e),    "MSE",                C_LINE_ALT, 1.8, False),
            (huber(e),  r"Huber ($\delta=0.01$)", C_LINE_MAIN, 2.0, False),
        ],
    )

    # (b) MedSE shape (per-observation).
    draw_panel(
        axs[0, 1],
        "(b) MedSE per-observation shape",
        yhat,
        [(medse_curve(e), "squared residual (batch median reduction)", C_DYN, 2.0, False)],
    )
    # Annotation placed in the central clear region (curve is low around y=0.05).
    axs[0, 1].text(
        0.50,
        0.50,
        "Batch reduction is median(squared residual);\n"
        "the per-observation curve matches MSE.",
        transform=axs[0, 1].transAxes,
        fontsize=9,
        ha="center",
        va="center",
        color="#333",
        bbox=dict(facecolor="white", edgecolor="#C8C8C8", boxstyle="round,pad=0.4"),
    )

    # (c) MADL vs GMADL
    draw_panel(
        axs[1, 0],
        "(c) MADL vs GMADL (fixed $y=0.05$)",
        yhat,
        [
            (madl(y, yhat),  r"MADL: $-\tanh(25\,y\hat y)|y|$", C_LINE_MAIN, 1.8, False),
            (gmadl(y, yhat), r"GMADL: $-[\sigma(100\,y\hat y)-0.5]|y|^2$", C_LINE_ALT, 1.8, False),
        ],
        highlight_zero_band=True,
    )

    # (d) Multiplicative hybrid m1 vs Huber backbone
    draw_panel(
        axs[1, 1],
        r"(d) hybrid\_mul\_m1 ($\lambda_{dir}=2$) vs Huber backbone",
        yhat,
        [
            (huber(e),                       r"Huber backbone", C_LINE_BASE, 1.6, True),
            (mul_hybrid(y, yhat),            r"(1 + $\lambda_{dir}\cdot D(y,\hat y)$) $\cdot$ Huber", C_LINE_MAIN, 2.0, False),
        ],
    )

    fig.suptitle(
        "Figure 2.1 — Conceptual loss-function shapes at $y=0.05$ (illustrative; no training data)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
