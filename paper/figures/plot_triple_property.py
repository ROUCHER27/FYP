"""
Figure 3.3 — Triple property of the multiplicative hybrid loss.

Three panels (1×3), line_training_curve style:
  (a) Directional asymmetry  — loss vs ŷ for fixed y = +5%, comparing MSE,
      Huber, and Multiplicative hybrid (M2, λ=5).
  (b) Magnitude awareness    — loss at a sign-wrong point (ŷ = -3%) vs |y|,
      showing how the hybrid penalty scales super-linearly with realised return.
  (c) Implicit variance penalty — expected batch loss vs prediction spread (std),
      showing the multiplicative gate amplifies loss when predictions are dispersed.

Illustrative; closed-form components from §3.3; no training data.
Output: paper/figures/fig3_3_triple_property.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from _style import apply_paper_style, style_framed_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig3_3_triple_property.png"

# ── loss components (closed-form, matches Model_Train/losses.py) ─────────────
A, B, DELTA, LAMBDA_DIR = 100.0, 2.0, 0.01, 5.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dir_gate(y: float, yhat: np.ndarray) -> np.ndarray:
    """Normalised directional gate D(y, ŷ) — scalar y, array ŷ."""
    penalty = 1.0 - sigmoid(A * y * yhat)
    weight = abs(y) ** B
    # batch normalisation: single-sample batch → weight / (weight + eps)
    return penalty * (weight / (weight + 1e-8))


def huber(err: np.ndarray) -> np.ndarray:
    ae = np.abs(err)
    return np.where(ae <= DELTA, 0.5 * err**2, DELTA * (ae - 0.5 * DELTA))


def mse_loss(y: float, yhat: np.ndarray) -> np.ndarray:
    return (y - yhat) ** 2


def hybrid_mul(y: float, yhat: np.ndarray) -> np.ndarray:
    return (1.0 + LAMBDA_DIR * dir_gate(y, yhat)) * huber(y - yhat)


# ── colours ──────────────────────────────────────────────────────────────────
C_MSE = "#A8C8E8"       # light steel blue
C_HUBER = "#5BBCCA"     # teal
C_HYBRID = "#C0392B"    # deep red (recommended)
C_SHADE_OK = "#D6EAF8"  # sign-correct shading
C_SHADE_BAD = "#FDEBD0" # sign-wrong shading


def panel_a(ax: plt.Axes) -> None:
    """Directional asymmetry: loss vs ŷ for y = +5%."""
    y = 0.05
    yhat = np.linspace(-0.20, 0.20, 400)

    ax.axvspan(-0.20, 0.0, color=C_SHADE_BAD, alpha=0.55, zorder=0)
    ax.axvspan(0.0, 0.20, color=C_SHADE_OK, alpha=0.55, zorder=0)
    ax.axvline(0, color="#888", lw=0.9, ls="--", zorder=1)

    ax.plot(yhat, mse_loss(y, yhat), color=C_MSE, lw=1.8, label="MSE", zorder=2)
    ax.plot(yhat, huber(y - yhat), color=C_HUBER, lw=1.8, ls="--", label="Huber", zorder=2)
    ax.plot(yhat, hybrid_mul(y, yhat), color=C_HYBRID, lw=2.2, label=r"Hybrid ($\lambda$=5)", zorder=3)

    ax.text(-0.10, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 0.003,
            "Sign-wrong", ha="center", va="top", fontsize=9, color="#8B4513")
    ax.text(0.10, 0.0, "Sign-correct", ha="center", va="bottom", fontsize=9, color="#1A5276")

    ax.set_xlabel(r"Prediction $\hat{y}$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss value", fontsize=11, fontweight="bold")
    ax.set_title(r"(a) Directional asymmetry  ($y = +5\%$)", fontsize=11, pad=5)
    ax.set_xlim(-0.20, 0.20)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=9, loc="upper center", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9.5)


def panel_b(ax: plt.Axes) -> None:
    """Magnitude awareness: loss at sign-wrong point (ŷ=-3%) vs |y|."""
    yhat_fixed = -0.03          # sign-wrong for all positive y
    y_vals = np.linspace(0.005, 0.15, 300)

    ax.plot(y_vals, mse_loss(y_vals, yhat_fixed),
            color=C_MSE, lw=1.8, label="MSE", zorder=2)
    ax.plot(y_vals, huber(y_vals - yhat_fixed),
            color=C_HUBER, lw=1.8, ls="--", label="Huber", zorder=2)
    ax.plot(y_vals, hybrid_mul(y_vals, yhat_fixed),
            color=C_HYBRID, lw=2.2, label=r"Hybrid ($\lambda$=5)", zorder=3)

    # Annotate the super-linear divergence region
    ax.annotate("Hybrid penalty\ngrows faster",
                xy=(0.12, hybrid_mul(0.12, yhat_fixed)),
                xytext=(0.07, hybrid_mul(0.12, yhat_fixed) * 1.15),
                fontsize=8.5, color=C_HYBRID,
                arrowprops=dict(arrowstyle="->", color=C_HYBRID, lw=1.0))

    ax.set_xlabel(r"Realised return magnitude $|y|$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss value", fontsize=11, fontweight="bold")
    ax.set_title(r"(b) Magnitude awareness  ($\hat{y} = -3\%$, sign-wrong)", fontsize=11, pad=5)
    ax.set_xlim(0, 0.155)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=9, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9.5)


def panel_c(ax: plt.Axes) -> None:
    """Implicit variance penalty: expected batch loss vs prediction std."""
    # Simulate a batch: y drawn from a realistic monthly return distribution
    rng = np.random.default_rng(42)
    y_batch = rng.normal(0.0, 0.04, size=2000)   # ~4% monthly std, realistic

    pred_stds = np.linspace(0.005, 0.12, 60)
    exp_mse, exp_huber, exp_hybrid = [], [], []

    for sigma in pred_stds:
        yhat_batch = rng.normal(0.0, sigma, size=len(y_batch))
        exp_mse.append(mse_loss(y_batch, yhat_batch).mean())
        exp_huber.append(huber(y_batch - yhat_batch).mean())
        exp_hybrid.append(hybrid_mul(y_batch, yhat_batch).mean())

    exp_mse = np.array(exp_mse)
    exp_huber = np.array(exp_huber)
    exp_hybrid = np.array(exp_hybrid)

    ax.plot(pred_stds, exp_mse, color=C_MSE, lw=1.8, label="MSE", zorder=2)
    ax.plot(pred_stds, exp_huber, color=C_HUBER, lw=1.8, ls="--", label="Huber", zorder=2)
    ax.plot(pred_stds, exp_hybrid, color=C_HYBRID, lw=2.2, label=r"Hybrid ($\lambda$=5)", zorder=3)

    ax.annotate("Super-linear\ngrowth",
                xy=(0.09, exp_hybrid[np.searchsorted(pred_stds, 0.09)]),
                xytext=(0.055, exp_hybrid[np.searchsorted(pred_stds, 0.09)] * 1.2),
                fontsize=8.5, color=C_HYBRID,
                arrowprops=dict(arrowstyle="->", color=C_HYBRID, lw=1.0))

    ax.set_xlabel(r"Prediction spread (std of $\hat{y}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected batch loss", fontsize=11, fontweight="bold")
    ax.set_title("(c) Implicit variance penalty", fontsize=11, pad=5)
    ax.set_xlim(0, 0.125)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=9, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#CCCCCC")
    style_framed_axes(ax)
    ax.tick_params(direction="out", labelsize=9.5)


def main() -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.15, top=0.84, wspace=0.30)

    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    fig.suptitle(
        "Figure 3.3 — Triple property of the multiplicative hybrid loss  "
        r"($a$=100, $b$=2, $\delta$=0.01, $\lambda$=5)  |  Illustrative; no training data",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
