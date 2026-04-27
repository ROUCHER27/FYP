from typing import Literal, Optional

import torch


Reduction = Optional[Literal["mean", "median", "sum", "none"]]


def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """
    Apply a standard reduction to an element-wise loss tensor.
    对逐元素损失按照 mean/sum/none 指令做标准化聚合。
    """
    if reduction == "mean":
        return loss.mean()
    if reduction == "median":
        return loss.median()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none" or reduction is None:
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def mse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Standard mean squared error loss used for NN_Mean.
    常规均方误差，用作 NN_Mean 的基准损失。
    """
    loss = (y_true - y_pred) ** 2
    return _reduce(loss, reduction)


def medse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "median",
) -> torch.Tensor:
    """
    Median squared error for robustness against outliers.
    以平方误差的中位数为准的鲁棒损失，弱化异常值的影响。
    """
    loss = (y_true - y_pred) ** 2
    return _reduce(loss, reduction)


def madl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Differentiable approximation of MADL using a smoothed sign via tanh.
    用 tanh 平滑符号函数得到的可导 MADL 近似，方向一致奖励、方向相反惩罚。
    """
    prod = y_true * y_pred
    alignment = torch.tanh(temperature * prod)
    loss = -alignment * torch.abs(y_true)
    return _reduce(loss, reduction)


def gmadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    GMADL loss based on a scaled sigmoid of a * y_true * y_pred and |y_true|^b.
    基于 sigmoid(a·y·ŷ) 与 |y|^b 缩放的 GMADL 损失。
    """
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    loss = -(sigmoid - 0.5) * torch.abs(y_true) ** b
    return _reduce(loss, reduction)


def custom_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
    **kwargs,
) -> torch.Tensor:
    """
    Placeholder for a user-defined loss function to be customized for experiments.
    自定义损失的占位实现，可按实验需要自行扩展。
    """
    loss = mse_loss(y_true, y_pred, reduction="none")
    return _reduce(loss, reduction)


# ============================================================================
# Phase 2: Hybrid Loss Functions
# ============================================================================

def m2_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    M2 loss: Signed squared error that rewards correct directional predictions.
    M2 损失：带符号的平方误差，奖励方向正确的预测。

    Formula: L = -sign(y_true) * (y_pred - y_true)^2
    """
    loss = -torch.sign(y_true) * (y_pred - y_true) ** 2
    return _reduce(loss, reduction)


def imadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    IMADL (Improved MADL) loss - alias for madl_loss for clarity.
    IMADL 损失 - madl_loss 的别名，用于 Phase 2 实验中的清晰表达。
    """
    return madl_loss(y_true, y_pred, temperature, reduction)


# ----------------------------------------------------------------------------
# Variant 1: IMADL + M2 Linear Combination (7 functions)
# ----------------------------------------------------------------------------

def imadl_m2_linear_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Linear combination of IMADL and M2 losses.
    IMADL 和 M2 的线性组合。

    Formula: L = α * IMADL + (1-α) * M2

    Args:
        alpha: Weight for IMADL (0 = pure M2, 1 = pure IMADL)
    """
    imadl = imadl_loss(y_true, y_pred, temperature, reduction="none")
    m2 = m2_loss(y_true, y_pred, reduction="none")
    combined = alpha * imadl + (1 - alpha) * m2
    return _reduce(combined, reduction)


def imadl_m2_alpha02_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.2 (20% IMADL, 80% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.2, reduction=reduction)


def imadl_m2_alpha03_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.3 (30% IMADL, 70% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.3, reduction=reduction)


def imadl_m2_alpha04_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.4 (40% IMADL, 60% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.4, reduction=reduction)


def imadl_m2_alpha05_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.5 (50% IMADL, 50% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.5, reduction=reduction)


def imadl_m2_alpha06_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.6 (60% IMADL, 40% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.6, reduction=reduction)


def imadl_m2_alpha07_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.7 (70% IMADL, 30% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.7, reduction=reduction)


def imadl_m2_alpha08_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.8 (80% IMADL, 20% M2)"""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.8, reduction=reduction)


# ----------------------------------------------------------------------------
# Variant 2: IMADL + GMADL Weighted Combination (3 functions)
# ----------------------------------------------------------------------------

def imadl_gmadl_weighted_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 25.0,
    a: float = 100.0,
    b: float = 2.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Weighted combination of IMADL and GMADL losses.
    IMADL 和 GMADL 的加权组合。

    Formula: L = β * IMADL + (1-β) * GMADL

    Args:
        beta: Weight for IMADL (0 = pure GMADL, 1 = pure IMADL)
    """
    imadl = imadl_loss(y_true, y_pred, temperature, reduction="none")
    gmadl = gmadl_loss(y_true, y_pred, a, b, reduction="none")
    combined = beta * imadl + (1 - beta) * gmadl
    return _reduce(combined, reduction)


def imadl_gmadl_beta03_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + GMADL with beta=0.3 (30% IMADL, 70% GMADL)"""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.3, reduction=reduction)


def imadl_gmadl_beta05_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + GMADL with beta=0.5 (50% IMADL, 50% GMADL)"""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.5, reduction=reduction)


def imadl_gmadl_beta07_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + GMADL with beta=0.7 (70% IMADL, 30% GMADL)"""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.7, reduction=reduction)


# ----------------------------------------------------------------------------
# Variant 3: M2 Robustness Enhanced (3 functions)
# ----------------------------------------------------------------------------

def m2_robustness_enhanced_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    gamma: float = 0.1,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    M2 loss with robustness penalty to reduce prediction variance.
    M2 损失 + 鲁棒性惩罚项，降低预测方差。

    Formula: L = M2 - γ * std(y_pred)^2

    Args:
        gamma: Weight for robustness penalty
    """
    m2 = m2_loss(y_true, y_pred, reduction="none")

    # Robustness penalty: penalize high prediction variance
    pred_std = y_pred.std()
    robustness_penalty = pred_std ** 2

    # Combine (subtract penalty to encourage stability)
    combined = m2 - gamma * robustness_penalty
    return _reduce(combined, reduction)


def m2_robust_gamma001_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + Robustness with gamma=0.01 (minimal penalty)"""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.01, reduction=reduction)


def m2_robust_gamma01_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + Robustness with gamma=0.1 (moderate penalty)"""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.1, reduction=reduction)


def m2_robust_gamma10_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + Robustness with gamma=1.0 (strong penalty)"""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=1.0, reduction=reduction)


# ----------------------------------------------------------------------------
# Variant 4: Adaptive Hybrid (3 functions)
# ----------------------------------------------------------------------------

def adaptive_hybrid_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    lambda_: float = 5.0,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Adaptive hybrid loss that dynamically blends IMADL and M2 based on |y_true|.
    根据 |y_true| 自适应混合 IMADL 和 M2 的损失函数。

    Formula: L = IMADL * exp(-λ|y|) + M2 * (1 - exp(-λ|y|))

    Mechanism:
    - Small |y_true| (near 0): Use IMADL (stable)
    - Large |y_true| (far from 0): Use M2 (aggressive)

    Args:
        lambda_: Controls transition speed (higher = faster transition to M2)
    """
    abs_y = torch.abs(y_true)
    weight_imadl = torch.exp(-lambda_ * abs_y)
    weight_m2 = 1 - weight_imadl

    imadl = imadl_loss(y_true, y_pred, temperature, reduction="none")
    m2 = m2_loss(y_true, y_pred, reduction="none")

    combined = weight_imadl * imadl + weight_m2 * m2
    return _reduce(combined, reduction)


def adaptive_lambda10_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Adaptive hybrid with lambda=1.0 (slow transition, IMADL-dominant)"""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=1.0, reduction=reduction)


def adaptive_lambda50_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Adaptive hybrid with lambda=5.0 (balanced transition)"""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=5.0, reduction=reduction)


def adaptive_lambda100_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Adaptive hybrid with lambda=10.0 (fast transition, M2-dominant)"""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=10.0, reduction=reduction)


__all__ = [
    "Reduction",
    "mse_loss",
    "medse_loss",
    "madl_loss",
    "gmadl_loss",
    "custom_loss",
    # Phase 2 base functions
    "m2_loss",
    "imadl_loss",
    # Variant 1: IMADL + M2 Linear (7)
    "imadl_m2_linear_loss",
    "imadl_m2_alpha02_loss",
    "imadl_m2_alpha03_loss",
    "imadl_m2_alpha04_loss",
    "imadl_m2_alpha05_loss",
    "imadl_m2_alpha06_loss",
    "imadl_m2_alpha07_loss",
    "imadl_m2_alpha08_loss",
    # Variant 2: IMADL + GMADL Weighted (3)
    "imadl_gmadl_weighted_loss",
    "imadl_gmadl_beta03_loss",
    "imadl_gmadl_beta05_loss",
    "imadl_gmadl_beta07_loss",
    # Variant 3: M2 Robustness Enhanced (3)
    "m2_robustness_enhanced_loss",
    "m2_robust_gamma001_loss",
    "m2_robust_gamma01_loss",
    "m2_robust_gamma10_loss",
    # Variant 4: Adaptive Hybrid (3)
    "adaptive_hybrid_loss",
    "adaptive_lambda10_loss",
    "adaptive_lambda50_loss",
    "adaptive_lambda100_loss",
]
