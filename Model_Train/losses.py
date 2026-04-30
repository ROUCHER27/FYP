from typing import Callable, Dict, Literal, Optional

import torch


Reduction = Optional[Literal["mean", "median", "sum", "none"]]
ExperimentLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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


def _normalized_direction_term(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    product = a * y_true * y_pred
    dir_penalty = 1.0 - torch.sigmoid(product)
    weight = torch.abs(y_true) ** b
    mean_weight = weight.mean() + eps
    normalized_weight = weight / mean_weight
    return dir_penalty * normalized_weight


def _huber_term(error: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    abs_error = torch.abs(error)
    return torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )


def imadl_rebalanced_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    lambda_dir: float = 1.0,
    lambda_mag: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Rebalanced Improved MADL that combines directional pressure with magnitude error.
    通过归一化方向项与幅度误差的加法组合，避免小收益样本的方向信号被淹没。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    mag_term = (y_true - y_pred) ** 2
    loss = lambda_dir * dir_term + lambda_mag * mag_term
    return _reduce(loss, reduction)


def directional_huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 10.0,
    delta: float = 0.01,
    lambda_dir: float = 1.0,
    lambda_hub: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Additive directional-Huber loss with a smooth directional penalty.
    用 tanh 方向惩罚 + Huber 幅度项结合方向性与鲁棒性。
    """
    product = a * y_true * y_pred
    dir_penalty = 0.5 * (1.0 - torch.tanh(product))
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = lambda_dir * dir_penalty + lambda_hub * huber_term
    return _reduce(loss, reduction)


def hybrid_dir_huber_add_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    delta: float = 0.01,
    lambda_dir: float = 1.0,
    lambda_hub: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Additive hybrid directional-Huber loss.
    归一化方向项与 Huber 项线性相加，是首轮主贡献候选形式。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = lambda_dir * dir_term + lambda_hub * huber_term
    return _reduce(loss, reduction)


def hybrid_dir_huber_mul_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    delta: float = 0.01,
    lambda_dir: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Multiplicative hybrid directional-Huber loss.
    用方向项放大 Huber 误差，突出“错方向的误差经济代价更高”。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = (1.0 + lambda_dir * dir_term) * huber_term
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
    Phase 2 M2 baseline, aligned with the Phase 1.5 hybrid_mul M2.

    This intentionally delegates to hybrid_dir_huber_mul_loss with
    lambda_dir=2.0. Do not replace it with the older signed squared-error
    approximation; Phase 2 Variant 1 and Variant 4 depend on this baseline.
    """
    return hybrid_dir_huber_mul_loss(
        y_true, y_pred, lambda_dir=2.0, reduction=reduction
    )


def imadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    IMADL loss alias for Phase 2 experiment naming.
    IMADL 损失别名，用于 Phase 2 实验中的清晰表达。
    """
    return madl_loss(y_true, y_pred, temperature, reduction)


def imadl_m2_linear_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Linear combination of IMADL and the corrected Phase 1.5-style M2 baseline.

    Formula: L = alpha * IMADL + (1 - alpha) * M2
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
    """IMADL + M2 with alpha=0.2 (20% IMADL, 80% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.2, reduction=reduction)


def imadl_m2_alpha03_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.3 (30% IMADL, 70% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.3, reduction=reduction)


def imadl_m2_alpha04_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.4 (40% IMADL, 60% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.4, reduction=reduction)


def imadl_m2_alpha05_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.5 (50% IMADL, 50% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.5, reduction=reduction)


def imadl_m2_alpha06_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.6 (60% IMADL, 40% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.6, reduction=reduction)


def imadl_m2_alpha07_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.7 (70% IMADL, 30% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.7, reduction=reduction)


def imadl_m2_alpha08_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + M2 with alpha=0.8 (80% IMADL, 20% M2)."""
    return imadl_m2_linear_loss(y_true, y_pred, alpha=0.8, reduction=reduction)


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

    Formula: L = beta * IMADL + (1 - beta) * GMADL
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
    """IMADL + GMADL with beta=0.3 (30% IMADL, 70% GMADL)."""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.3, reduction=reduction)


def imadl_gmadl_beta05_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + GMADL with beta=0.5 (50% IMADL, 50% GMADL)."""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.5, reduction=reduction)


def imadl_gmadl_beta07_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """IMADL + GMADL with beta=0.7 (70% IMADL, 30% GMADL)."""
    return imadl_gmadl_weighted_loss(y_true, y_pred, beta=0.7, reduction=reduction)


def m2_robustness_enhanced_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    gamma: float = 0.1,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Corrected M2 loss with a positive prediction-variance penalty.

    Formula: L = M2 + gamma * var(y_pred)
    """
    m2 = m2_loss(y_true, y_pred, reduction="none")
    variance_penalty = y_pred.var()
    combined = m2 + gamma * variance_penalty
    return _reduce(combined, reduction)


def m2_robust_gamma001_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=0.01 (minimal penalty)."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.01, reduction=reduction)


def m2_robust_gamma01_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=0.1 (moderate penalty)."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.1, reduction=reduction)


def m2_robust_gamma03_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=0.3."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.3, reduction=reduction)


def m2_robust_gamma05_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=0.5."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.5, reduction=reduction)


def m2_robust_gamma07_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=0.7."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=0.7, reduction=reduction)


def m2_robust_gamma10_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=1.0 (strong penalty)."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=1.0, reduction=reduction)


def m2_robust_gamma15_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """M2 + robustness with gamma=1.5."""
    return m2_robustness_enhanced_loss(y_true, y_pred, gamma=1.5, reduction=reduction)


def adaptive_hybrid_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    lambda_: float = 5.0,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Adaptive blend of IMADL and the corrected Phase 1.5-style M2 baseline.

    Formula: L = IMADL * exp(-lambda * |y|) + M2 * (1 - exp(-lambda * |y|))
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
    """Adaptive hybrid with lambda=1.0 (slow transition, IMADL-dominant)."""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=1.0, reduction=reduction)


def adaptive_lambda50_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Adaptive hybrid with lambda=5.0 (balanced transition)."""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=5.0, reduction=reduction)


def adaptive_lambda100_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Adaptive hybrid with lambda=10.0 (fast transition, M2-dominant)."""
    return adaptive_hybrid_loss(y_true, y_pred, lambda_=10.0, reduction=reduction)


EXPERIMENT_LOSS_NAMES = (
    "mse",
    "medse",
    "gmadl",
    "imadl",
    "dirhuber",
    "hybrid_add",
    "hybrid_mul",
    "m2",
    "imadl_m2_alpha02",
    "imadl_m2_alpha03",
    "imadl_m2_alpha04",
    "imadl_m2_alpha05",
    "imadl_m2_alpha06",
    "imadl_m2_alpha07",
    "imadl_m2_alpha08",
    "imadl_gmadl_beta03",
    "imadl_gmadl_beta05",
    "imadl_gmadl_beta07",
    "m2_robust_gamma001",
    "m2_robust_gamma01",
    "m2_robust_gamma03",
    "m2_robust_gamma05",
    "m2_robust_gamma07",
    "m2_robust_gamma10",
    "m2_robust_gamma15",
    "adaptive_lambda10",
    "adaptive_lambda50",
    "adaptive_lambda100",
)


_PHASE2_LOSS_FNS: Dict[str, Callable[..., torch.Tensor]] = {
    "m2": m2_loss,
    "imadl_m2_alpha02": imadl_m2_alpha02_loss,
    "imadl_m2_alpha03": imadl_m2_alpha03_loss,
    "imadl_m2_alpha04": imadl_m2_alpha04_loss,
    "imadl_m2_alpha05": imadl_m2_alpha05_loss,
    "imadl_m2_alpha06": imadl_m2_alpha06_loss,
    "imadl_m2_alpha07": imadl_m2_alpha07_loss,
    "imadl_m2_alpha08": imadl_m2_alpha08_loss,
    "imadl_gmadl_beta03": imadl_gmadl_beta03_loss,
    "imadl_gmadl_beta05": imadl_gmadl_beta05_loss,
    "imadl_gmadl_beta07": imadl_gmadl_beta07_loss,
    "m2_robust_gamma001": m2_robust_gamma001_loss,
    "m2_robust_gamma01": m2_robust_gamma01_loss,
    "m2_robust_gamma03": m2_robust_gamma03_loss,
    "m2_robust_gamma05": m2_robust_gamma05_loss,
    "m2_robust_gamma07": m2_robust_gamma07_loss,
    "m2_robust_gamma10": m2_robust_gamma10_loss,
    "m2_robust_gamma15": m2_robust_gamma15_loss,
    "adaptive_lambda10": adaptive_lambda10_loss,
    "adaptive_lambda50": adaptive_lambda50_loss,
    "adaptive_lambda100": adaptive_lambda100_loss,
}


def get_experiment_loss_fn(name: str) -> ExperimentLossFn:
    """
    Return the canonical training loss callable used by sanity-check experiments.
    为静态 sanity-check 实验返回统一命名的训练损失函数。
    """
    name_lower = name.lower()
    if name_lower == "mse":
        return lambda y_true, y_pred: mse_loss(y_true, y_pred, reduction="mean")
    if name_lower == "medse":
        return lambda y_true, y_pred: medse_loss(y_true, y_pred, reduction="median")
    if name_lower == "gmadl":
        return lambda y_true, y_pred: gmadl_loss(y_true, y_pred, reduction="mean")
    if name_lower == "imadl":
        return lambda y_true, y_pred: imadl_rebalanced_loss(
            y_true, y_pred, reduction="mean"
        )
    if name_lower == "dirhuber":
        return lambda y_true, y_pred: directional_huber_loss(
            y_true, y_pred, reduction="mean"
        )
    if name_lower == "hybrid_add":
        return lambda y_true, y_pred: hybrid_dir_huber_add_loss(
            y_true, y_pred, reduction="mean"
        )
    if name_lower == "hybrid_mul":
        return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
            y_true, y_pred, reduction="mean"
        )
    if name_lower in _PHASE2_LOSS_FNS:
        phase2_loss_fn = _PHASE2_LOSS_FNS[name_lower]
        return lambda y_true, y_pred: phase2_loss_fn(
            y_true, y_pred, reduction="mean"
        )
    raise ValueError(f"Unsupported experiment loss: {name}")


__all__ = [
    "ExperimentLossFn",
    "EXPERIMENT_LOSS_NAMES",
    "Reduction",
    "mse_loss",
    "medse_loss",
    "madl_loss",
    "gmadl_loss",
    "imadl_rebalanced_loss",
    "directional_huber_loss",
    "hybrid_dir_huber_add_loss",
    "hybrid_dir_huber_mul_loss",
    "m2_loss",
    "imadl_loss",
    "imadl_m2_linear_loss",
    "imadl_m2_alpha02_loss",
    "imadl_m2_alpha03_loss",
    "imadl_m2_alpha04_loss",
    "imadl_m2_alpha05_loss",
    "imadl_m2_alpha06_loss",
    "imadl_m2_alpha07_loss",
    "imadl_m2_alpha08_loss",
    "imadl_gmadl_weighted_loss",
    "imadl_gmadl_beta03_loss",
    "imadl_gmadl_beta05_loss",
    "imadl_gmadl_beta07_loss",
    "m2_robustness_enhanced_loss",
    "m2_robust_gamma001_loss",
    "m2_robust_gamma01_loss",
    "m2_robust_gamma03_loss",
    "m2_robust_gamma05_loss",
    "m2_robust_gamma07_loss",
    "m2_robust_gamma10_loss",
    "m2_robust_gamma15_loss",
    "adaptive_hybrid_loss",
    "adaptive_lambda10_loss",
    "adaptive_lambda50_loss",
    "adaptive_lambda100_loss",
    "get_experiment_loss_fn",
    "custom_loss",
]
