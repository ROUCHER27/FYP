from dataclasses import dataclass
from typing import Callable, Literal, Optional

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
    direction_penalty = 1.0 - torch.sigmoid(product)
    weight = torch.abs(y_true) ** b
    return direction_penalty * (weight / (weight.mean() + eps))


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
    Improved MADL: normalized directional penalty plus squared magnitude error.
    IMADL：归一化方向惩罚项 + 平方幅度误差项。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    mag_term = (y_true - y_pred) ** 2
    loss = lambda_dir * dir_term + lambda_mag * mag_term
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
    Additive Phase 1.5 hybrid: lambda_dir * direction + lambda_hub * Huber(error).
    加法混合损失：方向项和 Huber 幅度项显式加权相加。
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
    Multiplicative Phase 1.5 hybrid: (1 + lambda_dir * direction) * Huber(error).
    乘法混合损失：方向错配放大 Huber 幅度误差。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = (1.0 + lambda_dir * dir_term) * huber_term
    return _reduce(loss, reduction)


@dataclass(frozen=True)
class LossSpec:
    family: str
    lambda_dir: float | None = None
    lambda_hub: float | None = None


# Stable names used in final-report output file names.
LOSS_SPECS = {
    "mse": LossSpec("mse"),
    "medse": LossSpec("medse"),
    "madl": LossSpec("madl"),
    "gmadl": LossSpec("gmadl"),
    "imadl": LossSpec("imadl"),
    "hybrid_add_a1": LossSpec("hybrid_add", lambda_dir=5.0, lambda_hub=1.0),
    "hybrid_add_a2": LossSpec("hybrid_add", lambda_dir=10.0, lambda_hub=1.0),
    "hybrid_add_a3": LossSpec("hybrid_add", lambda_dir=1.0, lambda_hub=0.1),
    "hybrid_add_a4": LossSpec("hybrid_add", lambda_dir=5.0, lambda_hub=0.1),
    "hybrid_add_a5": LossSpec("hybrid_add", lambda_dir=10.0, lambda_hub=0.1),
    "hybrid_mul_m1": LossSpec("hybrid_mul", lambda_dir=2.0),
    "hybrid_mul_m2": LossSpec("hybrid_mul", lambda_dir=5.0),
    "hybrid_mul_m3": LossSpec("hybrid_mul", lambda_dir=0.5),
    "hybrid_mul_m4": LossSpec("hybrid_mul", lambda_dir=0.1),
}

EXPERIMENT_LOSS_NAMES = tuple(LOSS_SPECS)


def get_experiment_loss_fn(name: str) -> ExperimentLossFn:
    """
    Return the canonical single-loss training callable used by sanity-check runs.
    根据稳定实验名返回单损失训练函数。
    """
    name_lower = name.lower()
    if name_lower not in LOSS_SPECS:
        supported = ", ".join(EXPERIMENT_LOSS_NAMES)
        raise ValueError(f"Unsupported experiment loss: {name}. Supported: {supported}")

    spec = LOSS_SPECS[name_lower]
    if spec.family == "mse":
        return lambda y_true, y_pred: mse_loss(y_true, y_pred, reduction="mean")
    if spec.family == "medse":
        return lambda y_true, y_pred: medse_loss(y_true, y_pred, reduction="median")
    if spec.family == "madl":
        return lambda y_true, y_pred: madl_loss(
            y_true, y_pred, temperature=25.0, reduction="mean"
        )
    if spec.family == "gmadl":
        return lambda y_true, y_pred: gmadl_loss(y_true, y_pred, reduction="mean")
    if spec.family == "imadl":
        return lambda y_true, y_pred: imadl_rebalanced_loss(
            y_true, y_pred, reduction="mean"
        )
    if spec.family == "hybrid_add":
        return lambda y_true, y_pred: hybrid_dir_huber_add_loss(
            y_true,
            y_pred,
            lambda_dir=float(spec.lambda_dir),
            lambda_hub=float(spec.lambda_hub),
            reduction="mean",
        )
    if spec.family == "hybrid_mul":
        return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
            y_true,
            y_pred,
            lambda_dir=float(spec.lambda_dir),
            reduction="mean",
        )
    raise ValueError(f"Unsupported experiment loss family: {spec.family}")


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


__all__ = [
    "ExperimentLossFn",
    "EXPERIMENT_LOSS_NAMES",
    "LOSS_SPECS",
    "LossSpec",
    "Reduction",
    "mse_loss",
    "medse_loss",
    "madl_loss",
    "gmadl_loss",
    "imadl_rebalanced_loss",
    "hybrid_dir_huber_add_loss",
    "hybrid_dir_huber_mul_loss",
    "get_experiment_loss_fn",
    "custom_loss",
]
