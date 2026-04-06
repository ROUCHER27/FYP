from typing import Literal, Optional

import torch


Reduction = Optional[Literal["mean", "median", "sum", "none"]]


def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """
    Apply a standard reduction to an element-wise loss tensor.
    中文：对逐元素损失按照 mean/sum/none 指令做标准化聚合。
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
    中文：常规均方误差，用作 NN_Mean 的基准损失。
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
    中文：以平方误差的中位数为准的鲁棒损失，弱化异常值的影响。
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
    中文：用 tanh 平滑符号函数得到的可导 MADL 近似，方向一致奖励、方向相反惩罚。
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
    中文：基于 sigmoid(a·y·ŷ) 与 |y|^b 缩放的 GMADL 损失。
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
    中文：自定义损失的占位实现，可按实验需要自行扩展。
    """
    loss = mse_loss(y_true, y_pred, reduction="none")
    return _reduce(loss, reduction)


__all__ = [
    "Reduction",
    "mse_loss",
    "medse_loss",
    "madl_loss",
    "gmadl_loss",
    "custom_loss",
]
