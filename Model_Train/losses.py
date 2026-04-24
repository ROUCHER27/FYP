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


def _validated_loss_kwargs(name: str, loss_kwargs: dict | None) -> dict:
    if not loss_kwargs:
        return {}

    allowed_by_loss = {
        "hybrid_add": {"lambda_dir", "lambda_hub"},
        "hybrid_mul": {"lambda_dir"},
    }
    allowed = allowed_by_loss.get(name, set())
    unknown = sorted(set(loss_kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported loss kwargs for {name}: {', '.join(unknown)}. "
            f"Supported loss kwargs: {', '.join(sorted(allowed)) or 'none'}"
        )
    return {key: float(value) for key, value in loss_kwargs.items()}


EXPERIMENT_LOSS_NAMES = (
    "mse",
    "medse",
    "gmadl",
    "imadl",
    "dirhuber",
    "hybrid_add",
    "hybrid_mul",
)


def get_experiment_loss_fn(
    name: str,
    loss_kwargs: dict | None = None,
) -> ExperimentLossFn:
    """
    Return the canonical training loss callable used by sanity-check experiments.
    为静态 sanity-check 实验返回统一命名的训练损失函数。
    """
    name_lower = name.lower()
    resolved_kwargs = _validated_loss_kwargs(name_lower, loss_kwargs)
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
            y_true, y_pred, reduction="mean", **resolved_kwargs
        )
    if name_lower == "hybrid_mul":
        return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
            y_true, y_pred, reduction="mean", **resolved_kwargs
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
    "get_experiment_loss_fn",
    "custom_loss",
]
