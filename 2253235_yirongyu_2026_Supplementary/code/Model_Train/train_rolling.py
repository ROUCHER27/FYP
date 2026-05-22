from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import gmadl_loss, madl_loss, medse_loss, mse_loss
from .models import MLP, MLPConfig


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class RollingConfig:
    """
    Configuration for rolling-window training and prediction.
    滚动训练窗口的配置（训练年数、测试月数、步长）。
    """

    train_years: int = 5
    test_months: int = 6
    step_months: int = 6


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """
    Number of whole months between two timestamps.
    计算两个日期之间的整月差值。
    """
    return (a.year - b.year) * 12 + (a.month - b.month)


def generate_time_windows(
    dates: Sequence[pd.Timestamp],
    config: RollingConfig,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate rolling windows (train_start, train_end, test_start, test_end).
    生成训练起止与测试起止组成的滚动时间窗口列表。
    """
    unique_dates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    if not unique_dates:
        return windows
    idx = 0
    while True:
        train_start = unique_dates[idx]
        train_end_idx = idx
        while (
            train_end_idx < len(unique_dates)
            and month_diff(unique_dates[train_end_idx], train_start)
            < config.train_years * 12
        ):
            train_end_idx += 1
        if train_end_idx >= len(unique_dates):
            break
        train_end = unique_dates[train_end_idx - 1]
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx
        while (
            test_end_idx < len(unique_dates)
            and month_diff(unique_dates[test_end_idx], unique_dates[test_start_idx])
            < config.test_months
        ):
            test_end_idx += 1
        if test_end_idx > len(unique_dates):
            break
        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[test_end_idx - 1]
        windows.append((train_start, train_end, test_start, test_end))
        step_target_months = config.step_months
        while (
            idx < len(unique_dates)
            and month_diff(unique_dates[idx], train_start) < step_target_months
        ):
            idx += 1
        if idx >= len(unique_dates):
            break
    return windows


def build_window_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    dates: Sequence[pd.Timestamp],
    window: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for a given rolling window.
    基于指定窗口切分特征/目标，并封装成训练与测试 DataLoader。
    """
    train_start, train_end, test_start, test_end = window
    date_array = pd.to_datetime(pd.Series(dates)).to_numpy()
    mask_train = (date_array >= train_start) & (date_array <= train_end)
    mask_test = (date_array >= test_start) & (date_array <= test_end)
    x_train = torch.from_numpy(features[mask_train]).to(device=device)
    y_train = torch.from_numpy(targets[mask_train]).to(device=device)
    x_test = torch.from_numpy(features[mask_test]).to(device=device)
    y_test = torch.from_numpy(targets[mask_test]).to(device=device)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_loss_fn(name: str) -> LossFn:
    """
    Map a short name to a loss function.
    根据名称返回对应的损失函数。
    """
    name_lower = name.lower()
    if name_lower == "mse":
        return lambda y_true, y_pred: mse_loss(y_true, y_pred, reduction="mean")
    if name_lower == "medse":
        return lambda y_true, y_pred: medse_loss(y_true, y_pred, reduction="median")
    if name_lower == "madl":
        return lambda y_true, y_pred: madl_loss(
            y_true, y_pred, temperature=25.0, reduction="mean"
        )
    if name_lower == "gmadl":
        return lambda y_true, y_pred: gmadl_loss(y_true, y_pred, reduction="mean")
    raise ValueError(f"Unsupported loss name: {name}")


def train_one_window(
    config: MLPConfig,
    train_loader: DataLoader,
    loss_fn: LossFn,
    device: torch.device,
    max_epochs: int = 50,
) -> MLP:
    """
    Train a single MLP on one rolling window.
    在指定窗口的数据上训练一份 MLP 模型。
    """
    model = MLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for _ in range(max_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(batch_y, preds)
            loss.backward()
            optimizer.step()
    return model


def predict_window(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Generate predictions for one window and return them as a numpy array.
    对测试窗口生成预测，并返回拼接后的 numpy 数组。
    """
    model.eval()
    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            preds_list.append(preds.cpu().numpy())
    if not preds_list:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(preds_list, axis=0)


__all__ = [
    "RollingConfig",
    "month_diff",
    "generate_time_windows",
    "build_window_dataloaders",
    "get_loss_fn",
    "train_one_window",
    "predict_window",
]
