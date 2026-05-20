from dataclasses import asdict
from itertools import product
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import mse_loss
from .models import MLP, MLPConfig


def build_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap features and targets into a simple TensorDataset and DataLoader.
    将特征与标签封装进 TensorDataset，再生成 DataLoader。
    """
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single training epoch using MSE loss.
    执行一次使用 MSE 损失的训练轮。
    """
    model.train()
    running_loss = 0.0
    count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = mse_loss(batch_y, preds)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)
    epoch_loss = running_loss / max(count, 1)
    return epoch_loss


def evaluate_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate mean squared error on a validation loader.
    在验证集 DataLoader 上评估均方误差。
    """
    model.eval()
    running_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = mse_loss(batch_y, preds)
            running_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    mse_value = running_loss / max(count, 1)
    return mse_value


def grid_search_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    param_grid: Dict[str, Iterable],
    input_dim: int,
    device: torch.device,
    max_epochs: int = 50,
) -> Tuple[MLPConfig, float]:
    """
    Perform a simple grid search over MLP hyperparameters using MSE on a validation set.
    通过验证集 MSE 对 MLP 超参数做简单网格搜索。
    """
    best_config = None
    best_score = float("inf")
    train_loader = None
    val_loader = None
    for hidden_dims, activation, dropout, lr, batch_size in product(
        param_grid.get("hidden_dims", [[64, 64]]), # hidden_dims: List[int] 隐藏层维度，指每个隐藏层的神经元数量
        param_grid.get("activation", ["relu"]), # activation: str 激活函数，指每个神经元在接收到输入后，通过激活函数进行非线性变换，增加模型的表达能力
        param_grid.get("dropout", [0.0]), # dropout: float 丢弃率/防止过拟合，指在训练中随机“关掉”多少比例的神经元
        param_grid.get("lr", [1e-3]), # lr: float 学习率/步长，指每次更新参数时，根据损失函数的变化调整的幅度
        param_grid.get("batch_size", [512]), # batch_size: int 批量大小，指每次训练时，使用的样本数量
    ):
        config = MLPConfig(
            input_dim=input_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            dropout=float(dropout),
        )
        model = MLP(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        train_loader = build_dataloader(
            x_train, y_train, batch_size=int(batch_size), shuffle=True
        )
        val_loader = build_dataloader(
            x_val, y_val, batch_size=int(batch_size), shuffle=False
        )
        for _ in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, device)
        val_mse = evaluate_mse(model, val_loader, device)
        if val_mse < best_score:
            best_score = val_mse
            best_config = config
    if best_config is None:
        raise RuntimeError("Grid search did not evaluate any configuration.")
    return best_config, best_score


def config_to_dict(config: MLPConfig) -> Dict:
    """
    Convert an MLPConfig dataclass to a plain dictionary for saving or logging.
    将 MLPConfig 数据类展开成可记录的字典。
    """
    return asdict(config)


__all__ = [
    "build_dataloader",
    "train_one_epoch",
    "evaluate_mse",
    "grid_search_mlp",
    "config_to_dict",
]
