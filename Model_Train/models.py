from dataclasses import dataclass
from typing import List

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """
    Map a string name to a PyTorch activation module.
    """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "tanh":
        return nn.Tanh()
    if name_lower == "elu":
        return nn.ELU()
    if name_lower == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class MLPConfig:
    """
    Configuration for a fully connected feed-forward network.
    定义前馈全连接网络的输入维度、隐藏层等超参数。
    """

    input_dim: int
    hidden_dims: List[int]
    activation: str = "relu"
    dropout: float = 0.0


class MLP(nn.Module):
    """
    Simple multi-layer perceptron for cross-sectional return prediction.
    用于截面收益预测的多层感知机主体。
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = config.input_dim
        activation = get_activation(config.activation)
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns predicted one-period-ahead returns.
        前向计算，输出下一期预期收益。
        """
        out = self.network(x)
        out = out.squeeze(-1)
        return out


__all__ = ["MLPConfig", "MLP", "get_activation"]
