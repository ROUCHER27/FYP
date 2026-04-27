# 损失变体 3：鲁棒性增强的 M2

> **目标：** 为 M2 添加显式鲁棒性约束，以降低方差并改善跨种子稳定性，同时保持高收益。

**变体类型：** 正则化损失  
**基础损失：** M2（hybrid_mul_m2）  
**正则化：** 鲁棒性惩罚（月度收益方差）  
**参数：** γ（惩罚权重）

---

## 数学定义

### 公式

```
loss = hybrid_mul_m2 + γ * robustness_penalty

其中：
  hybrid_mul_m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
  robustness_penalty = Var(monthly_returns)
  γ ∈ {0.01, 0.1, 1.0}
```

### 组件分解

**M2 基础损失：**
```python
def hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0, delta=1.0):
    dir_term = _normalized_direction_term(y_true, y_pred)
    mag_term = huber_loss(y_true, y_pred, delta=delta)
    return (dir_term * mag_term * lambda_dir).mean()
```

**鲁棒性惩罚：**
```python
def robustness_penalty(y_true, y_pred, returns):
    """
    惩罚月度收益的高方差。
    
    参数：
        y_true: 真实收益
        y_pred: 预测收益
        returns: 月度投资组合收益（从预测计算）
    
    返回：
        月度收益的方差
    """
    monthly_returns = compute_monthly_returns(y_pred)
    return torch.var(monthly_returns)
```

**组合损失：**
```python
def m2_robust_loss(y_true, y_pred, gamma):
    base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    penalty = robustness_penalty(y_true, y_pred)
    return base_loss + gamma * penalty
```

---

## 原理

### 为什么添加鲁棒性惩罚？

1. **M2 的问题：**
   - 高平均 Sharpe（0.914）但极端方差（CV=1.396）
   - 33% 失败率（seed=52：Sharpe=-0.239）
   - Sharpe 跨种子范围从 -0.239 到 2.285

2. **根本原因：**
   - M2 优化预测准确性，而非交易鲁棒性
   - 对收益波动率没有显式约束
   - 乘法结构放大种子特定偏差

3. **解决方案：**
   - 添加方差惩罚以直接控制月度收益波动率
   - 鼓励在不同市场条件下的稳定性能
   - 平衡高收益与风险管理

### 不同 γ 的预期行为

| γ | 惩罚权重 | 预期行为 |
|---|---------|---------|
| 0.01 | 非常弱 | 影响最小，接近纯 M2 |
| 0.1 | 中等 | 显著降低方差 |
| 1.0 | 强 | 显著降低方差，可能牺牲收益 |

**假设：** γ=0.1 将提供最优平衡，降低 M2 的方差同时保持高收益。

---

## 参数网格

### 3 个配置

1. **m2_robust_001**（γ=0.01）：弱鲁棒性惩罚
2. **m2_robust_01**（γ=0.1）：中等鲁棒性惩罚
3. **m2_robust_10**（γ=1.0）：强鲁棒性惩罚

---

## 代码实现

### 添加到 `losses.py`

```python
# 在 EXPERIMENT_LOSS_NAMES 元组中
EXPERIMENT_LOSS_NAMES = (
    # ... 现有损失 ...
    "m2_robust_001",
    "m2_robust_01",
    "m2_robust_10",
)

# 鲁棒性惩罚的辅助函数
def compute_robustness_penalty(y_pred: torch.Tensor, 
                               batch_size: int = 32) -> torch.Tensor:
    """
    计算预测收益的方差作为鲁棒性惩罚。
    
    注意：这是简化版本。实际中，您可能想要：
    1. 按月分组预测
    2. 计算月度投资组合收益
    3. 计算月度收益的方差
    
    目前，我们使用预测的方差作为代理。
    """
    return torch.var(y_pred)

# 在 get_experiment_loss_fn() 函数中
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... 现有损失实现 ...
    
    # 变体 3：鲁棒性增强的 M2
    if name_lower == "m2_robust_001":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 0.01 * penalty
        return loss_fn
    
    if name_lower == "m2_robust_01":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 0.1 * penalty
        return loss_fn
    
    if name_lower == "m2_robust_10":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 1.0 * penalty
        return loss_fn
```

### 创建运行脚本

创建 3 个文件：`run_sanity_check_m2_robust_001.py`，`run_sanity_check_m2_robust_01.py`，`run_sanity_check_m2_robust_10.py`

**模板：**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (M2 Robust γ=0.01)")
    args = parser.parse_args()
    run_sanity_check("m2_robust_001", args)

if __name__ == "__main__":
    main()
```

### 更新 `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... 现有映射 ...
    "m2_robust_001": "run_sanity_check_m2_robust_001.py",
    "m2_robust_01": "run_sanity_check_m2_robust_01.py",
    "m2_robust_10": "run_sanity_check_m2_robust_10.py",
}
```

---

## 实验配置

### Phase 2.1：初步筛选

**运行次数：** 3 个损失 × 3 个种子 = 9 次运行

| 损失名称 | γ | 种子 | 权重上限 | 测试周期 |
|---------|---|------|---------|---------|
| m2_robust_001 | 0.01 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| m2_robust_01 | 0.1 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| m2_robust_10 | 1.0 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |

---

## 预期结果

### 性能预测

**低 γ（0.01）：弱惩罚**
- 预期 Sharpe：0.8-1.0
- 预期 CV：1.2-1.4
- 失败率：20-30%
- 行为：接近纯 M2，方差降低最小

**中等 γ（0.1）：中等惩罚**
- 预期 Sharpe：0.7-0.9
- 预期 CV：0.9-1.1
- 失败率：10-20%
- 行为：平衡，显著改善稳定性

**高 γ（1.0）：强惩罚**
- 预期 Sharpe：0.5-0.7
- 预期 CV：0.7-0.9
- 失败率：0-10%
- 行为：非常稳定，可能牺牲收益

### 成功标准

**最低要求：**
- 平均 Sharpe > 0.6
- CV < 1.2（优于纯 M2 的 1.396）
- 失败率 < 20%

**理想性能：**
- 平均 Sharpe > 0.8
- CV < 1.0
- 失败率 < 10%
- 无灾难性失败（Sharpe < -0.2）

---

## 分析计划

### 关键问题

1. **鲁棒性惩罚是否降低方差？**
   - 比较不同 γ 值的 CV
   - 检查是否防止 seed=52 失败

2. **收益-稳定性权衡是什么？**
   - 绘制不同 γ 的 Sharpe vs CV
   - 在帕累托前沿上识别最优 γ

3. **惩罚是否防止灾难性失败？**
   - 检查 seed=52 Sharpe 是否从 -0.239 改善
   - 验证其他种子上没有新失败

4. **惩罚机制是否有效？**
   - 分析训练期间的损失曲线
   - 检查惩罚项是否实际降低方差

### 可视化

1. **γ vs Sharpe 图：** 显示性能趋势
2. **γ vs CV 图：** 显示稳定性改进
3. **种子比较：** 每个 γ的箱线图
4. **损失分解：** 绘制训练期间的 base_loss vs penalty

---

## 实现说明

### 鲁棒性惩罚变体

当前实现使用 `torch.var(y_pred)` 作为简单代理。更复杂的版本可以包括：

**变体 A：月度收益方差**
```python
def compute_monthly_return_variance(y_pred, dates):
    # 按月分组预测
    monthly_returns = group_by_month(y_pred, dates)
    # 计算每月投资组合收益
    portfolio_returns = compute_portfolio_returns(monthly_returns)
    # 返回方差
    return torch.var(portfolio_returns)
```

**变体 B：最大回撤惩罚**
```python
def compute_drawdown_penalty(y_pred):
    cumulative_returns = torch.cumsum(y_pred, dim=0)
    running_max = torch.cummax(cumulative_returns, dim=0)[0]
    drawdown = running_max - cumulative_returns
    max_drawdown = torch.max(drawdown)
    return max_drawdown
```

**变体 C：下行偏差**
```python
def compute_downside_deviation(y_pred):
    negative_returns = torch.clamp(y_pred, max=0)
    return torch.std(negative_returns)
```

**建议：** 从简单的 `torch.var(y_pred)` 开始。如果结果有希望，在 Phase 2.2 中探索更复杂的惩罚。

---

## 论文贡献

### 方法创新

- **鲁棒性感知损失设计：** 首次为交易损失显式方差惩罚
- **直接风险控制：** 同时优化收益和稳定性
- **正则化方法：** 将 ML 正则化概念应用于金融损失函数

### 预期发现

1. **最优 γ：** 可能 γ=0.1 提供最佳权衡
2. **方差降低：** 相比纯 M2 降低 20-30% 的 CV
3. **失败预防：** 消除或减少 seed=52 灾难性失败
4. **收益牺牲：** 10-20% 的 Sharpe 降低对稳定性增益是可接受的

### 理论洞察

- 为什么交易损失需要显式鲁棒性约束
- 预测准确性和交易稳定性之间的权衡
- 正则化在防止对特定种子过拟合中的作用

---

**文档版本：** v1.0  
**创建日期：** 2026-04-26  
**作者：** Yirong Yu  
**状态：** 准备实施
