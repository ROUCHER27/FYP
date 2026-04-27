# 损失变体 4：自适应混合

> **目标：** 基于样本重要性动态调整损失权重 - 对小收益使用稳定的 IMADL，对大收益使用激进的 M2。

**变体类型：** 自适应权重  
**基础损失：** IMADL（小收益）+ M2（大收益）  
**权重函数：** 基于 |y_true| 的指数衰减  
**参数：** λ（衰减率）

---

## 数学定义

### 公式

```
loss = IMADL * w_imadl + M2 * w_m2

其中：
  w_imadl = exp(-λ * |y_true|)
  w_m2 = 1 - exp(-λ * |y_true|)
  λ ∈ {1.0, 5.0, 10.0}
```

**完整表达式：**
```
loss = imadl_loss(y_true, y_pred) * exp(-λ * |y_true|) + 
       hybrid_mul_m2(y_true, y_pred) * (1 - exp(-λ * |y_true|))
```

### 权重函数行为

**对于小 |y_true|（例如 0.01）：**
- w_imadl ≈ 1.0（IMADL 主导）
- w_m2 ≈ 0.0（M2 影响最小）
- **原理：** 小收益常见，使用稳定的 IMADL

**对于大 |y_true|（例如 0.10）：**
- w_imadl ≈ 0.0（IMADL 影响最小）
- w_m2 ≈ 1.0（M2 主导）
- **原理：** 大收益罕见但重要，使用激进的 M2

### 不同 λ 的权重曲线

| |y_true| | λ=1.0 | λ=5.0 | λ=10.0 |
|---------|-------|-------|--------|
| 0.01 | w_imadl=0.99, w_m2=0.01 | w_imadl=0.95, w_m2=0.05 | w_imadl=0.90, w_m2=0.10 |
| 0.05 | w_imadl=0.95, w_m2=0.05 | w_imadl=0.78, w_m2=0.22 | w_imadl=0.61, w_m2=0.39 |
| 0.10 | w_imadl=0.90, w_m2=0.10 | w_imadl=0.61, w_m2=0.39 | w_imadl=0.37, w_m2=0.63 |
| 0.20 | w_imadl=0.82, w_m2=0.18 | w_imadl=0.37, w_m2=0.63 | w_imadl=0.14, w_m2=0.86 |

**解释：**
- **λ=1.0：** 慢速过渡，即使对大收益 IMADL 也主导
- **λ=5.0：** 中等过渡，平衡权重
- **λ=10.0：** 快速过渡，M2 对中等到大收益主导

---

## 原理

### 为什么使用自适应权重？

1. **样本异质性：**
   - 小收益（|y| < 0.05）：约 80% 的样本，低信噪比
   - 大收益（|y| > 0.10）：约 10% 的样本，高交易价值
   - 不同样本需要不同的损失函数

2. **损失函数优势：**
   - **IMADL：** 稳定，适合噪声小收益
   - **M2：** 激进，适合捕捉大波动
   - **固定组合：** 对所有样本一视同仁（次优）

3. **自适应优势：**
   - 基于样本重要性自动调整
   - 无需手动调整 α（如变体 1）
   - 理论上比线性组合更有原则

### 不同 λ 的预期行为

| λ | 过渡速度 | 预期行为 |
|---|---------|---------|
| 1.0 | 慢 | IMADL 主导，保守 |
| 5.0 | 中等 | 平衡，自适应 |
| 10.0 | 快 | M2 对中等收益主导 |

**假设：** λ=5.0 将提供最优平衡，随着收益幅度增加从 IMADL 平滑过渡到 M2。

---

## 参数网格

### 3 个配置

1. **adaptive_hybrid_10**（λ=1.0）：慢速过渡，IMADL 主导
2. **adaptive_hybrid_50**（λ=5.0）：中等过渡，平衡
3. **adaptive_hybrid_100**（λ=10.0）：快速过渡，M2 主导

---

## 代码实现

### 添加到 `losses.py`

```python
# 在 EXPERIMENT_LOSS_NAMES 元组中
EXPERIMENT_LOSS_NAMES = (
    # ... 现有损失 ...
    "adaptive_hybrid_10",
    "adaptive_hybrid_50",
    "adaptive_hybrid_100",
)

# 自适应权重的辅助函数
def adaptive_hybrid_loss(y_true: torch.Tensor, 
                        y_pred: torch.Tensor, 
                        lambda_param: float) -> torch.Tensor:
    """
    基于 |y_true| 权重 IMADL 和 M2 的自适应混合损失。
    
    参数：
        y_true: 真实收益
        y_pred: 预测收益
        lambda_param: 指数权重的衰减率
    
    返回：
        IMADL 和 M2 的加权组合
    """
    # 计算基础损失
    imadl = imadl_loss(y_true, y_pred)
    m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    
    # 计算自适应权重
    abs_y = torch.abs(y_true)
    w_imadl = torch.exp(-lambda_param * abs_y)
    w_m2 = 1.0 - w_imadl
    
    # 加权组合
    loss = imadl * w_imadl.mean() + m2 * w_m2.mean()
    return loss

# 在 get_experiment_loss_fn() 函数中
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... 现有损失实现 ...
    
    # 变体 4：自适应混合
    if name_lower == "adaptive_hybrid_10":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=1.0
        )
    
    if name_lower == "adaptive_hybrid_50":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=5.0
        )
    
    if name_lower == "adaptive_hybrid_100":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=10.0
        )
```

### 创建运行脚本

创建 3 个文件：`run_sanity_check_adaptive_hybrid_10.py`，`run_sanity_check_adaptive_hybrid_50.py`，`run_sanity_check_adaptive_hybrid_100.py`

**模板：**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (Adaptive Hybrid λ=1.0)")
    args = parser.parse_args()
    run_sanity_check("adaptive_hybrid_10", args)

if __name__ == "__main__":
    main()
```

### 更新 `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... 现有映射 ...
    "adaptive_hybrid_10": "run_sanity_check_adaptive_hybrid_10.py",
    "adaptive_hybrid_50": "run_sanity_check_adaptive_hybrid_50.py",
    "adaptive_hybrid_100": "run_sanity_check_adaptive_hybrid_100.py",
}
```

---

## 实验配置

### Phase 2.1：初步筛选

**运行次数：** 3 个损失 × 3 个种子 = 9 次运行

| 损失名称 | λ | 种子 | 权重上限 | 测试周期 |
|---------|---|------|---------|---------|
| adaptive_hybrid_10 | 1.0 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| adaptive_hybrid_50 | 5.0 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| adaptive_hybrid_100 | 10.0 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |

---

## 预期结果

### 性能预测

**低 λ（1.0）：慢速过渡**
- 预期 Sharpe：0.5-0.6
- 预期 CV：0.8-1.0
- 失败率：0-5%
- 行为：保守，接近 IMADL

**中等 λ（5.0）：中等过渡**
- 预期 Sharpe：0.6-0.8
- 预期 CV：0.9-1.1
- 失败率：5-15%
- 行为：平衡，适应样本重要性

**高 λ（10.0）：快速过渡**
- 预期 Sharpe：0.7-0.9
- 预期 CV：1.1-1.3
- 失败率：10-20%
- 行为：激进，对中等收益更接近 M2

### 成功标准

**最低要求：**
- 平均 Sharpe > 0.6
- CV < 1.1
- 失败率 < 15%

**理想性能：**
- 平均 Sharpe > 0.7
- CV < 1.0
- 失败率 < 10%
- 优于 IMADL 和 M2 基线

---

## 分析计划

### 关键问题

1. **自适应权重是否优于固定组合？**
   - 与变体 1（IMADL+M2 线性）比较
   - 检查自适应方法是否更鲁棒

2. **最优 λ 是多少？**
   - 哪个衰减率提供最佳风险调整收益？

3. **权重在实践中如何分布？**
   - 分析跨样本的实际权重分布
   - 验证小收益使用 IMADL，大收益使用 M2

4. **过渡是否平滑？**
   - 检查不连续性或不稳定性
   - 验证训练期间的梯度流

### 可视化

1. **λ vs Sharpe 图：** 显示性能趋势
2. **λ vs CV 图：** 显示稳定性趋势
3. **权重分布：** 跨样本的 w_imadl 和 w_m2 直方图
4. **权重 vs |y_true| 散点图：** 验证指数衰减模式
5. **样本级分析：** 显示哪些样本使用哪个损失

---

## 实现说明

### 替代权重函数

当前实现使用指数衰减。其他选项：

**变体 A：Sigmoid 权重**
```python
def sigmoid_weighting(y_true, lambda_param):
    w_m2 = torch.sigmoid(lambda_param * (torch.abs(y_true) - 0.05))
    w_imadl = 1.0 - w_m2
    return w_imadl, w_m2
```

**变体 B：基于阈值**
```python
def threshold_weighting(y_true, threshold=0.05):
    w_imadl = (torch.abs(y_true) < threshold).float()
    w_m2 = 1.0 - w_imadl
    return w_imadl, w_m2
```

**变体 C：多项式衰减**
```python
def polynomial_weighting(y_true, power=2):
    abs_y = torch.abs(y_true)
    w_m2 = abs_y ** power
    w_imadl = 1.0 - w_m2
    return w_imadl, w_m2
```

**建议：** 从指数衰减开始（最有原则）。如果结果有希望，在 Phase 2.2 中探索替代方案。

### 梯度考虑

**潜在问题：** 如果权重变化太快，自适应权重可能导致梯度不稳定。

**解决方案：** 使用平滑指数函数（已实现）并监控：
1. 训练期间损失曲线的平滑性
2. 梯度范数
3. 权重分布稳定性

如果出现不稳定，考虑：
- 减少 λ 范围
- 添加权重裁剪
- 使用权重的移动平均

---

## 论文贡献

### 方法创新

- **自适应损失权重：** 首次基于样本重要性的交易损失组合
- **有原则的方法：** 由样本异质性理论驱动
- **自动调整：** 无需手动 α 选择（不同于变体 1）

### 预期发现

1. **最优 λ：** 可能 λ=5.0 提供最佳平衡
2. **自适应优势：** 相比固定组合改进 5-10%
3. **权重分布：** 验证小收益使用 IMADL，大收益使用 M2
4. **鲁棒性：** 比纯 M2 更好的跨种子稳定性

### 理论洞察

- 为什么样本自适应权重优于固定权重
- 收益幅度在损失函数选择中的作用
- 简单性（固定 α）和自适应性（指数 λ）之间的权衡

---

## 与变体 1 的比较

| 方面 | 变体 1（线性）| 变体 4（自适应）|
|------|-------------|---------------|
| 权重 | 所有样本固定 α | 基于 \|y_true\| 自适应 |
| 参数 | 7 个值（α=0.2-0.8）| 3 个值（λ=1.0, 5.0, 10.0）|
| 复杂度 | 简单，可解释 | 更复杂，有原则 |
| 灵活性 | 所有样本一个权重 | 每个样本不同权重 |
| 预期性能 | 良好基线 | 可能更好 |

**假设：** 变体 4 将通过自适应权重比变体 1 表现好 5-10% 的 Sharpe。

---

**文档版本：** v1.0  
**创建日期：** 2026-04-26  
**作者：** Yirong Yu  
**状态：** 准备实施
