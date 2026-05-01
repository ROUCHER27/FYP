# 损失变体 2：IMADL + GMADL 加权组合

> **目标：** 结合两个结构相似的基于 MAD 的损失，以在保持稳定性的同时提高收益。

**变体类型：** 加权组合  
**基础损失：** IMADL + GMADL（均基于 MAD）  
**参数：** β（IMADL 权重）

---

## 数学定义

### 公式

```
loss = β * IMADL + (1-β) * GMADL

其中：
  IMADL = imadl_loss(y_true, y_pred)
  GMADL = gmadl_loss(y_true, y_pred)
  β ∈ {0.3, 0.5, 0.7}
```

### 组件分解

**IMADL（逆均值绝对方向损失）：**
```python
def imadl_loss(y_true, y_pred, a=100.0, b=2.0, eps=1e-8):
    product = a * y_true * y_pred
    dir_penalty = 1.0 - torch.sigmoid(product)
    weight = torch.abs(y_true) ** b
    mean_weight = weight.mean() + eps
    normalized_weight = weight / mean_weight
    weighted_penalty = dir_penalty * normalized_weight
    mag_term = (y_true - y_pred) ** 2
    return (weighted_penalty + mag_term).mean()
```

**GMADL（广义均值绝对方向损失）：**
```python
def gmadl_loss(y_true, y_pred, a=100.0, b=2.0, eps=1e-8):
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    loss = -(sigmoid - 0.5) * torch.abs(y_true) ** b
    return loss.mean()
```

**关键区别：**
- **IMADL：** 包含幅度项（MSE），归一化权重
- **GMADL：** 仅方向，无幅度项，未归一化权重
- **两者：** 都使用 sigmoid 作为方向信号，|y|^b 作为重要性权重

---

## 原理

### 为什么选择这个组合？

1. **结构相似性：**
   - 两者都使用 sigmoid(a * y_true * y_pred) 作为方向信号
   - 两者都使用 |y_true|^b 作为重要性权重
   - 相似的数学基础（基于 MAD）

2. **互补特性：**
   - IMADL：添加幅度项 → 更好的预测准确性
   - GMADL：纯方向焦点 → 更强的交易信号
   - IMADL：归一化权重 → 跨样本稳定
   - GMADL：未归一化权重 → 强调大收益

3. **Phase 1.5 性能：**
   - IMADL：Sharpe=0.464，CV=0.892（稳定）
   - GMADL：Sharpe=0.307，CV=1.168（中等）
   - 两者失败率均为 0%

### 不同 β 的预期行为

| β | IMADL 权重 | GMADL 权重 | 预期行为 |
|---|-----------|-----------|---------|
| 0.3 | 30% | 70% | GMADL 主导，更强的方向信号 |
| 0.5 | 50% | 50% | 等权重，平衡方法 |
| 0.7 | 70% | 30% | IMADL 主导，更稳定 |

**假设：** β=0.5 将提供最佳平衡，结合 IMADL 的稳定性和 GMADL 的方向强度。

---

## 参数网格

### 3 个配置

1. **imadl_gmadl_weighted_03**（β=0.3）：30% IMADL，70% GMADL
2. **imadl_gmadl_weighted_05**（β=0.5）：50% IMADL，50% GMADL
3. **imadl_gmadl_weighted_07**（β=0.7）：70% IMADL，30% GMADL

---

## 代码实现

### 添加到 `losses.py`

```python
# 在 EXPERIMENT_LOSS_NAMES 元组中
EXPERIMENT_LOSS_NAMES = (
    # ... 现有损失 ...
    "imadl_gmadl_weighted_03",
    "imadl_gmadl_weighted_05",
    "imadl_gmadl_weighted_07",
)

# 在 get_experiment_loss_fn() 函数中
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... 现有损失实现 ...
    
    # 变体 2：IMADL + GMADL 加权组合
    if name_lower == "imadl_gmadl_weighted_03":
        return lambda y_true, y_pred: (
            0.3 * imadl_loss(y_true, y_pred) + 
            0.7 * gmadl_loss(y_true, y_pred)
        )
    
    if name_lower == "imadl_gmadl_weighted_05":
        return lambda y_true, y_pred: (
            0.5 * imadl_loss(y_true, y_pred) + 
            0.5 * gmadl_loss(y_true, y_pred)
        )
    
    if name_lower == "imadl_gmadl_weighted_07":
        return lambda y_true, y_pred: (
            0.7 * imadl_loss(y_true, y_pred) + 
            0.3 * gmadl_loss(y_true, y_pred)
        )
```

### 创建运行脚本

创建 3 个文件：`run_sanity_check_imadl_gmadl_weighted_03.py`，`run_sanity_check_imadl_gmadl_weighted_05.py`，`run_sanity_check_imadl_gmadl_weighted_07.py`

**模板：**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (IMADL+GMADL Weighted β=0.3)")
    args = parser.parse_args()
    run_sanity_check("imadl_gmadl_weighted_03", args)

if __name__ == "__main__":
    main()
```

### 更新 `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... 现有映射 ...
    "imadl_gmadl_weighted_03": "run_sanity_check_imadl_gmadl_weighted_03.py",
    "imadl_gmadl_weighted_05": "run_sanity_check_imadl_gmadl_weighted_05.py",
    "imadl_gmadl_weighted_07": "run_sanity_check_imadl_gmadl_weighted_07.py",
}
```

---

## 实验配置

### Phase 2.1：初步筛选

**运行次数：** 3 个损失 × 3 个种子 = 9 次运行

| 损失名称 | β | 种子 | 权重上限 | 测试周期 |
|---------|---|------|---------|---------|
| imadl_gmadl_weighted_03 | 0.3 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_gmadl_weighted_05 | 0.5 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_gmadl_weighted_07 | 0.7 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |

---

## 预期结果

### 性能预测

**低 β（0.3）：GMADL 主导**
- 预期 Sharpe：0.35-0.45
- 预期 CV：1.0-1.2
- 失败率：0-10%
- 行为：更强的方向信号，中等稳定性

**中等 β（0.5）：平衡**
- 预期 Sharpe：0.40-0.50
- 预期 CV：0.9-1.1
- 失败率：0-5%
- 行为：方向和幅度的最优平衡

**高 β（0.7）：IMADL 主导**
- 预期 Sharpe：0.45-0.55
- 预期 CV：0.8-1.0
- 失败率：0%
- 行为：最稳定，更接近 IMADL 基线

### 成功标准

**最低要求：**
- 平均 Sharpe > 0.4（优于 GMADL 基线）
- CV < 1.0
- 失败率 < 10%

**理想性能：**
- 平均 Sharpe > 0.5（优于 IMADL 基线）
- CV < 0.9
- 失败率 = 0%
- 在所有 3 个种子上表现一致

---

## 分析计划

### 关键问题

1. **组合是否优于单个损失？**
   - 与 IMADL（Sharpe=0.464）和 GMADL（Sharpe=0.307）比较
   
2. **最优 β 是多少？**
   - 哪种权重提供最佳风险调整收益？
   
3. **幅度项是否重要？**
   - IMADL 有 MSE 项，GMADL 没有
   - 添加幅度是否改善交易性能？

4. **归一化是否重要？**
   - IMADL 归一化权重，GMADL 不归一化
   - 对稳定性的影响？

### 可视化

1. **β vs Sharpe 图：** 显示性能趋势
2. **β vs CV 图：** 显示稳定性趋势
3. **组件分析：** 将损失分解为方向和幅度项
4. **种子敏感性：** 比较跨种子的方差

---

## 论文贡献

### 方法创新

- **基于 MAD 的组合：** 首次探索组合两个基于 MAD 的损失
- **结构分析：** 理解幅度项和归一化的影响
- **互补优势：** 利用 IMADL 的稳定性和 GMADL 的方向焦点

### 预期发现

1. **最优 β：** 可能是 β=0.5 或 β=0.7
2. **幅度项重要性：** 量化 MSE 项对交易性能的影响
3. **归一化效果：** 理解权重归一化在稳定性中的作用
4. **适度改进：** 预期相比 IMADL 有 5-10% 的 Sharpe 改进

### 理论洞察

- 为什么基于 MAD 的损失比混合损失更稳定
- 幅度项在平衡方向和准确性中的作用
- 权重归一化对跨种子鲁棒性的影响

---

**文档版本：** v1.0  
**创建日期：** 2026-04-26  
**作者：** Yirong Yu  
**状态：** 准备实施
