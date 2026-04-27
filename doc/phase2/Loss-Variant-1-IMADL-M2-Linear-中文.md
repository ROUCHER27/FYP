# 损失变体 1：IMADL + M2 线性组合

> **目标：** 在稳定的 IMADL 和高收益的 M2 之间找到最优线性插值，以平衡风险和回报。

**变体类型：** 线性组合  
**基础损失：** IMADL（稳定）+ M2（激进）  
**参数：** α（IMADL 权重）

---

## 数学定义

### 公式

```
loss = α * IMADL + (1-α) * hybrid_mul_m2

其中：
  IMADL = imadl_loss(y_true, y_pred)
  hybrid_mul_m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
  α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
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

**M2（混合乘法，λ_dir=2.0）：**
```python
def hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0, delta=1.0):
    dir_term = _normalized_direction_term(y_true, y_pred)
    mag_term = huber_loss(y_true, y_pred, delta=delta)
    return (dir_term * mag_term * lambda_dir).mean()
```

**线性组合：**
```python
def imadl_m2_linear_loss(y_true, y_pred, alpha):
    imadl = imadl_loss(y_true, y_pred)
    m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    return alpha * imadl + (1 - alpha) * m2
```

---

## 原理

### 为什么选择这个组合？

1. **IMADL 的优势：**
   - 跨种子最稳定（CV=0.892）
   - 持续正收益（Sharpe=0.464）
   - 低失败率（0%）
   - 平衡的方向和幅度项

2. **M2 的优势：**
   - 最高平均 Sharpe（0.914）
   - 最佳情况性能（seed=62 上 Sharpe=2.285）
   - 强方向信号（λ_dir=2.0）

3. **互补特性：**
   - IMADL：加法结构（dir + mag）
   - M2：乘法结构（dir × mag × λ）
   - 线性组合允许平滑插值

### 不同 α 的预期行为

| α | IMADL 权重 | M2 权重 | 预期行为 |
|---|-----------|---------|---------|
| 0.2 | 20% | 80% | 激进，高方差，高潜在收益 |
| 0.3 | 30% | 70% | 中等激进 |
| 0.4 | 40% | 60% | 偏向 M2 的平衡 |
| 0.5 | 50% | 50% | 等权重 |
| 0.6 | 60% | 40% | 偏向 IMADL 的平衡 |
| 0.7 | 70% | 30% | 中等保守 |
| 0.8 | 80% | 20% | 保守，低方差，稳定 |

**假设：** 最优 α 将在 [0.4, 0.6] 范围内，平衡 IMADL 的稳定性和 M2 的高收益。

---

## 参数网格

### 7 个配置

1. **imadl_m2_linear_02**（α=0.2）：20% IMADL，80% M2
2. **imadl_m2_linear_03**（α=0.3）：30% IMADL，70% M2
3. **imadl_m2_linear_04**（α=0.4）：40% IMADL，60% M2
4. **imadl_m2_linear_05**（α=0.5）：50% IMADL，50% M2
5. **imadl_m2_linear_06**（α=0.6）：60% IMADL，40% M2
6. **imadl_m2_linear_07**（α=0.7）：70% IMADL，30% M2
7. **imadl_m2_linear_08**（α=0.8）：80% IMADL，20% M2

---

## 代码实现

### 添加到 `losses.py`

```python
# 在 EXPERIMENT_LOSS_NAMES 元组中
EXPERIMENT_LOSS_NAMES = (
    # ... 现有损失 ...
    "imadl_m2_linear_02",
    "imadl_m2_linear_03",
    "imadl_m2_linear_04",
    "imadl_m2_linear_05",
    "imadl_m2_linear_06",
    "imadl_m2_linear_07",
    "imadl_m2_linear_08",
)

# 在 get_experiment_loss_fn() 函数中
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... 现有损失实现 ...
    
    # 变体 1：IMADL + M2 线性组合
    if name_lower == "imadl_m2_linear_02":
        return lambda y_true, y_pred: (
            0.2 * imadl_loss(y_true, y_pred) + 
            0.8 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_03":
        return lambda y_true, y_pred: (
            0.3 * imadl_loss(y_true, y_pred) + 
            0.7 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_04":
        return lambda y_true, y_pred: (
            0.4 * imadl_loss(y_true, y_pred) + 
            0.6 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_05":
        return lambda y_true, y_pred: (
            0.5 * imadl_loss(y_true, y_pred) + 
            0.5 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_06":
        return lambda y_true, y_pred: (
            0.6 * imadl_loss(y_true, y_pred) + 
            0.4 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_07":
        return lambda y_true, y_pred: (
            0.7 * imadl_loss(y_true, y_pred) + 
            0.3 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_08":
        return lambda y_true, y_pred: (
            0.8 * imadl_loss(y_true, y_pred) + 
            0.2 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
```

### 创建运行脚本

创建 7 个文件：`run_sanity_check_imadl_m2_linear_02.py` 到 `run_sanity_check_imadl_m2_linear_08.py`

**模板：**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (IMADL+M2 Linear α=0.2)")
    args = parser.parse_args()
    run_sanity_check("imadl_m2_linear_02", args)

if __name__ == "__main__":
    main()
```

### 更新 `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... 现有映射 ...
    "imadl_m2_linear_02": "run_sanity_check_imadl_m2_linear_02.py",
    "imadl_m2_linear_03": "run_sanity_check_imadl_m2_linear_03.py",
    "imadl_m2_linear_04": "run_sanity_check_imadl_m2_linear_04.py",
    "imadl_m2_linear_05": "run_sanity_check_imadl_m2_linear_05.py",
    "imadl_m2_linear_06": "run_sanity_check_imadl_m2_linear_06.py",
    "imadl_m2_linear_07": "run_sanity_check_imadl_m2_linear_07.py",
    "imadl_m2_linear_08": "run_sanity_check_imadl_m2_linear_08.py",
}
```

---

## 实验配置

### Phase 2.1：初步筛选

**运行次数：** 7 个损失 × 3 个种子 = 21 次运行

| 损失名称 | α | 种子 | 权重上限 | 测试周期 |
|---------|---|------|---------|---------|
| imadl_m2_linear_02 | 0.2 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_03 | 0.3 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_04 | 0.4 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_05 | 0.5 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_06 | 0.6 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_07 | 0.7 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |
| imadl_m2_linear_08 | 0.8 | 42, 52, 62 | 0.05 | 1995-01 至 1996-12 |

---

## 预期结果

### 性能预测

**低 α（0.2-0.3）：M2 主导**
- 预期 Sharpe：0.7-1.0
- 预期 CV：1.2-1.5
- 失败率：20-30%
- 行为：高方差，高潜在收益

**中等 α（0.4-0.6）：平衡**
- 预期 Sharpe：0.6-0.8
- 预期 CV：0.8-1.1
- 失败率：10-20%
- 行为：最优平衡点

**高 α（0.7-0.8）：IMADL 主导**
- 预期 Sharpe：0.5-0.6
- 预期 CV：0.7-0.9
- 失败率：0-10%
- 行为：保守，稳定

### 成功标准

**最低要求：**
- 平均 Sharpe > 0.5（优于 IMADL 基线）
- CV < 1.0（优于 M2 基线）
- 失败率 < 20%

**理想性能：**
- 平均 Sharpe > 0.7
- CV < 0.9
- 失败率 < 10%
- 在所有 3 个种子上表现一致

---

## 分析计划

### 跟踪指标

1. **主要指标：**
   - 3 个种子的平均 Sharpe 比率
   - Sharpe 标准差
   - 变异系数（CV）
   - 失败率（Sharpe < 0）

2. **次要指标：**
   - 累计收益
   - 最大回撤
   - 月度收益波动率
   - 方向准确率

3. **对比指标：**
   - vs IMADL 基线（Sharpe=0.464，CV=0.892）
   - vs M2 基线（Sharpe=0.914，CV=1.396）
   - 改进百分比

### 可视化

1. **α vs Sharpe 图：** 显示 Sharpe 如何随 α 变化
2. **α vs CV 图：** 显示稳定性趋势
3. **帕累托前沿：** 绘制 Sharpe vs CV 以找到最优权衡
4. **种子敏感性：** 箱线图显示每个 α 跨种子的方差

---

## 论文贡献

### 方法创新

- **新颖的组合策略：** 首次系统探索 IMADL+M2 线性插值
- **参数扫描：** 全面的 α 网格搜索以找到最优平衡
- **稳定性-收益权衡：** 量化帕累托前沿

### 预期发现

1. **最优 α 识别：** 可能在 [0.4, 0.6] 范围内
2. **平滑插值：** 线性组合提供连续权衡
3. **鲁棒性改进：** 相比纯 M2 降低方差
4. **性能提升：** 高于纯 IMADL 的 Sharpe

### 理论洞察

- 为什么线性组合对损失函数有效
- 加法（IMADL）和乘法（M2）结构之间的权衡
- α 在控制风险-收益特征中的作用

---

**文档版本：** v1.0  
**创建日期：** 2026-04-26  
**作者：** Yirong Yu  
**状态：** 准备实施
