# 损失函数实现分析报告

**生成日期**: 2026-04-22  
**分析对象**: `Model_Train/losses.py` vs `doc/plan_semester2.pdf` 和 `doc/Pre-plan-semester2.md`

---

## 📋 总体对齐情况

| 损失函数 | 文档来源 | 对齐状态 | 实现方式 |
|---------|---------|---------|---------|
| IMADL | 3.2.1 Improved MADL | ✅ 改进版 | Rebalanced with normalization |
| Directional Huber | 3.2.2 Directional Huber | ⚠️ 加法型 | Additive instead of multiplicative |
| Hybrid Add | Pre-plan 方案 A | ✅ 完全一致 | Additive hybrid |
| Hybrid Mul | Pre-plan 方案 B | ✅ 完全一致 | Multiplicative hybrid |

---

## 1️⃣ IMADL (Improved MADL)

### 📖 文档来源：`plan_semester2.pdf` Section 3.2.1

**文档原始公式**:

$$
L_i = (1 - \sigma(a \cdot y_i \cdot \hat{y}_i)) \cdot |y_i|^b + c \cdot |y_i - \hat{y}_i|^d
$$

**公式含义**:
- **第一项**: $(1 - \sigma(a \cdot y \cdot \hat{y})) \cdot |y|^b$ - 方向惩罚项
  - $\sigma(a \cdot y \cdot \hat{y})$: sigmoid 函数，方向一致时接近 1，方向相反时接近 0
  - $1 - \sigma(\cdots)$: 翻转后，方向一致惩罚小，方向相反惩罚大
  - $|y|^b$: 按真实收益率的幅度加权，大波动更重要
  
- **第二项**: $c \cdot |y - \hat{y}|^d$ - 幅度误差项
  - 直接惩罚预测误差
  - 修复 GMADL 只看方向不看精度的问题

### 💻 代码实现：`losses.py:109-125`

```python
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
```

**辅助函数** `_normalized_direction_term` (losses.py:85-97):
```python
def _normalized_direction_term(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    product = a * y_true * y_pred
    dir_penalty = 1.0 - torch.sigmoid(product)  # (1 - σ(a·y·ŷ))
    weight = torch.abs(y_true) ** b             # |y|^b
    mean_weight = weight.mean() + eps
    normalized_weight = weight / mean_weight    # 归一化
    return dir_penalty * normalized_weight
```

### 🔍 差异分析

| 组件 | 文档版本 | 代码版本 | 差异说明 |
|-----|---------|---------|---------|
| 方向项基础 | $(1 - \sigma(a \cdot y \cdot \hat{y}))$ | $(1 - \sigma(a \cdot y \cdot \hat{y}))$ | ✅ 完全一致 |
| 权重项 | $\|y\|^b$ | $\|y\|^b / \text{mean}(\|y\|^b)$ | ⚠️ **归一化处理** |
| 幅度项 | $c \cdot \|y - \hat{y}\|^d$ | $(y - \hat{y})^2$ | ⚠️ 固定为平方误差 |
| 组合方式 | 直接相加 | $\lambda_{\text{dir}} \cdot \text{dir} + \lambda_{\text{mag}} \cdot \text{mag}$ | ✅ 加权相加 |

### 📊 为什么要归一化？

**问题来源**: `Pre-plan-semester2.md` 第 86-101 行

> 例如：y = 0.01, ŷ = -0.1
> - directional term ≈ 5.25e-05
> - magnitude term (若 d=2, c=1) ≈ 0.0121
> 
> **方向项只占幅度项的 0.4% 左右**

**股票收益率特点**:
- 月收益率通常很小（-0.1 到 +0.1 之间）
- $|y|^b$ 在小收益率下会非常小
- 如果不归一化，方向项会被幅度项完全淹没
- 模型会退化成纯 MSE，失去方向敏感性

**归一化效果**:
```python
# 假设 batch 中的 |y|^b 值为 [0.0001, 0.0004, 0.0009, 0.0016]
mean_weight = 0.00075
normalized = [0.133, 0.533, 1.2, 2.133]  # 归一化后量级接近 1
```

### ✅ 结论

代码实现了 **Rebalanced 版本**，这是有意的改进：
- 解决了原始公式在小收益率场景下方向项被淹没的问题
- 符合 `Pre-plan-semester2.md` 第 109-127 行的建议
- 论文中应说明这是 "scale rebalancing" 改进

---

## 2️⃣ Directional Huber

### 📖 文档来源：`plan_semester2.pdf` Section 3.2.2

**文档原始公式**:

$$
L_i = \text{pen}(y_i, \hat{y}_i) \cdot H_\delta(y_i - \hat{y}_i)
$$

其中:

$$
\text{pen}(y, \hat{y}) = 0.5 \cdot (1 - \tanh(a \cdot y \cdot \hat{y}))
$$

$$
H_\delta(e) = \begin{cases}
0.5 \cdot e^2 & \text{if } |e| \leq \delta \\
\delta \cdot (|e| - 0.5\delta) & \text{if } |e| > \delta
\end{cases}
$$

**公式含义**:
- **方向惩罚项** $\text{pen}(y, \hat{y})$:
  - 使用 $\tanh$ 平滑符号函数
  - 方向一致时 pen ≈ 0，方向相反时 pen ≈ 1
  - 范围: [0, 1]

- **Huber 误差项** $H_\delta(e)$:
  - 小误差用平方惩罚（像 MSE）
  - 大误差用线性惩罚（鲁棒性）
  - 对极端值不敏感

- **乘法组合**:
  - 方向对时，整体损失被缩小
  - 方向错时，误差被放大惩罚

### 💻 代码实现：`losses.py:128-145`

```python
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
    loss = lambda_dir * dir_penalty + lambda_hub * huber_term  # ⚠️ 加法型
    return _reduce(loss, reduction)
```

**Huber 辅助函数** (losses.py:100-106):
```python
def _huber_term(error: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    abs_error = torch.abs(error)
    return torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )
```

### 🔍 关键差异：乘法 vs 加法

| 维度 | 文档版（乘法型） | 代码版（加法型） |
|-----|----------------|----------------|
| **公式** | $\text{pen} \times H_\delta(e)$ | $\lambda_{\text{dir}} \cdot \text{pen} + \lambda_{\text{hub}} \cdot H_\delta(e)$ |
| **方向项** | $0.5 \cdot (1 - \tanh(a \cdot y \cdot \hat{y}))$ | $0.5 \cdot (1 - \tanh(a \cdot y \cdot \hat{y}))$ ✅ 一致 |
| **Huber 项** | $H_\delta(y - \hat{y})$ | $H_\delta(y - \hat{y})$ ✅ 一致 |
| **组合方式** | 相乘 | 加权相加 ⚠️ **不同** |

### 📊 两种形式的行为差异

**场景 1: 方向对，误差小**
```
y = +0.10, ŷ = +0.09
pen ≈ 0.0, H_δ(e) ≈ 0.00005

乘法型: L = 0.0 × 0.00005 = 0.000
加法型: L = 1.0×0.0 + 1.0×0.00005 = 0.00005
```

**场景 2: 方向对，误差大**
```
y = +0.10, ŷ = +0.02
pen ≈ 0.0, H_δ(e) ≈ 0.00032

乘法型: L = 0.0 × 0.00032 = 0.000  ⚠️ 几乎不惩罚
加法型: L = 1.0×0.0 + 1.0×0.00032 = 0.00032  ✅ 仍然惩罚
```

**场景 3: 方向错，误差小**
```
y = +0.10, ŷ = -0.01
pen ≈ 1.0, H_δ(e) ≈ 0.00061

乘法型: L = 1.0 × 0.00061 = 0.00061
加法型: L = 1.0×1.0 + 1.0×0.00061 = 1.00061
```

**场景 4: 方向错，误差大**
```
y = +0.10, ŷ = -0.10
pen ≈ 1.0, H_δ(e) ≈ 0.00195

乘法型: L = 1.0 × 0.00195 = 0.00195
加法型: L = 1.0×1.0 + 1.0×0.00195 = 1.00195
```

### 🎯 直观对比

假设真实值 y = +0.10，不同预测值的损失：

```
预测值 ŷ:  -0.10  -0.05   0.00   0.05   0.10   0.15   0.20
          ←─────方向错─────┤←─────方向对─────→

文档版（乘法）:
损失:      ████   ███    ██     █      ▁      ▁      ▁
          高     高     中     低     极低    极低    极低
          ↑ 方向错被放大        ↑ 方向对几乎不惩罚

代码版（加法）:
损失:      ████   ███    ██     █      ▁      █      ██
          高     高     中     低     最低    低     中
          ↑ 方向错被额外惩罚    ↑ 方向对但误差大仍惩罚
```

### 📖 为什么选择加法型？

**来源**: `Pre-plan-semester2.md` 第 217-238 行

> **为什么我更推荐加法**
> - 方向错时，penalty 可以直接增大
> - same sign 但 magnitude 很差时，Huber 仍然有效
> - 两个效应更清晰可拆解

**实际原因**:

1. **避免过度宽容**: 乘法型对"方向对但误差大"过于宽容
   - 真实涨 10%，预测涨 1% vs 涨 100% 的经济意义完全不同
   - 乘法型会给这两种情况几乎相同的低损失

2. **更好调参**: 加法型可以通过 λ_dir 和 λ_hub 灵活控制
   - 如果更重视方向，增大 λ_dir
   - 如果更重视精度，增大 λ_hub

3. **更稳定训练**: 加法型的梯度更平滑
   - 乘法型当 pen ≈ 0 时，梯度可能消失
   - 加法型始终保持 Huber 项的梯度

### ✅ 结论

代码采用 **加法型** 是更优的选择：
- 在保留方向敏感性的同时，不会忽视预测精度
- 更适合股票预测任务的实际需求
- 训练更稳定，调参更灵活

---

## 3️⃣ Hybrid Directional-Huber (Add)

### 📖 文档来源：`Pre-plan-semester2.md` 第 242-253 行 (方案 A)

**文档公式**:

$$
L_i = \lambda_1 \cdot D_i + \lambda_2 \cdot H_\delta(e_i)
$$

其中:

$$
D_i = (1 - \sigma(a \cdot y \cdot \hat{y})) \cdot \tilde{w}_i
$$

$$
\tilde{w}_i = \frac{|y|^b}{\text{mean}(|y|^b) + \epsilon}
$$

$$
e_i = y_i - \hat{y}_i
$$

**设计思想**:
- 结合 IMADL 的归一化方向项
- 结合 Directional Huber 的鲁棒误差项
- 加法组合，两个目标独立优化

### 💻 代码实现：`losses.py:148-165`

```python
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
```

### ✅ 对齐情况

| 组件 | 文档 | 代码 | 状态 |
|-----|------|------|------|
| 方向项 | $(1 - \sigma(a \cdot y \cdot \hat{y})) \cdot \tilde{w}$ | `_normalized_direction_term` | ✅ 完全一致 |
| Huber 项 | $H_\delta(e)$ | `_huber_term` | ✅ 完全一致 |
| 组合方式 | $\lambda_1 \cdot D + \lambda_2 \cdot H$ | `λ_dir·dir + λ_hub·hub` | ✅ 完全一致 |

### 🎯 优势

1. **三重保护**:
   - 方向敏感（来自 GMADL/IMADL）
   - 精度要求（来自误差项）
   - 鲁棒性（来自 Huber）

2. **灵活调参**:
   - λ_dir 控制方向重要性
   - λ_hub 控制精度重要性

3. **适合股票预测**:
   - 方向错误会被额外惩罚
   - 方向对但误差大也会被惩罚
   - 对极端收益率鲁棒

---

## 4️⃣ Hybrid Directional-Huber (Mul)

### 📖 文档来源：`Pre-plan-semester2.md` 第 255-268 行 (方案 B)

**文档公式**:

$$
L_i = (1 + \lambda \cdot D_i) \cdot H_\delta(e_i)
$$

其中:

$$
D_i = (1 - \sigma(a \cdot y \cdot \hat{y})) \cdot \tilde{w}_i
$$

**设计思想**:
- 用方向项**放大** Huber 误差
- 金融解释："错方向的误差经济代价更高"

### 💻 代码实现：`losses.py:168-184`

```python
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
    用方向项放大 Huber 误差，突出"错方向的误差经济代价更高"。
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = (1.0 + lambda_dir * dir_term) * huber_term
    return _reduce(loss, reduction)
```

### ✅ 对齐情况

| 组件 | 文档 | 代码 | 状态 |
|-----|------|------|------|
| 方向项 | $(1 - \sigma(a \cdot y \cdot \hat{y})) \cdot \tilde{w}$ | `_normalized_direction_term` | ✅ 完全一致 |
| Huber 项 | $H_\delta(e)$ | `_huber_term` | ✅ 完全一致 |
| 组合方式 | $(1 + \lambda \cdot D) \times H$ | `(1 + λ_dir·dir) × hub` | ✅ 完全一致 |

### 🎯 特点

**行为分析**:

$$
\begin{align}
\text{方向对 } (D \approx 0): \quad L &\approx (1 + 0) \times H = H \\
\text{方向错 } (D \approx 1): \quad L &\approx (1 + \lambda) \times H = (1+\lambda) \cdot H \quad \text{# 误差被放大}
\end{align}
$$

**与加法型对比**:
- 加法型：方向和误差独立惩罚
- 乘法型：方向错误时误差被放大

**适用场景**:
- 极度强调"错方向的代价"
- 金融解释更直观

---

## 📊 总结对比表

| 损失函数 | 文档章节 | 核心特性 | 实现差异 | 推荐场景 |
|---------|---------|---------|---------|---------|
| **IMADL** | 3.2.1 | 方向+精度 | 归一化权重 | 修复 GMADL 缺陷 |
| **Directional Huber** | 3.2.2 | 方向+鲁棒 | 加法型 | 方向和精度并重 |
| **Hybrid Add** | Pre-plan 方案 A | 方向+精度+鲁棒 | 完全一致 | **主推方案** |
| **Hybrid Mul** | Pre-plan 方案 B | 方向放大误差 | 完全一致 | 强调方向代价 |

---

## 🎯 实现建议

### 1. 论文中如何说明

**对于 IMADL**:
> "We implement a rebalanced version of Improved MADL with normalized directional weights to prevent the directional signal from being overwhelmed by magnitude errors in small-return regimes."

**对于 Directional Huber**:
> "We adopt an additive formulation instead of the multiplicative form to ensure that magnitude errors are penalized even when the directional prediction is correct, providing a better balance between directional awareness and prediction accuracy."

### 2. 实验建议

**第一轮对比** (6 个损失函数):
1. MSE (baseline)
2. MedSE (robustness baseline)
3. GMADL (directional baseline)
4. IMADL (rebalanced)
5. Directional Huber (additive)
6. Hybrid Add (main contribution)

**第二轮扩展** (可选):
7. Hybrid Mul (alternative formulation)

### 3. 关键指标

除了常规指标 (MSE, $R^2$, Sharpe)，务必加入：
- **Directional Accuracy**: $\mathbb{1}[\text{sign}(y) = \text{sign}(\hat{y})]$ 的比例
- **Sign Mismatch on Large** $|y|$: 大波动月份的方向错误率

---

## 📚 参考文献映射

- **Section 3.2.1**: Improved MADL → `imadl_rebalanced_loss`
- **Section 3.2.2**: Directional Huber → `directional_huber_loss`
- **Pre-plan 方案 A**: Additive Hybrid → `hybrid_dir_huber_add_loss`
- **Pre-plan 方案 B**: Multiplicative Hybrid → `hybrid_dir_huber_mul_loss`

---

**分析完成** ✅
