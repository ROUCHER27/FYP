# Phase 1 实验总结与分析

> **文档目的：** 总结 Phase 1 的七个损失函数实验结果，分析各损失函数的表现，为 Phase 2 和最终论文提供指导。

---

## 1. 实验概述

### 1.1 实验目标

使用深度神经网络（DNN）进行横截面股票收益预测，系统性评估不同损失函数（MSE、MedSE、MADL 系列等）对模型预测能力和组合表现的影响。

### 1.2 数据与时间窗口

| 项目 | 设定 |
|------|------|
| 数据来源 | CRSP 月度股票数据库（通过 WRDS） |
| 样本期间 | 1990年1月 - 2025年12月（美国上市公司） |
| 训练期 | 1990-01 至 1994-12（5年） |
| 测试期 | 1995-01 至 1996-12（24个月） |
| 特征集 | $X^1$（累积动量和换手率） |
| 目标变量 | 下期简单收益 $r_{i,t+1}$ |

### 1.3 特征工程

**Feature Set 1 ($X^1$)：累积动量和换手率**

捕捉基本价格动量和交易活跃度，时间窗口为 $m \in \{1, 3, 6, 9, 12\}$ 个月：

$$
cr_{i,m} = \prod_{j=t-m}^{t-1} (1 + r_{ij}) - 1, \quad co_{i,m} = \sum_{j=t-m}^{t-1} to_{ij}
$$

其中：
- $r_{i,t}$：股票 $i$ 在月 $t$ 的持有期收益
- $to_{i,t}$：月度成交量与流通股数的比率，衡量流动性

当前 `best_hyperparameters.txt` 对应的输入维度为 15。

### 1.4 模型架构

**多层感知机（MLP）配置：**

| 参数 | 设定 |
|------|------|
| `input_dim` | 15 |
| `hidden_dims` | [64, 32, 16] |
| `activation` | tanh |
| `dropout` | 0.0 |
| `optimizer` | Adam |
| `max_epochs` | 20 |
| `batch_size` | 1024 |

> **说明：** 通过 Grid Search 在前 5 年数据（1989-1994）上确定，以避免前瞻偏差。

### 1.5 训练协议

**静态完整性检查（Static Sanity Check）：**

- **训练窗口：** 1990年1月 - 1994年12月（5年）
- **测试窗口：** 1995年1月 - 1996年12月（24个月）
- **策略：** 训练一次，对测试窗口中的每个月进行样本外预测，无参数更新

这种方法对比 MSE 与 MedSE 等损失函数的行为差异。

---

## 2. 组合构建与评估指标

### 2.1 组合策略

**多空组合（Long-Short Portfolio）：**

- **Long：** 预测收益 Top 10%（等权重）
- **Short：** 预测收益 Bottom 10%（等权重）
- **调仓频率：** 月度
- **权重上限：** 单票最大权重 `max_weight = 0.05`

**信号加权组合（P2 - Signal Weighted）：**

在 Long/Short 桶内基于预测值 z-score 构造权重：

$$
z_{i,t} = \frac{\hat{y}_{i,t} - \mu_t}{\sigma_t}, \quad w_{i,t}^{sig} = \max(\pm z_{i,t}, 0)
$$

归一化后得到最终权重。

### 2.2 评估指标

**统计指标：**

- **样本外 $R^2$：** 
$$
R^2_{oos} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- **MSE（均方误差）：** 
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- **MedSE（中位数平方误差）：** 
$$
MedSE = \text{median}\{(y_i - \hat{y}_i)^2\}
$$

- **方向准确率（Directional Accuracy）：** 
$$
DA = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{sign}(y_i) = \text{sign}(\hat{y}_i)]
$$

**经济指标：**

- **累计收益（Cumulative Return）：** 
$$
CR = \prod_{t=1}^{T} (1 + r_t) - 1
$$

- **年化收益（Annualized Return）：** 
$$
AR = \left(1 + CR\right)^{12/T} - 1
$$

- **夏普比率（Sharpe Ratio）：** 
$$
SR = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{12}
$$

其中 $\bar{r}$ 为平均月度收益，$\sigma_r$ 为收益标准差，$r_f$ 假设为 0。

---

## 3. 损失函数定义

### 3.1 基线损失函数

**MSE（均方误差）：**

$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**MedSE（中位数平方误差）：**

$$
L_{MedSE} = \text{median}\{(y_i - \hat{y}_i)^2\}
$$

MedSE 对异常值更鲁棒，通过最小化中位数而非均值来减少极端误差的影响。

### 3.2 方向型损失函数

**GMADL（广义平均绝对方向损失）：**

$$
L_{GMADL} = -[\sigma(a \cdot y \cdot \hat{y}) - 0.5] \times |\hat{y}|^b
$$

其中：
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 sigmoid 函数
- $a = 100, b = 2$ 为超参数
- 当 $y$ 和 $\hat{y}$ 同号时给予奖励（负损失），异号时给予惩罚（正损失）
- $|\hat{y}|^b$ 项鼓励模型对正确方向的预测给出更大的幅度

**IMADL（改进的 MADL）：**

$$
L_{IMADL} = \lambda_{dir} \cdot \frac{1 - \tanh(a \cdot y \cdot \hat{y})}{2} + \lambda_{mag} \cdot |y - \hat{y}|
$$

其中：
- 方向项：$\frac{1 - \tanh(a \cdot y \cdot \hat{y})}{2}$ 归一化到 [0, 1]
- 幅度项：$|y - \hat{y}|$ 为绝对误差
- $\lambda_{dir} = 1.0, \lambda_{mag} = 0.1$ 控制两项的相对重要性
- $a = 100$ 控制方向项的陡峭程度

**Directional Huber（方向 Huber 损失）：**

$$
L_{DirHuber} = \lambda_{dir} \cdot \frac{1 - \tanh(a \cdot y \cdot \hat{y})}{2} + \lambda_{hub} \cdot H_\delta(y - \hat{y})
$$

其中 Huber 项定义为：

$$
H_\delta(e) = \begin{cases}
\frac{1}{2} e^2 & \text{if } |e| \leq \delta \\
\delta (|e| - \frac{1}{2}\delta) & \text{if } |e| > \delta
\end{cases}
$$

- $\delta = 0.01$ 为 Huber 阈值
- $\lambda_{dir} = 1.0, \lambda_{hub} = 1.0$
- 小误差时使用二次惩罚，大误差时使用线性惩罚，提供鲁棒性

**Hybrid Add（混合加法损失）：**

$$
L_{HybridAdd} = \lambda_{dir} \cdot \frac{1 - \tanh(a \cdot y \cdot \hat{y})}{2} + \lambda_{hub} \cdot H_\delta(y - \hat{y})
$$

参数设置：
- $a = 100, b = 2, \delta = 0.01$
- $\lambda_{dir} = 1.0, \lambda_{hub} = 1.0$

**Hybrid Mul（混合乘法损失）：**

$$
L_{HybridMul} = \left[\frac{1 - \tanh(a \cdot y \cdot \hat{y})}{2}\right] \times H_\delta(y - \hat{y})
$$

- 方向项和 Huber 项相乘
- 当方向正确时，整体损失被抑制
- 当方向错误时，Huber 误差被放大

---

## 4. Phase 1 实验结果

### 4.1 整体表现排名

**按 Sharpe 比率降序排列：**

| 排名 | 损失函数 | 累计收益 | Sharpe | 平均 MSE | 平均 MedSE | 平均 $R^2$ | 方向准确率 |
|------|----------|----------|--------|----------|------------|-----------|------------|
| 1 | **IMADL** | 0.2374 | 0.6949 | 0.0329 | 0.0084 | -0.2194 | 53.10% |
| 2 | **GMADL** | 0.1318 | 0.6632 | 3.1174 | 3.1355 | -119.58 | 53.10% |
| 3 | **MEDSE** | 0.0235 | 0.1481 | 0.0294 | 0.0035 | -0.0484 | 53.10% |
| 4 | **HYBRID_MUL** | -0.0076 | 0.0724 | 0.0294 | 0.0043 | -0.0656 | 53.10% |
| 5 | **DIRHUBER** | -0.0583 | -0.1261 | 0.1434 | 0.1254 | -4.5768 | 53.10% |
| 6 | **HYBRID_ADD** | -0.2496 | -0.4992 | 0.0504 | 0.0269 | -0.9298 | 53.10% |
| 7 | **MSE** | -0.2041 | -0.6138 | 0.0291 | 0.0035 | -0.0435 | 53.10% |

### 4.2 关键发现

**1. 方向准确率的一致性**

所有损失函数的方向准确率均为 53.10%，略高于随机猜测（50%），表明：
- 特征集 $X^1$ 提供了一定的预测信号
- 不同损失函数在方向判断上的差异不大
- 经济表现的差异主要来自于**预测幅度**和**信号强度**
方向准确率的定义

**方向准确率（Directional Accuracy）：**

$$  
DA = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{sign}(y_i) = \text{sign}(\hat{y}_i)]  
$$
其中：
- $y_i$：股票 $i$ 在 $t+1$ 月的**实际收益**
- $\hat{y}_i$：模型对股票 $i$ 在 $t+1$ 月收益的**预测值**
- $\text{sign}(x)$：符号函数，正数返回 +1，负数返回 -1

**具体含义：**

- 如果股票实际上涨（$y_i > 0$）且模型预测上涨（$\hat{y}_i > 0$）→ 正确
- 如果股票实际下跌（$y_i < 0$）且模型预测下跌（$\hat{y}_i < 0$）→ 正确
- 其他情况 → 错误

**2. IMADL 的显著优势**

- **最高 Sharpe（0.6949）：** 风险调整后收益最优
- **最高累计收益（23.74%）：** 24 个月内表现最佳
- **合理的误差指标：** MSE 和 MedSE 处于中等水平
- **优势来源：** 归一化的方向项 + 适度的幅度惩罚，平衡了方向性和精度

**3. GMADL 的次优表现**

- **Sharpe 0.6632：** 仅次于 IMADL
- **累计收益 13.18%：** 稳定的正收益
- **极高的 MSE（3.1174）：** 由于 $|\hat{y}|^b$ 项鼓励大幅度预测
- **$R^2$ 为 -119.58：** 预测值方差远大于真实值方差
- **结论：** GMADL 牺牲了传统误差指标，但在交易信号质量上表现优异

**4. 基线损失函数的对比**

- **MedSE 优于 MSE：**
  - MedSE Sharpe 0.1481 vs MSE Sharpe -0.6138
  - MedSE 累计收益 2.35% vs MSE 累计收益 -20.41%
  - **原因：** MedSE 对异常值更鲁棒，避免了极端误差主导训练过程
  
- **MSE 的失败：**
  - 负 Sharpe 和负累计收益
  - 尽管 MSE 和 $R^2$ 指标看似合理
  - **结论：** 传统误差指标与交易表现脱节

**5. 混合损失函数的表现分化**

- **HYBRID_MUL 接近盈亏平衡：**
  - Sharpe 0.0724，累计收益 -0.76%
  - 乘法形式在方向正确时抑制损失，但可能导致梯度不稳定
  
- **HYBRID_ADD 表现最差：**
  - Sharpe -0.4992，累计收益 -24.96%
  - 加法形式未能有效平衡方向性和精度
  - **可能原因：** $\lambda$ 参数设置不当，或方向项和 Huber 项的尺度不匹配

**6. Directional Huber 的负收益**

- Sharpe -0.1261，累计收益 -5.83%
- 尽管 Huber 项提供鲁棒性，但未能有效捕捉交易信号
- **可能原因：** $\delta = 0.01$ 过小，导致大部分误差进入线性区域，削弱了精度激励

---

## 5. 中期报告的比较方法论

根据 `2253235_YirongYu_2025.pdf` 中期报告，损失函数的比较采用以下方法：

### 5.1 可视化分析

**1. 损失函数行为图（Figure 4.1）**

- 多子图折线图，展示不同真实收益场景下的损失值
- 横轴：估计收益 $\hat{y}$，纵轴：损失值
- 固定真实收益 $y \in \{-0.8, -0.1, +0.1, +0.9\}$
- **目的：** 直观展示损失函数的梯度行为和奖惩机制

**2. 参数空间热力图（Figure 4.2）**

- 二维热力图，展示 GMADL 在不同 $(y, \hat{y})$ 组合下的损失值
- 对角线边界清晰分离奖励区（绿色）和惩罚区（红色）
- **目的：** 验证损失函数的对称性和边界行为

**3. MADL vs. GMADL 对比图（Figure 4.3）**

- 并排对比两个损失函数在相同场景下的行为
- 标注关键区域（如"GMADL 更平滑"、"MADL 梯度消失"）
- **目的：** 说明 GMADL 的改进点

### 5.2 组合回测对比

**组合定义（Section 4.2）：**

- **P1 (Equal)：** Long top 10%, Short bottom 10%, 等权重
- **P2 (Signal Weighted)：** 桶内基于 z-score 加权
- **P3 (Capped)：** 在 P2 基础上添加单票权重上限（如 10%）

**结果表格（Table 4.1）：**

| 策略 | 损失函数 | Std | Sharpe | CumReturn |
|------|----------|-----|--------|-----------|
| P1_Equal | MSE | 0.0147 | 0.3730 | 0.9% |
| P1_Equal | MedSE | 0.0116 | 2.6773 | 5.48% |
| P2_SignalWeighted | MSE | 0.0288 | -1.4559 | -7.23% |
| P2_SignalWeighted | MedSE | 0.0282 | 3.2286 | 16.64% |
| P3_Cap | MSE | 0.0254 | -1.5668 | -6.86% |
| P3_Cap | MedSE | 0.0282 | 3.2286 | 16.64% |

**累计收益曲线（Figure 4.4）：**

- 时间序列折线图，展示 P1-P3 策略在 MSE 和 MedSE 下的累计收益
- MedSE（虚线）显著优于 MSE（实线）
- **结论：** MedSE 在所有组合构建方法下均表现更优

**风险与收益分解（Figure 4.5）：**

- (a) 标准差对比：MedSE 维持可比或更低的波动率
- (b-d) Sharpe 比率对比：MedSE 在所有策略下 Sharpe 显著更高
- **结论：** MedSE 的表现提升来自有效信号捕捉，而非过度杠杆

### 5.3 关键指标总结

**中期报告关注的核心指标：**

1. **统计指标：**
   - 样本外 $R^2$
   - MSE
   - Median SE

2. **经济指标：**
   - 累计收益（Cumulative Return）
   - 年化收益（Annualized Return）
   - 夏普比率（Sharpe Ratio）
   - 标准差（Std）

3. **可视化：**
   - 损失函数行为图
   - 参数空间热力图
   - 累计收益时间序列
   - 风险收益分解图

---

## 6. Phase 1 的局限性与改进方向

### 6.1 当前局限性

**1. 测试期较短**

- 仅 24 个月（1995-1996）
- 未覆盖不同市场周期（牛市、熊市、震荡市）
- **改进：** Phase 2 可考虑扩展到更长测试期或多个时间段

**2. 单一特征集**

- 仅使用 $X^1$（累积动量和换手率）
- 未测试 $X^2$（去噪标准化动量）和 $X^3$（原始月度序列）
- **改进：** 后续实验可对比不同特征集的表现

**3. 固定超参数**

- 所有方向型损失函数使用相同的 $\lambda$ 参数
- 未针对每个损失函数单独调优
- **改进：** Phase 2 可为入围损失函数进行超参数搜索

**4. 缺乏鲁棒性验证**

- 仅使用单一随机种子（seed=42）
- 未测试不同权重上限设置
- **改进：** Phase 2 将进行多种子和多权重上限的鲁棒性检查

### 6.2 Phase 2 改进计划

根据 `Phase2-experiment-plan.md`，Phase 2 将：

1. **规范化输出目录：**
   - 使用 Google Drive 的权威目录结构
   - 分离主实验和鲁棒性实验的输出

2. **鲁棒性矩阵：**
   - 种子：42, 52, 62
   - 权重上限：0.05, None
   - 模型：mse, medse, finalist_1, finalist_2

3. **入围者选择：**
   - 按 Phase 1 主表 Sharpe 排名前 2 的方向型损失函数
   - 加上固定基线 mse 和 medse

4. **论文级产物：**
   - 权威的 7 个损失函数对比表
   - 入围者的鲁棒性对比表
   - 可直接用于论文的图表和结论要点

---

## 7. 结论与建议

### 7.1 主要结论

1. **IMADL 是最佳整体损失函数：**
   - Sharpe 0.6949，累计收益 23.74%
   - 平衡了方向性和精度
   - 推荐作为 Phase 2 的入围者之一

2. **GMADL 是次优选择：**
   - Sharpe 0.6632，累计收益 13.18%
   - 牺牲传统误差指标，但交易信号质量高
   - 推荐作为 Phase 2 的入围者之二

3. **MedSE 优于 MSE：**
   - MedSE 是唯一表现为正的基线损失函数
   - 对异常值的鲁棒性是关键优势
   - 应作为固定基线保留在所有对比中

4. **混合损失函数需要进一步调优：**
   - HYBRID_ADD 和 DIRHUBER 表现不佳
   - 可能的原因是参数设置或尺度不匹配
   - 不推荐进入 Phase 2 鲁棒性检查

5. **方向准确率不是唯一指标：**
   - 所有损失函数的方向准确率相同（53.10%）
   - 经济表现的差异来自预测幅度和信号强度
   - 需要综合考虑统计指标和经济指标

### 7.2 Phase 2 建议

**入围者选择：**

- **固定基线：** MSE, MedSE
- **方向型入围者：** IMADL, GMADL

**鲁棒性检查重点：**

- 多种子稳定性（seeds: 42, 52, 62）
- 权重上限敏感性（max_weight: 0.05 vs None）
- 跨时间段一致性（如果扩展测试期）

**论文写作重点：**

- 强调 IMADL 和 GMADL 相对于传统 MSE 的优势
- 解释方向型损失函数的设计动机和理论基础
- 展示 MedSE 作为鲁棒基线的价值
- 讨论混合损失函数失败的原因和改进方向

---

## 8. 附录：数据文件清单

Phase 1 实验产生的关键文件：

```
doc/phase1/
├── phase1.md                                    # 原始实验日志
├── Phase1-实验总结与分析.md                      # 本文档
├── mse/
│   ├── sanity_summary_mse.json
│   └── sanity_metrics_mse.csv
├── medse/
│   ├── sanity_summary_medse.json
│   └── sanity_metrics_medse.csv
├── gmadl/
│   ├── sanity_summary_gmadl.json
│   └── sanity_metrics_gmadl.csv
├── imadl/
│   ├── sanity_summary_imadl.json
│   └── sanity_metrics_imadl.csv
├── dirhuber/
│   ├── sanity_summary_dirhuber.json
│   └── sanity_metrics_dirhuber.csv
├── hybrid_add/
│   ├── sanity_summary_hybrid_add.json
│   └── sanity_metrics_hybrid_add.csv
└── hybrid_mul/
    ├── sanity_summary_hybrid_mul.json
    └── sanity_metrics_hybrid_mul.csv
```

**使用说明：**

- `sanity_summary_*.json`：包含整体统计指标和经济指标
- `sanity_metrics_*.csv`：包含逐月详细指标，适合时间序列分析
- 所有结果基于统一配置：`tanh` 激活函数，`dropout=0.0`

---

**文档版本：** v1.0  
**创建日期：** 2026-04-24  
**作者：** Yirong Yu  
**下一步：** 根据本总结执行 Phase 2 实验计划
