# Phase 1.5-2.5 阶段性研究总结与导师汇报提纲

**目的**：完整总结 Phase 1.5 到 Phase 2.5 的研究进度、已完成工作、实验结果、当前结论与后续风险点，方便向导师汇报当前 Final Year Project 的推进情况。

**当前主线**：本项目从“传统预测误差最小化”转向“面向交易目标的损失函数设计”。Semester 1 的 interim report 已经证明，损失函数不是单纯的数学误差指标，而是会通过预测信号、组合权重和收益曲线传导到最终 Sharpe、累计收益和风险暴露。Phase 1.5-2.5 的工作就是在这个基础上，系统地设计、筛选和诊断更贴近交易目标的 hybrid loss。

---

## 0. 一页版结论

### 0.1 已经完成了什么

从 Phase 1.5 到 Phase 2.5，我已经完成了四类工作：

1. **参数调优与鲁棒性验证**：Phase 1.5 对 hybrid add / hybrid mul 的方向惩罚参数进行了系统搜索，并发现单一 seed 下的高 Sharpe 结果存在明显 seed sensitivity。
2. **新 loss 族设计与筛选**：Phase 2 设计并测试了四类候选方向，包括 IMADL+M2 linear blend、IMADL+GMADL weighted blend、M2+robustness penalty、adaptive hybrid。
3. **Phase 2.2 深入精调**：在 Phase 2 P0 结果基础上，对 M2+robustness 的 gamma 参数进行 refinement，并对 top losses 做 loss-scale diagnostics 和 normalized variants 验证。
4. **Phase 2.5 对齐诊断**：系统检查 Phase 1.5 与 Phase 2 runner 之间的配置、`lambda_dir`、seed、data preprocessing、test window、model architecture 和 loss implementation 差异，明确哪些结论能直接报告，哪些需要谨慎表述。

### 0.2 当前最重要的实验结论

**Phase 2 内部比较中，当前最有汇报价值的候选是 `m2_robust_gamma07`。**

根据 Phase 2.2/phase2-fix 结果：

| 候选 loss | Mean Sharpe | CV | 角色 |
| --- | ---: | ---: | --- |
| `m2_robust_gamma07` | 0.9156 | 0.1808 | 当前主推荐，Sharpe 与稳定性最均衡 |
| `m2_robust_gamma10` | 1.0043 | 0.5613 | Sharpe 最高，但 seed 波动明显更大 |
| `imadl_m2_alpha06` | 0.6895 | 0.2443 | 稳定 fallback / 对照候选 |

Phase 2 P0 也已经显示：

- IMADL+M2 linear blend 中，`alpha06` 是最稳的候选。
- IMADL+GMADL blend 整体失败，Sharpe 接近 0 或为负，R2 极差。
- Adaptive hybrid 不够稳定，`adaptive_lambda10` 接近 Sharpe 0.5 但 CV 超过 1.5。
- M2+robustness 是目前最值得写入论文主线的改进方向。

### 0.3 当前最重要的诊断结论

Phase 2.5 的对齐诊断把之前的“alignment failure”拆开了：

1. **M2 对齐问题不是 runner regression，而是参数口径错误。**  
   Phase 1.5 M2 使用 `lambda_dir=5.0`，Phase 2.1b 的 `hybrid_mul` 使用默认 `lambda_dir=1.0`，Phase 2 新 M2 族多使用 `lambda_dir=2.0`。这三者是不同 loss variants，不能直接比较。

2. **IMADL 的 +17.6% 偏差不是简单性能提升，而是命名碰撞。**  
   Phase 1.5 的 IMADL 是 simple tanh-based `madl_loss()`；Phase 2 当前的 `imadl` 指向 `imadl_rebalanced_loss()`，加入了 normalized directional term 和 magnitude term。两者公式不同，因此不能说“Phase 2 IMADL 提升了 Phase 1.5 IMADL”。

3. **GMADL 的 +62.4% 偏差不再视为阻塞性 bug。**  
   完整诊断报告认为 GMADL implementation 逐字节一致，data preprocessing、test window、architecture、training hyperparameters、seed initialization 等因素也已排除；剩余最合理解释是 PyTorch/CUDA/数值精度、实际 batch/training trajectory 或环境差异。这一项证据强度弱于 M2 和 IMADL，因为它是“部分解释”而不是公式级 mismatch，但足以支持“无需停止 Phase 2 或重跑全部实验”的结论。

4. **Loss-scale diagnostics 发现分量量级严重不平衡，但 normalization 不是通用解。**  
   Proxy diagnostics 显示代表 loss 的 scale ratio 大于 30x，触发了 normalization 检查；authored analysis report 显示 normalization 只改善 `gamma07`（+5.96%），但降低 `gamma10` 和 `imadl_m2_alpha06`。因此这应被表述为“normalization 是 gamma07 的可选增强，不是所有 loss 的通用修复”。

### 0.4 向导师汇报时可以安全宣称的内容

可以安全说：

- 我已经完成从 Phase 1.5 到 Phase 2.5 的系统实验链条：参数调优、multi-seed robustness、四类 hybrid loss 设计、gamma refinement、loss-scale diagnostics 和 alignment investigation。
- Phase 2 内部比较中，`m2_robust_gamma07` 是当前最稳妥的主候选，`gamma10` 是高 Sharpe 但高波动备选，`imadl_m2_alpha06` 是稳定 fallback。
- 目前结论更支持“robustness-enhanced M2 variant”这条研究路线，而不支持 IMADL+GMADL blend 或 adaptive hybrid。
- Phase 2.5 发现了关键实验口径问题，尤其是 `lambda_dir` 和 IMADL 命名碰撞；完整诊断报告进一步判断对齐失败不是代码 bug，不需要重跑实验，Phase 2 内部比较仍然有效。

暂时不要说：

- 不要说 “Phase 2 M2 直接超越 Phase 1.5 M2”，除非用相同 `lambda_dir` 和同一 runner 重新跑。
- 不要说 “IMADL alignment 证明 Phase 2 IMADL 更好”，因为当前 IMADL 名称对应的公式不同。
- 不要说 “Phase 2 runner 已严格数值复现 Phase 1.5”，因为 Phase 2 的目标和实际配置是探索新的 loss variants，而不是 exact replication。
- 不要说 “已经证明长期可交易盈利”或 “production-ready trading system”。当前结果仍是 controlled/static sanity-check setup 下的 empirical evidence，尚未覆盖交易成本、更多市场 regime 和完整 rolling deployment。
- 不要直接使用 thesis draft 中与 Phase 2.5 diagnostics 冲突的配置说法，例如 LSTM/6-month/seeds 42,123,456 等，需要先统一为当前 repo 证据口径。

---

## 1. 研究背景与总目标

Interim report 的核心观点是：在 neural-network-based equity strategy 中，loss function 不只是预测误差函数，而是训练阶段的 trading objective proxy。不同 loss 会改变模型认为“好预测”的标准，最终影响预测信号、排序质量、组合权重、累计收益和 Sharpe ratio。

Semester 1 已经完成两件关键事情：

1. **复现并分析 MADL/GMADL**  
   MADL/GMADL 把方向正确性引入 loss，试图让模型更关注交易方向，而不是只关注点预测误差。GMADL 使用 sigmoid 平滑方向奖励/惩罚，但也暴露出一些问题：例如正确方向下可能缺乏精确幅度激励、关键区域信号较弱，以及 reward/penalty 结构可能不够符合标准 penalty loss 的习惯。

2. **比较 MSE 与 MedSE**  
   Interim report 发现，MedSE 相比 MSE 在多种 portfolio construction 下具有更高 Sharpe 和更好的累计收益，说明金融收益预测中 outliers 很重要，robustness 是 loss function 设计必须考虑的属性。

因此，Phase 1.5-2.5 的目标可以概括为：

> 在 MADL/GMADL 和 MedSE 的基础上，设计更适合月度横截面股票策略的 hybrid loss，使它同时关注方向、幅度、鲁棒性和最终组合表现，并通过多 seed、多参数和对齐诊断验证结果是否可靠。

---

## 2. Phase 1.5：Hybrid 参数调优与鲁棒性暴露

### 2.1 研究问题

Phase 1 中原始 hybrid add / hybrid mul 表现不理想：

- 原始 Hybrid Add：`lambda_dir=1.0, lambda_hub=1.0`，Sharpe -0.4992。
- 原始 Hybrid Mul：`lambda_dir=1.0`，Sharpe 0.0724。

Phase 1.5 的问题是：这些 hybrid loss 是公式本身失败，还是参数设置不合适？

### 2.2 实验设置

Phase 1.5 主要测试两类 hybrid loss：

- **Hybrid Add**：方向项 + Huber/magnitude 项，调节 `lambda_dir` 与 `lambda_hub`。
- **Hybrid Mul**：用方向项乘法放大 Huber loss，调节 `lambda_dir`。

关键配置：

- 训练期：1990-01 到 1994-12。
- 测试期：1995-01 到 1996-12，24 个月。
- 模型：MLP `[64, 32, 16]`，tanh，dropout 0.0。
- 特征：X1 cumulative return / turnover feature set。

### 2.3 单 seed lambda sweep 发现

Phase 1.5 单 seed sweep 中，表现最好的三个候选是：

| 变体 | Loss | 参数 | Sharpe | 累计收益 | 解释 |
| --- | --- | --- | ---: | ---: | --- |
| M1 | hybrid_mul | `lambda_dir=2.0` | 1.6295 | 65.33% | 中等方向惩罚，单 seed 最高 |
| A4 | hybrid_add | `lambda_dir=5.0, lambda_hub=0.1` | 1.4518 | 42.33% | 强方向 + 弱幅度 |
| M2 | hybrid_mul | `lambda_dir=5.0` | 1.0316 | 54.28% | 更强方向惩罚，高收益但可能更不稳 |

重要发现：

- Hybrid Add 的关键不是单纯提高方向项，而是**降低 magnitude/Huber 项权重**，否则幅度项会压过方向信号。
- Hybrid Mul 的关键是 `lambda_dir` 不能太弱；`0.1` 到 `0.5` 表现差，`2.0` 到 `5.0` 显著改善。
- 方向准确率在不同 loss 下几乎一致，说明收益差异主要来自排序质量、权重分布和信号强度，而不是简单的 direction accuracy 提升。

### 2.4 Multi-seed robustness 发现

后续鲁棒性测试暴露了更重要的问题：单 seed 高 Sharpe 不等于稳健。

| Loss | Avg Sharpe | Std Dev | 结论 |
| --- | ---: | ---: | --- |
| M2 / `hybrid_mul_m2` | 0.914 | 1.276 | 平均最高，但 seed variance 极大 |
| MedSE | 0.644 | 1.288 | robustness baseline 仍高波动 |
| IMADL | 0.464 | 0.414 | 更稳健的 directional baseline |
| M1 / `hybrid_mul_m1` | 0.410 | 0.973 | failure rate 高 |
| GMADL | 0.307 | 0.358 | 中等但表现下降 |

Phase 1.5 的真正结论不是“M2 已经可以直接作为最终方案”，而是：

> 参数调优能让 hybrid loss 在单 seed 下大幅提升，但高方向惩罚会放大 seed sensitivity。因此下一阶段必须把“高 Sharpe”与“跨 seed 稳定性”一起优化。

特别是 M2 在 cap=0.05 的 3 seeds 下范围从 -0.239 到 2.285，均值 0.914 但标准差 1.276；这说明它有强潜力，但单独使用风险太高。这个结果直接推动 Phase 2 转向 robustness-enhanced M2，而不是简单继续放大 `lambda_dir`。

这直接推动了 Phase 2 的设计：不是继续盲目提高 `lambda_dir`，而是设计新的 hybrid loss，让方向信号、幅度误差和 robustness 更平衡。

---

## 3. Phase 2：四类 Hybrid Loss 设计与 P0 筛选

### 3.1 设计目标

Phase 2 的目标是把 Phase 1.5 暴露的问题转化为新的 loss design：

- 保留 M2 的高收益潜力；
- 引入 IMADL 的稳定性；
- 加入 MedSE/Huber 启发的 robustness；
- 避免 GMADL 单独使用时的弱信号和幅度不足问题；
- 用 multi-seed 筛选，而不是只看 seed=42。

### 3.2 四类候选

Phase 2 设计并测试了四类 variants：

1. **Variant 1：IMADL + M2 Linear Combination**  
   用 alpha 控制 IMADL 与 M2 的权重，目标是在 IMADL 稳定性和 M2 高收益之间找到平衡。

2. **Variant 2：IMADL + GMADL Weighted Combination**  
   尝试将两个 directional loss 结合，观察 GMADL 是否能通过 blending 被补救。

3. **Variant 3：M2 + Robustness Enhancement**  
   在 M2 基础上加入 robustness penalty，用 gamma 控制 robustness 强度。这是当前结果最好的方向。

4. **Variant 4：Adaptive Hybrid**  
   尝试动态调整 loss component 权重，但实验显示不稳定。

### 3.3 Phase 2 P0 结果

Phase 2 P0 完成了 16 losses x 3 seeds = 48 runs。Top results：

| Rank | Loss | Mean Sharpe | CV | Mean Cumulative Return | 解读 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `m2_robust_gamma10` | 1.0043 | 0.5613 | 0.2368 | 最高 Sharpe，但波动较大 |
| 2 | `m2_robust_gamma01` | 0.7470 | 0.5270 | 0.2718 | 稳定性略好，Sharpe 较低 |
| 3 | `m2_robust_gamma001` | 0.6919 | 1.1936 | 0.1705 | 平均不错但 CV 高 |
| 4 | `imadl_m2_alpha06` | 0.6895 | 0.2443 | 0.3042 | 最稳的高质量候选 |
| 5 | `imadl_m2_alpha05` | 0.5822 | 0.5484 | 0.2465 | alpha06 附近的敏感性点 |

P0 结论：

- **Variant 3 是主胜者**：`m2_robust_gamma10` 的 Sharpe 最高，但 CV 较高；说明 M2+robustness 是值得继续精调的方向。
- **Variant 1 有稳定 fallback**：`imadl_m2_alpha06` 虽然 Sharpe 低于 robust M2，但 CV 明显较低，累计收益也高。
- **Variant 2 失败**：IMADL+GMADL 的所有 beta variants Sharpe 接近 0 或负值，R2 极差，说明 GMADL 的问题不能靠简单 blending 修复。
- **Variant 4 不成熟**：adaptive_lambda10 接近 Sharpe 0.5，但 CV 超过 1.5，不适合作为主线。

### 3.4 Phase 2 P0 后的 shortlist

P0 后 shortlist 冻结为：

1. `m2_robust_gamma10`：最高 Sharpe，适合作为性能上限候选。
2. `m2_robust_gamma01`：相对稳健的 robust M2 早期候选。
3. `imadl_m2_alpha06`：稳定 hybrid fallback。

这为 Phase 2.2 的 gamma refinement 和 diagnostics 提供了明确方向。

---

## 4. Phase 2.1b：Runner Alignment 与口径检查

### 4.1 为什么要做 alignment

Phase 2 的新实验建立在 Phase 1.5 结果之上。如果 Phase 2 runner 无法复现 Phase 1.5 的基础 loss，那么后续对新 loss 的解释会有风险。因此 Phase 2.1b 尝试用 Phase 2 runner 对齐 Phase 1.5 baseline。

初始结果显示：

| Loss | Phase 1.5 Target | Phase 2.1b Observed | Deviation |
| --- | ---: | ---: | ---: |
| IMADL | 0.464 | 0.546 | +17.64% |
| GMADL | 0.307 | 0.499 | +62.38% |
| M2 / hybrid_mul | 0.914 | 0.657 | -28.13% |

这看起来像 alignment failure，但 Phase 2.5 后续诊断表明不能简单地把它解释为 runner bug。

### 4.2 M2 alignment 的真实原因：`lambda_dir` 口径不一致

Phase 2.5 已经确认：

- Phase 1.5 M2：`hybrid_mul_m2`，`lambda_dir=5.0`。
- Phase 2.1b `hybrid_mul`：默认 `lambda_dir=1.0`。
- Phase 2 robust M2 family：基于新的 M2 variant，多数使用 `lambda_dir=2.0`。

所以 Phase 2.1b 实际上把 `lambda_dir=1.0` 与 Phase 1.5 的 `lambda_dir=5.0` 进行比较，这是不同 loss functions 之间的比较。

正确表述：

> M2 的 -28.1% deviation 不是 runner regression，而是 parameter mismatch。Phase 2.2 的 M2 robust results 是新 M2 variant 的内部比较，不能直接声称相对 Phase 1.5 M2 提升。

### 4.3 IMADL alignment 的真实原因：命名碰撞

Phase 2.5 的 `07_loss_implementation_details.md` 进一步发现，IMADL 的名字在不同阶段代表了不同公式：

| 阶段 | 名称 | 实际函数 | 特征 |
| --- | --- | --- | --- |
| Phase 1.5 | IMADL | `madl_loss()` / tanh-based directional loss | 简单方向 reward/penalty，无 magnitude term |
| Phase 2 | IMADL | `imadl_rebalanced_loss()` | normalized directional term + magnitude error |

因此 IMADL 的 +17.6% deviation 不能直接解释为提升，也不能解释为 bug。它本质上是两个不同公式共享同一个名字。

正确表述：

> Phase 2 introduced a rebalanced IMADL variant. It should be named and documented separately, rather than being treated as the same IMADL as Phase 1.5.

### 4.4 GMADL alignment：非阻塞的环境/数值差异

GMADL implementation 被记录为相同，但 observed mean 从 0.307 到 0.499，偏差 +62.38%。完整诊断报告的判断是：这不是需要修复的代码 bug，更可能是环境/数值/训练轨迹差异造成的统计波动或运行口径差异。现有检查已经排除了很多常见解释：

- Seed values 和 seed initialization：相同或 Phase 2 更 deterministic。
- Data preprocessing：同一 pipeline、同一 X1 feature set、同一 target construction。
- Test window：两者都使用 1995-01 到 1996-12 的 24-month window。
- Model architecture 和 hyperparameters：同一 MLP config、同一 training loop、同一 optimizer defaults。

剩余可能原因：

- PyTorch / NumPy / Pandas / CUDA 或 MPS 环境差异；
- 细微数值精度差异；
- batch order 或训练轨迹差异；
- evaluation pipeline 中尚未发现的口径差异；
- Phase 1.5 target 数字来源可能与当前 comparison table 不完全同口径。

汇报时应说：

> GMADL alignment 不再是阻塞 Phase 2 结论的 bug。它没有像 M2/IMADL 那样被公式级差异完全解释，但主要配置与实现都已经排除，完整诊断报告判断剩余差异可归因于环境/数值/训练轨迹因素。若导师要求 strict reproducibility，可以再做 GMADL minimal reproduction；否则不需要因此重跑 Phase 2。

---

## 5. Phase 2.2：Gamma Refinement 与 Loss-scale Diagnostics

### 5.1 Gamma refinement 的目的

Phase 2 P0 显示 `m2_robust_gamma10` 最高 Sharpe，但 CV 也高。Phase 2.2 的目标是探索 gamma 是否存在更好的 Sharpe-CV trade-off。

测试思路：

- 在 robust M2 family 内部做 gamma refinement。
- 观察 Sharpe 是否随 gamma 单调上升，还是存在峰值。
- 优先寻找稳定性更好的主推荐，而不是只追最高 mean Sharpe。

### 5.2 Gamma refinement 结果

核心结果：

| Loss | Mean Sharpe | Std | CV | Mean Cum Return |
| --- | ---: | ---: | ---: | ---: |
| `m2_robust_gamma03` | 0.3234 | 0.3418 | 1.0570 | 0.0818 |
| `m2_robust_gamma05` | 0.7054 | 0.1488 | 0.2109 | 0.2392 |
| `m2_robust_gamma07` | 0.9156 | 0.1655 | 0.1808 | 0.2799 |
| `m2_robust_gamma10` | 1.0043 | 0.5638 | 0.5613 | 0.2368 |
| `m2_robust_gamma15` | 0.8163 | 0.3724 | 0.4562 | 0.2277 |

解释：

- Sharpe 从 gamma=0.3 到 1.0 上升，gamma=1.5 回落，说明存在峰值区间。
- `gamma10` 平均 Sharpe 最高，但 seed 间波动较大。
- `gamma07` 的 mean Sharpe 接近 0.92，CV 只有 0.18，累计收益也优于 gamma10。

当前推荐：

> 如果论文主线强调稳定、可复现和风险调整收益，应把 `m2_robust_gamma07` 作为主推荐；如果强调 performance upper bound，可以把 `gamma10` 作为高 Sharpe 备选。

### 5.3 Loss-scale diagnostics 结果

Phase 2.2 diagnostics 对代表 loss 做了分量量级估计：

| Loss | Scale Ratio | Largest Component | Smallest Component | 判断 |
| --- | ---: | --- | --- | --- |
| `imadl_m2_alpha06` | 34.36x | `r2` | `medse` | severe imbalance |
| `m2_robust_gamma01` | 113.07x | `directional_accuracy` | `medse` | severe imbalance |
| `m2_robust_gamma10` | 113.69x | `directional_accuracy` | `medse` | severe imbalance |

需要谨慎解释：

- 当前 diagnostics 是 `metric_proxy`，不是训练时真实 loss components 的逐 batch logger。
- 它足够说明 component scale 存在显著差异，但不能直接证明反向传播中某一项被完全淹没。
- 根据 Phase 2.2 criteria，>10x 会触发 normalization 检查，因此后续做 normalized variants 是合理的。

---

## 6. Phase 2.2-fix1：Normalization Experiment

### 6.1 为什么要做 normalization

Loss-scale diagnostics 显示代表 loss 的 component scale ratio 超过 30x，最严重超过 100x。按照 criteria，这属于 severe imbalance，需要验证 normalized variants 是否能改善结果。

### 6.2 Normalization 结果

根据 `LOSS_COMPONENT_ANALYSIS_RESULTS.md` 的 supervisor-facing analysis：

| Loss | Original Sharpe | Normalized Sharpe 平均 | 结论 |
| --- | ---: | ---: | --- |
| `m2_robust_gamma07` | 0.9156 | 0.9702 | 提升 +5.96%，可作为备选增强 |
| `m2_robust_gamma10` | 1.0043 | 0.9420 | 下降 -6.20%，不推荐 normalized |
| `imadl_m2_alpha06` | 0.6895 | 0.5337 | 下降 -22.59%，不推荐 normalized |

### 6.3 解释

Normalization experiment 给出的结论不是“所有 loss 都应该归一化”，而是更细：

- 对 `gamma07`，归一化带来 +5.96% Sharpe 提升，说明该配置中的 robustness term 可能确实受到 scale imbalance 抑制。
- 对 `gamma10`，归一化反而降低表现，说明更高 gamma 可能已经部分补偿了 scale imbalance，机械 normalization 会破坏已有平衡。
- 对 `imadl_m2_alpha06`，归一化明显破坏原本稳定的组合关系，说明 IMADL 与 M2 的自然比例可能是其稳定性的来源之一。

本地结果目录里存在重复 summary 文件，部分 JSON copy 与 authored analysis report 的 normalized 平均值不一致。向导师汇报时建议以 `LOSS_COMPONENT_ANALYSIS_RESULTS.md` 作为当前人工整理口径，同时在正式论文前回到原始 per-seed CSV/summary 做一次最终核对。

汇报时可以说：

> 我没有只根据 scale ratio 直接改公式，而是做了 normalized contrast。结果显示，scale imbalance 并不自动等于 bug；至少在当前 experiments 中，原始 `m2_robust_gamma07` 仍是更稳妥选择。

---

## 7. Phase 2.5：当前诊断进度与收敛结论

Phase 2.5 的目标不是再盲目扩大实验，而是解释 Phase 2.1b alignment 中的偏差，并清理论文叙事中的口径风险。

### 7.1 已排除的因素

| 检查项 | 结论 | 对 alignment 偏差的解释力 |
| --- | --- | --- |
| Test window | Phase 1.5 与 Phase 2.1b 都是 24 months | 不能解释 |
| Data preprocessing | 同一 `prepare_panel_data()`、同一 feature pipeline | 不能解释 |
| Feature set | 同一 X1 features | 不能解释 |
| Train/test split | 同一 period-based split | 不能解释 |
| Model architecture | 同一 MLP `[64,32,16]`, tanh, dropout 0 | 不能解释 |
| Training loop | `sanity_check_core.py` 记录为一致 | 不能解释 |
| Seed setting | seeds 与 `set_seed()` 一致，Phase 2 甚至更 deterministic | 不能单独解释 |

### 7.2 已确认的问题

#### 7.2.1 M2 参数命名问题

M2 的 `lambda_dir` 必须显式写在 loss name 或文档中：

- `m2_lambda1`：默认 hybrid_mul，`lambda_dir=1.0`。
- `m2_lambda2`：Phase 2 新 M2 family 基础。
- `m2_lambda5`：Phase 1.5 M2 baseline。

后续所有表格应避免只写 “M2”。

#### 7.2.2 IMADL 名称碰撞

应区分：

- Phase 1.5 original IMADL / MADL-style directional loss；
- Phase 2 rebalanced IMADL / normalized directional + magnitude loss。

后续 thesis 中应把 Phase 2 的版本命名为 `imadl_rebalanced` 或 `imadl_v2`，避免误导导师以为是同一个 loss。

### 7.3 尚未完全解决的问题

GMADL 的 +62.4% deviation 不再作为阻塞性 open issue。它仍然不是像 M2/IMADL 那样的公式级闭环解释，但完整诊断报告已将它收敛为环境/数值/训练轨迹差异，而不是代码 bug。因此它不影响 Phase 2 内部 comparison 的有效性，也不要求重跑 Phase 2；它只限制“Phase 2 runner 与 Phase 1.5 严格数值复现”这种强声明。

如果导师特别要求 strict reproducibility，可做后续最小闭环实验：

1. 固定同一环境和同一 commit；
2. 明确使用 Phase 1.5 GMADL exact implementation；
3. 固定 train/test window、seed、cap、data snapshot；
4. 输出 first batch checksum、initial model checksum、first epoch loss、final predictions checksum；
5. 比较 Phase 1.5 target 与 Phase 2 rerun 的差异来源。

---

## 8. 当前论文/汇报叙事建议

### 8.1 建议主线

建议向导师用以下逻辑汇报：

1. **Semester 1**：我发现 loss function 会显著影响 trading performance，不只是预测误差；MedSE 的表现证明 robustness 很重要。
2. **Phase 1.5**：我系统调参 hybrid losses，发现方向性惩罚确实能提高 Sharpe，但也会带来 seed sensitivity。
3. **Phase 2**：我基于这个问题设计了四类新的 hybrid losses，并通过 48 runs 做 P0 screening。
4. **Phase 2.2**：我对最有前景的 M2+robustness 做 gamma refinement，找到 `gamma=0.7` 这个更平衡的候选。
5. **Phase 2.5**：我没有直接把结果写成“提升”，而是做了对齐诊断，发现 `lambda_dir` 和 IMADL 命名是关键口径问题，因此现在能更准确地区分“真实贡献”和“不可直接比较的结果”。

### 8.2 论文核心贡献可以这样表述

当前阶段可形成四个贡献：

1. **实验贡献**：完成了从 baseline、lambda sweep、multi-seed robustness 到 hybrid variant screening 的系统实验流程。
2. **方法贡献**：提出并验证了 M2 + robustness penalty 的 loss family，其中 `m2_robust_gamma07` 在 Phase 2 内部比较中取得最优稳定性-收益平衡。
3. **负结果贡献**：证明 IMADL+GMADL blending 和 adaptive hybrid 当前不可行，避免把计算预算浪费在不稳定方向上。
4. **方法论贡献**：通过 Phase 2.5 diagnostics 发现并修正了 loss naming 与 parameter alignment 风险，强调同口径比较的重要性。

### 8.2.1 根据完整诊断报告新增的写作口径

建议在 Chapter 5 或 Chapter 6 加一段 cross-phase comparison note：

> Phase 2 should be interpreted as exploration of new directional-robust loss variants rather than exact replication of Phase 1.5 baselines. The apparent Phase 2.1b alignment failure is explained by different loss definitions: M2 variants use different `lambda_dir` values, Phase 2 IMADL is a rebalanced IMADL v2 rather than the original tanh-based IMADL, and GMADL differences are attributable to environment/numerical/training-trajectory variation. Therefore, direct numerical comparison between Phase 1.5 and Phase 2 is not meaningful, while internal comparisons within Phase 2 remain valid.

中文汇报可说：

> Phase 2 的定位不是复刻 Phase 1.5，而是探索新的 loss design space。所谓对齐失败主要来自比较对象不同，不是代码 bug。因此不需要重跑 Phase 2，但论文中必须明确 Phase 2 是新的变体探索，不能把不同 loss 当成同一个 baseline 直接比较。

### 8.3 当前最终候选排序

按“论文可解释性 + 稳定性 + 性能”综合排序：

1. **主推荐：`m2_robust_gamma07`**  
   解释：在 robust M2 family 内部，Sharpe 接近 0.92，CV 显著低于 gamma10，是当前最适合作为 final candidate 的方案。

2. **性能上限备选：`m2_robust_gamma10`**  
   解释：mean Sharpe 超过 1.0，但 CV 高，适合作为高收益上限而非主部署候选。

3. **稳定 fallback：`imadl_m2_alpha06`**  
   解释：Sharpe 不如 robust M2，但 CV 较低，可作为机制不同的对照组。

4. **历史对照：Phase 1.5 IMADL / A4 / M2**  
   解释：用于说明从 Phase 1.5 到 Phase 2 的研究动机，而不是直接作为同口径 performance benchmark。

---

## 9. 需要修正或统一的 thesis draft 口径

`doc/thesis/chapter6_experimental_results.md` 已经有完整章节框架，但与 Phase 2.5 diagnostics 存在若干口径冲突。向导师汇报或正式写论文前建议统一：

1. **模型架构**  
   Phase 2.5 diagnostics 和 repo 配置显示当前使用 MLP `[64,32,16]`、tanh、dropout 0.0；thesis draft 中出现 “3-layer LSTM with 128 hidden units” 的说法，需要核对并修正。

2. **测试窗口**  
   Phase 2.5 确认 Phase 1.5/2.1b 使用 24-month test window；interim report 的 Semester 1 baseline 曾使用 Jan-Jun 1995 的 6-month setting。thesis 中必须区分不同阶段，不能混写。

3. **seed 集合**  
   Phase 1.5 robustness 使用 `42, 52, 62`；Phase 2.2-fix normalized runs 使用 `42, 123, 456` 的记录。不同阶段的 seed 不应混为同一套。

4. **CV 数字**  
   Phase 2.2 integrated analysis 记录 `gamma07` CV 为 0.1808，`gamma10` CV 为 0.5613；thesis draft 中有 CV 0.0356 / 0.1151 的表格，应回到原始 CSV/JSON 后统一。

5. **跨阶段提升说法**  
   因为 `lambda_dir` 和 IMADL implementation 不同，不能直接写 “Phase 2 improved Phase 1.5 M2”。更稳妥写法是：Phase 2 explored a new M2 variant with robustness penalty and achieved strong internal performance.

---

## 10. 导师可能会问的问题与建议回答

### Q1. 你现在最推荐哪个 loss？

建议回答：

> 当前我推荐 `m2_robust_gamma07`。它不是 mean Sharpe 最高的，`gamma10` 更高，但 `gamma07` 在 Sharpe 和 seed stability 之间更平衡。对于论文和实际策略，稳定性比单纯最高 Sharpe 更重要。

### Q2. Phase 2 是否已经证明超过 Phase 1.5？

建议回答：

> 我会谨慎表述。Phase 2 内部比较显示 robust M2 family 很有前景，但因为 Phase 1.5 M2 与 Phase 2 M2 的 `lambda_dir` 不同，不能直接说 Phase 2 超过 Phase 1.5 M2。我已经在 Phase 2.5 中识别并记录了这个口径问题。

### Q3. 为什么不选择最高 Sharpe 的 `gamma10`？

建议回答：

> `gamma10` mean Sharpe 最高，但 CV 和 seed sensitivity 明显更大。`gamma07` 牺牲少量 mean Sharpe，换来更低的波动和更好的累计收益稳定性。因此如果目标是稳健的 trading loss，`gamma07` 更适合作为主推荐。

### Q4. Normalization 失败说明什么？

建议回答：

> 它说明 component scale imbalance 不一定是 bug，也不是所有 loss 都适合同一种 normalization。Diagnostics 发现 scale ratio 很大，所以我测试了 normalized variants；结果显示 normalization 只改善 `gamma07`，但降低 `gamma10` 和 `alpha06`。因此 normalized `gamma07` 可以作为备选增强，但原始 `gamma07` 更简单、更稳妥，适合作为主推荐。

### Q5. GMADL deviation 会不会影响结论？

建议回答：

> 不影响 Phase 2 内部 ranking 的主要结论。完整诊断报告认为这不是 bug，而是环境/数值/训练轨迹差异；而 gamma refinement 和 Phase 2 P0 都是在同一 runner、同一配置下做内部比较。它只影响“严格复现 Phase 1.5 数值”的强声明；如果需要完全闭环，我可以做 GMADL minimal reproduction，锁定环境版本和 checksum。

---

## 11. 后续工作建议

### 11.1 必做

1. **修正 thesis 中的配置口径**  
   统一模型架构、test window、seed、CV 和跨阶段比较说法。

2. **更新 loss naming**  
   明确区分 `m2_lambda1/2/5`，并区分 original IMADL 与 rebalanced IMADL。

3. **把最终主线收敛到 robust M2**  
   论文主结果围绕 `m2_robust_gamma07`，把 `gamma10` 和 `imadl_m2_alpha06` 作为对照。

### 11.2 可选但有价值

1. **GMADL minimal reproduction**  
   可选项。只有在导师要求 strict reproducibility 时才需要；当前完整诊断报告已判断它不是 Phase 2 bug。

2. **真实 loss-component logger**  
   当前 diagnostics 是 metric proxy；如果时间允许，记录训练时 batch-level loss terms 的 mean/std，会让 normalization 结论更有说服力。

3. **同口径 `m2_lambda5` rerun**  
   如果导师特别关心 Phase 1.5 vs Phase 2 直接比较，可以用 Phase 2 runner 跑 `m2_lambda5`，形成 apples-to-apples result。

---

## 12. 汇报用 2 分钟口述稿

> 我这一阶段主要完成了从 Phase 1.5 到 Phase 2.5 的 loss function 系统实验和诊断。Phase 1.5 先证明了 hybrid loss 经过参数调优后可以在单 seed 下获得很高 Sharpe，例如 M1 和 M2，但 multi-seed robustness 也显示这些高收益结果有明显 seed sensitivity。因此 Phase 2 我设计了四类新的 hybrid variants，重点尝试把 M2 的高收益潜力和 robust penalty 结合起来。
>
> Phase 2 P0 完成了 48 个 runs，结果显示 M2+robustness 是最有前景的方向，IMADL+M2 alpha06 可以作为稳定 fallback，而 IMADL+GMADL 和 adaptive hybrid 基本失败。随后 Phase 2.2 对 gamma 做精调，发现 `m2_robust_gamma10` 的 mean Sharpe 最高，但波动较大；`m2_robust_gamma07` 的 Sharpe 约 0.916，CV 约 0.181，是更好的稳定性-收益平衡点。后续 normalization experiment 显示 `gamma07_normalized` 可以把 Sharpe 提到约 0.970，但会增加实现复杂度和潜在方差，所以我仍把原始 `gamma07` 作为主推荐，把 normalized 版本作为备选。
>
> Phase 2.5 的重点是对齐诊断。完整诊断报告的结论是：之前 alignment failure 不是 runner bug，而是比较对象不一致。M2 的问题主要是 `lambda_dir` 口径不同，Phase 1.5 是 5.0，Phase 2.1b 默认是 1.0，Phase 2 robust M2 又是 2.0；IMADL 也有命名碰撞，Phase 1.5 和 Phase 2 的实际公式不同；GMADL 则更可能是环境/数值差异。因此我现在会谨慎表述：Phase 2 不是直接证明超过 Phase 1.5 M2，而是在一个新的 M2 variant 上找到了有效的 robustness-enhanced loss。这个诊断也帮助我避免论文里出现不公平比较。

---

## 13. 主要来源

- Interim report: `doc/2253235_YirongYu_2025.pdf`
- Phase 1.5 results: `doc/phase1.5/Phase1.5-Results-Summary.md`
- Phase 1.5 lambda sweep: `doc/phase1.5/Phase1.5-Lambda-Sweep-Analysis.md`
- Phase 1.5 robustness analysis: `doc/phase1.5/Phase1.5-Robustness-Analysis.md`
- Phase 2 design docs: `doc/phase2/*.md`
- Phase 2 P0 reports: `doc/phase2-fix/reports/phase2_results_review.md`, `doc/phase2-fix/reports/phase2_grouped_summary.csv`, `doc/phase2-fix/reports/phase2_raw_runs.csv`
- Phase 2.2 integrated analysis: `doc/phase2-fix/reports/phase2_2_integrated_analysis.md`
- Phase 2.5 full alignment diagnosis: `phase2.5-对齐失败完整诊断报告.md`
- Phase 2.2 normalization analysis: `doc/phase2-fix/phase2.2-fix/LOSS_COMPONENT_ANALYSIS_RESULTS.md`
- Phase 2.2/fix1 local summaries for final cross-check: `doc/phase2-fix/phase2.2-fix/phase1_summary.json`, `doc/phase2-fix/phase2.2-fix/phase2_summary.json`, `doc/phase2-fix/phase2.2-fix1/phase1_summary.json`, `doc/phase2-fix/phase2.2-fix1/phase2_summary.json`
- Phase 2.5 diagnostics:
  - `doc/phase2.5/01_config_comparison.md`
  - `doc/phase2.5/02_lambda_dir_check.md`
  - `doc/phase2.5/03_seed_verification.md`
  - `doc/phase2.5/04_data_preprocessing.md`
  - `doc/phase2.5/05_test_window_analysis.md`
  - `doc/phase2.5/06_architecture_comparison.md`
  - `doc/phase2.5/07_loss_implementation_details.md`
- Thesis draft: `doc/thesis/chapter1_introduction.md`, `doc/thesis/chapter2_literature_review.md`, `doc/thesis/chapter3_methodology.md`, `doc/thesis/chapter5_loss_function_design.md`, `doc/thesis/chapter6_experimental_results.md`
