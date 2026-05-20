

> 演讲时间时长：15–20 分钟 | 页数：15–18 页 | 占比：20% 总成绩

---

## 整体叙事策略

**开场抛出三个问题 → 中间展示方法与实验 → 结尾逐一回答三个问题（揭晓贡献）**

三个问题即论文的 RQ1–RQ3：
1. Loss function 的选择如何影响 prediction-level 和 portfolio-level 的表现？
2. 哪种 hybrid-loss 设计能给出最佳的 Sharpe-stability trade-off？
3. 领先的 hybrid-loss 候选者在 component normalisation 诊断下是否稳定？

---

## 逐页规划（共 17 页）

### Part 1: 引言 (2 页)

> **流程参考**：第 1 页对应论文 Chapter 1 (Introduction) 的逻辑链；第 2 页对应 Chapter 2 (Literature Review) 的三段结构。**简洁，不啰嗦**。

| 页码    | 标题                           | 内容要点                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Background & Three Questions | **按 Introduction 的因果链讲，每条只说一句**：<br>① ML 已广泛用于 cross-sectional return prediction 和 long-short portfolio；<br>② 但绝大多数研究**只调 architecture 和 features，loss function 默认用 MSE**；<br>③ 这个默认选择和任务不匹配——portfolio 只依赖 **ranking**（不是 calibrated values），且月度收益是 **heavy-tailed**（违反 MSE 的 Gaussian 假设）；<br>④ 因此引出三个 RQ（大字体，编号 ①②③，告诉听众"答案在最后揭晓"）。<br>**控制时间在 1 分钟以内**，每条背景一句话即可，不展开。                                                                                                                                       |
| **2** | Literature Snapshot          | **按 Chapter 2 的三段结构精炼**：<br>① **ML for Return Prediction**（Gu, Kelly, Xiu 2020；Daniel-Moskowitz 2016）—— 关键 finding: 非线性 ML 优于线性因子模型，但 loss 始终是 MSE；<br>② **Robust Losses**（Huber 1964；MedSE）—— 处理 heavy tails，但忽略 directional 信息；<br>③ **Directional Losses**（MADL/GMADL, Michańków et al. 2024）—— 奖励正确方向，但缺乏 magnitude calibration。<br>右侧放 2×2 矩阵图（Regression↔Directional × Standard↔Robust），收尾一句：*"No prior work compares all four quadrants under the same protocol — that's the gap."*<br>**控制时间在 1 分钟以内**。 |

### Part 2: 我们的工作 + 贡献 (2 页)

| 页码 | 标题 | 内容要点 |
|------|------|----------|
| **3** | What We Did (Overview) | 一张流程图：Data → MLP → Loss → Portfolio → Evaluation。核心信息：我们**设计了一种新的 hybrid loss function（M2-robust-γ）**，将 directional alignment 与 robust magnitude control 结合，并通过多阶段实验验证其有效性和稳定性。 |
| **4** | Four Contributions | 用 4 个 bullet points（带图标）：① 设计了 hybrid loss（additive + multiplicative 两个家族）② 提出 M2-robust-γ variance penalty 机制 ③ 多阶段递进验证：单 seed → multi-seed → normalisation 诊断 ④ 给出 bounded recommendation（不是 absolute claim）。每条一句话。 |

### Part 3: 方法论 (4 页) — 对齐论文 Chapter 3

> **流程参考**：论文 Chapter 3 包含 §3.1 Research Design / §3.2 Model / §3.3 Loss Families / §3.4 Training / §3.5 Portfolio / §3.6 Metrics / §3.7 Phases。压缩到 4 页：

| 页码 | 标题 | 内容要点 |
|------|------|----------|
| **5** | Research Design & Architecture (§3.1 + §3.2) | **左半**：Single-factor controlled comparison 示意图——固定 data / features / MLP / portfolio / window，**只变 loss**。一句话："Clean causal attribution of performance to loss choice."<br>**右半**：MLP architecture 框图：input 15 → 64 → 32 → 16 → 1，ReLU + dropout 0.2。说明 architecture 由先期 grid search 选定后**冻结**。 |
| **6** | Loss Function Families (§3.3) | **核心页**。展示四类 loss 的层级关系：<br>① **Regression**: MSE, MedSE<br>② **Directional**: MADL, GMADL, IMADL<br>③ **Hybrid Additive (A1–A5)**: $L = \lambda_{dir} \cdot D + \lambda_{hub} \cdot H_\delta$<br>④ **Hybrid Multiplicative (M1–M4)**: $L = (1 + \lambda_{dir} \cdot D) \cdot H_\delta$<br>⑤ **M2-robust-γ (我们的设计)**: $L_{M2\text{-}robust} = L_{M2} + \gamma \cdot \text{Var}(\hat{y})$ ← 框红/高亮<br>用 fig3_2_hybrid_loss_surfaces 或 fig3_3_triple_property 配图。 |
| **7** | Portfolio Construction & Evaluation (§3.5 + §3.6) | **左半**：用 fig3_4_portfolio_flow 展示 6-step pipeline：predictions → top/bottom 10% bucket → within-bucket z-score (clip [-3,3]) → sign-consistent weights → 5% cap projection → long-short return。<br>**右半**：Evaluation metrics 三类：<br>• Per-month: MSE, MedSE, **R²**（说明它在 directional loss 下会爆负，是 scale diagnostic 而非 performance metric）<br>• Portfolio: monthly long-short return<br>• Annual: **Sharpe** = √12 · r̄/σ, Cumulative return, **CV** = σ_S/\|μ_S\| |
| **8** | Experimental Phases (§3.7) | 用时间线/流程图展示**四阶段递进**（这是论文 §3.7 的精华）：<br>• **Phase 1**: 7 个 baseline losses, seed 42（Table 5.1）<br>• **Phase 2**: A1–A5 + M1–M4 共 9 个 hybrid 变体, seed 42（Table 5.2）<br>• **Phase 3a**: M2-robust γ ∈ {0.3, 0.5, 0.7, 1.0, 1.5}, **3 seeds/row**（Table 5.3）<br>• **Phase 3b**: Integrated α/β/λ sweeps, 3 seeds/row（Table 5.4）<br>• **Phase 4**: Normalisation probe（Table 5.5）<br>底部一行：Train **1990-01..1994-12** / Test **1995-01..1996-12** (24 months)。 |
### Part 4: 数据 (1 页) — 对齐论文 Chapter 4

| 页码 | 标题 | 内容要点 |
|------|------|----------|
| **9** | Data & Features (Chapter 4) | 用 fig4_1_data_coverage 展示数据覆盖。**关键数字**：US equities (CRSP-style panel), monthly frequency, ~2000 stocks/month；**X1 features (15D)** = 5 base panel cols (VOL, RET, SHROUT, r, to) + 10 engineered (cum return & cum turnover at 1/3/6/9/12 月)；**Train 1990-01..1994-12 / Test 1995-01..1996-12** (24 months OOS)。说明：no look-ahead，包含 delisted stocks（minimal survivorship bias）。 |
### Part 5: 实证结果 (6 页) — 核心，对齐论文 §5.2–§5.8

> **流程参考**：完全镜像论文 Chapter 5 的章节顺序，每页对应一个 phase。所有数字直接引用 Table 5.1–5.5。

| 页码 | 标题 | 内容要点 |
|------|------|----------|
| **10** | Phase 1: Baseline Loss Comparison (§5.2) | **图**：fig5_1_baseline_comparison（左：cum return paths；右：Sharpe 排序）。<br>**关键数字**（Table 5.1, seed 42）：MSE Sharpe **−0.46**, MedSE **0.09**, GMADL **0.20**, hybrid_mul_m1 **0.44**（baseline 最高）。<br>**三个观察**：① 传统 regression loss 在 portfolio 上失败；② Directional loss 的 R² 爆负到 **−4×10⁹** 但 Sharpe 仍可正——说明 **R² 和 Sharpe 解耦**；③ hybrid_mul_m1 baseline 最优，**motivates Phase 2**。 |
| **11** | Phase 2: Hybrid A/M Sweep (§5.3) | **图**：fig5_2_phase15_variants（A1–A5 + M1–M4 条形图）。<br>**关键数字**（Table 5.2, seed 42）：**A3 (hybrid_add_a3) Sharpe = 0.5738**（seed-42 peak），M1 = 0.4435（M-series 最高，volatility 最低）。M3 = −0.97 表明参数敏感。<br>**结论**：Hybrid 优于纯 regression/directional；multiplicative form 更稳定 → 选 M2 作为下一步 base 继续 refine。 |
| **12** | Phase 3a: Multi-Seed γ Refinement (§5.4) | **图**：fig5_4_gamma_tuning_curve（三联图：Sharpe / CV / portfolio vol）。<br>**关键数字**（Table 5.3, 3 seeds/row）：γ=0.7 → Sharpe **0.92**, CV **0.18**, cum return **+27.99%**；γ=1.0 → Sharpe 1.00 但 CV 0.56；γ=0.3 → CV 1.06（崩塌）。<br>**结论**：γ 与稳定性是 **U-shaped**，γ=0.7 是 internal optimum；γ=1.0 高 Sharpe 但 seed 敏感。 |
| **13** | Phase 3b: Integrated α/β/λ Sweeps (§5.5) | **图**：fig5_5_imadl_alpha_sweep 或 fig5_6_sharpe_cv_frontier。<br>**关键数字**（Table 5.4, 3 seeds/row）：**imadl_m2_alpha06 Sharpe = 0.69, CV = 0.24, cum return +30.42%**（IMADL-m2 家族 peak）；β/λ 家族大多崩塌（CV > 1）。<br>**结论**：M2-robust-γ07 不是 local optimum——独立家族（IMADL-m2）在不同 corner 也得到一致结论：**hybrid-multiplicative + robust component** 是 productive region。 |
| **14** | Phase 4: Normalisation Probe (§5.6) | **图**：fig5_7_normalisation_probe。<br>**关键数字**（Table 5.5, 3 candidates）：γ07 **0.92 → 0.91**（稳定）；γ10 **1.00 → 0.41**（崩塌）；alpha06 **0.69 → −0.02**（崩塌）。<br>**结论**：γ07 的 signal 不依赖 component scale ratio——是 **genuine loss-family effect**，不是 scale artefact。这是 RQ3 的直接答案。 |
| **15** | Headline Findings & Cumulative Paths (§5.8) | **图**：fig5_8_cumulative_return_paths（左：Phase 1 baseline paths；右：γ sweep 多 seed envelope）。<br>**视觉冲击**：γ07 三条 seed line 都向上、envelope 窄；MSE 全部向下；γ10 best seed 高但 envelope 极宽。<br>一句话总结：*"Within the same protocol, only γ07 delivers consistent positive returns across all seeds AND survives the normalisation probe."* |
### Part 6: 结论 — 揭晓答案 (2 页) — 对齐论文 Chapter 6

| 页码 | 标题 | 内容要点 |
|------|------|----------|
| **16** | Answering the Three Questions (§6.1) | 回到第 1 页的三个问题，逐一给出简洁答案：<br>**A1**: R² 和 Sharpe 解耦——loss 选择对 portfolio 影响巨大（Phase 1 证据）<br>**A2**: **m2_robust_gamma07 最优**——mean Sharpe **0.92**, CV **0.18**, cum return **+27.99%**（Phase 3a 证据）<br>**A3**: 只有 gamma07 在 normalisation 下稳定（0.92→0.91）；gamma10 和 alpha06 都崩塌（Phase 4 证据）<br>每个答案配一个 ✓ 与对应 phase 标签。 |
| **17** | Recommendation, Limitations & Future Work (§6.2–§6.3) | **三层推荐**（Bounded Recommendation）：<br>• **Primary**: gamma07（best Sharpe-stability trade-off）<br>• **High-return alternative**: gamma10（highest mean Sharpe 1.00 但 seed-sensitive）<br>• **Stable fallback**: imadl_m2_alpha06（cum return +30.42%，独立家族 corroboration）<br>**Limitations**（论文 §6.2）：① 仅 3 seeds，无 CI；② 单一 24-month window，无 regime coverage；③ Per-component logger 未实现（normalisation probe 是 diagnostics-grade）。<br>**Future work**：① ≥10 seeds；② Rolling window；③ Per-component logger。<br>结束语：*"If you change one thing in your prediction pipeline — change the loss."* |

---

## 时间分配建议

| 部分 | 页数 | 时间 |
|------|------|------|
| 引言 (抛出问题 + 文献) | 2 | 2 min |
| 我们的工作 + 贡献 | 2 | 1.5 min |
| 方法论 | 4 | 4 min |
| 数据 | 1 | 1 min |
| 实证结果 | 6 | 8 min |
| 结论 (揭晓答案) | 2 | 2.5 min |
| **合计** | **17** | **~19 min** |


---

## 可用图表资源 (`2253235_yirongyu_2026_Supplementary/latex/figures/`)

| 图文件 | 用于页码 | 用途 / 对应论文章节 |
|--------|---------|--------------------|
| fig2_1_loss_shapes.png | 6 | Loss 形状对比（§2.4 / §3.3） |
| fig2_2_madl_gmadl_comparison.png | 6（备用） | MADL vs GMADL（§2.4） |
| fig3_1_loss_reward_penalty_response.png | 6（备用） | Reward/penalty 逻辑（§3.3） |
| fig3_2_hybrid_loss_surfaces.png | 6 | Hybrid loss 3D surfaces（§3.3） |
| fig3_3_triple_property.png | 6 | M2 三个关键性质（§3.3.4） |
| fig3_4_portfolio_flow.png | 7 | Portfolio 6-step pipeline（§3.5） |
| fig4_1_data_coverage.png | 9 | 数据覆盖与样本规模（§4.1–4.2） |
| fig5_1_baseline_comparison.png | 10 | Phase 1 baseline 对比（§5.2 / Table 5.1） |
| fig5_2_phase15_variants.png | 11 | Phase 2 A/M 变体扫描（§5.3 / Table 5.2） |
| fig5_3_gamma_refinement.png | 12（备用） | γ 多 seed Sharpe + CV（§5.4 / Table 5.3） |
| fig5_4_gamma_tuning_curve.png | 12 | γ 三联调参曲线（§5.4 / Table 5.3） |
| fig5_4_normalisation_retention.png | 14（备用） | Normalisation retention（§5.6） |
| fig5_5_imadl_alpha_sweep.png | 13 | IMADL α sweep（§5.5 / Table 5.4） |
| fig5_6_sharpe_cv_frontier.png | 13（备用） | Sharpe-CV frontier（§5.5） |
| fig5_7_normalisation_probe.png | 14 | Normalisation 诊断（§5.6 / Table 5.5） |
| fig5_8_cumulative_return_paths.png | 15 | 累计收益路径（§5.8） |


