# Q&A Preparation（占 10% 评分）

> 评分标准：Responds accurately, confidently, and fully to all questions; demonstrates deep understanding.

以下按"高概率 → 中概率 → 低概率"排列，每题给出**考官可能的措辞**和**建议回答**。

---

## 🔴 高概率问题（几乎必问）

### Q1: 为什么只用 3 个 seeds？统计显著性够吗？

**考官意图**：质疑 evidence 的统计力度。

**回答**：
> 3 seeds 是计算资源约束下的务实选择——每个 config 需要完整的 24-month forward pass。我承认 3 seeds 不能给出 confidence interval，但它已经足够区分 CV 差异一个数量级的 candidates（gamma07 CV=0.18 vs gamma10 CV=0.56）。论文明确将 "≥10 seeds for tighter CI" 列为 future work 的第一优先级。这是一个 bounded claim：我们说的是 "best-supported under 3-seed evidence"，不是 "statistically significant at p<0.05"。

---

### Q2: 为什么不用 rolling window？单一 24-month 窗口能代表什么？

**考官意图**：质疑 external validity / generalisability。

**回答**：
> 这是一个 deliberate design choice。Rolling window 同时改变了 training data 和 market regime，引入了多个 confounding factors。我们的研究问题是 "loss function 本身的效应"，所以需要一个 controlled single-factor design：固定一切，只变 loss。这给了 clean internal validity，代价是 limited external validity。Rolling window 是 future work 的第二优先级——一旦确认了 loss 的 marginal effect，下一步就是测试它在不同 regime 下是否 robust。

---

### Q3: R² 极度负值（-10⁹）怎么解释？模型是不是完全失败了？

**考官意图**：测试你对 evaluation metrics 的理解深度。

**回答**：
> 不是。R² 衡量的是 point prediction 的 calibration quality——预测值和真实值在 scale 上的匹配。但 long-short portfolio 只依赖 cross-sectional ranking，不需要 calibrated magnitudes。Directional losses（MADL, GMADL）让模型输出的 scale 偏离真实 return 几个数量级，所以 R² 爆炸性地负。但 ranking 信息完好，portfolio Sharpe 仍然正。这正是 RQ1 的核心发现：prediction-level metrics 和 portfolio-level metrics 解耦。R² 在这个 setting 下是 scale diagnostic，不是 performance metric。

---

### Q4: Phase 1/2 用 ReLU+dropout=0.2，Phase 3 用 tanh+dropout=0。为什么改了？能跨 phase 比较吗？

**考官意图**：质疑实验一致性 / 是否 cherry-picking。

**回答**：
> Phase 3 的 activation/dropout 变化来自 branch `phase2.2-fix` 的 codebase evolution。论文明确声明：**intra-phase comparison is strong**（同一 table 内只变 loss），**inter-phase comparison is moderate to weak**（跨 phase 有 confounding）。我们从不做 "Phase 3 improves Phase 2 by X points" 这样的 claim。Phase 1/2 的作用是 motivation（确定 hybrid-multiplicative 方向），Phase 3 是 final evidence（multi-seed robustness）。每个 phase 内部的 controlled comparison 是 valid 的。

---

### Q5: Variance penalty $\gamma \cdot \text{Var}(\hat{y})$ 的直觉是什么？为什么 γ=0.7 是最优？

**考官意图**：测试 mathematical insight。

**回答**：
> 直觉：没有 variance penalty 时，multiplicative hybrid loss 可能让模型对少数股票给出极端预测（因为 directional term 奖励大幅正确预测）。这导致 portfolio weights 集中在少数 names 上，增加 idiosyncratic risk。Var(ŷ) penalty 惩罚 prediction spread，迫使模型给出更均匀的 cross-sectional signal。
>
> 为什么 0.7？这是一个 empirical sweet spot：太小（γ=0.3）under-regularises，prediction 仍然发散，seed sensitivity 高（CV=1.06）；太大（γ=1.5）over-compresses signal，丢失 ranking information。γ=0.7 恰好在 Sharpe、CV、portfolio volatility 三个维度同时达到 internal optimum。这不是 monotone 关系，是 U-shaped stability curve 的底部。

---

## 🟡 中概率问题

### Q6: 为什么 normalisation 让 gamma10 崩塌但 gamma07 稳定？

**回答**：
> Normalisation 改变了 directional component 和 MSE component 的相对权重。gamma07 的 signal 不依赖于两个 component 的特定 scale ratio——它在 original 和 normalised 设置下都产生类似的 ranking。gamma10 的 higher Sharpe 部分来自 scale imbalance 带来的 implicit weighting，一旦 equalise scale，这个 advantage 消失。这说明 gamma07 的 signal 是 genuine loss-family effect，而 gamma10 的 edge 是 contingent on scale。

---

### Q7: 你的 portfolio construction（top/bottom 10%, 5% cap）是否过于简单？

**回答**：
> 是 deliberately simple。研究目的是隔离 loss function 的效应，不是设计最优 portfolio strategy。如果用更复杂的 portfolio rule（e.g., mean-variance optimization），performance 差异可能来自 portfolio construction 而非 loss。Simple rule = clean attribution。Future work 可以测试 loss recommendation 在不同 portfolio rules 下是否 robust。

---

### Q8: 为什么选 MLP 而不是更复杂的模型（LSTM, Transformer）？

**回答**：
> 同样的 single-factor logic。如果同时变 architecture 和 loss，无法 attribute performance 差异。MLP 是 return prediction 文献中最常用的 baseline architecture。一旦确认 loss 的 marginal effect，下一步可以测试 architecture sensitivity——但那是一个不同的 research question。

---

### Q9: 24-month test period (1995-1996) 是否有 survivorship bias 或 look-ahead bias？

**回答**：
> Training window 严格在 test window 之前（1990-1994 train, 1995-1996 test），没有 look-ahead。数据来自 CRSP-style panel，包含 delisted stocks，所以 survivorship bias 是 minimal。但我承认 single window 可能恰好是 favorable/unfavorable regime——这是 rolling window future work 要解决的问题。

---

### Q10: Hybrid multiplicative loss 的数学形式是什么？能写出来吗？

**回答**：（准备在白板/备用 slide 上写）
> M2 base: $L_{M2} = \text{MSE}(y, \hat{y}) \cdot (1 + \text{DirectionalPenalty})$
>
> M2-robust: $L = L_{M2} + \gamma \cdot \text{Var}(\hat{y})$
>
> 其中 DirectionalPenalty 在预测方向错误时增大 loss，正确时减小。Var(ŷ) 是 batch 内预测值的方差。

---

## 🟢 低概率但需准备

### Q11: 这个研究的 practical implication 是什么？谁会用？

**回答**：
> Quantitative fund managers 在训练 return prediction models 时，通常默认用 MSE。我们的结果表明：如果目标是 long-short portfolio performance 而非 point prediction accuracy，换一个 hybrid loss（加 directional alignment + variance regularisation）可以在不改变 architecture 和 data 的情况下显著提升 Sharpe。这是一个 low-cost, high-impact 的改进。

---

### Q12: 你的 contribution 和现有 MADL/GMADL 文献的区别？

**回答**：
> MADL/GMADL 文献提出了 directional loss 的概念，但没有在 controlled protocol 下和 regression/robust losses 比较，也没有做 multi-seed robustness。我们的贡献是：(1) 把所有 loss families 放在同一个 protocol 下比较；(2) 设计了 hybrid loss 把 directional 和 robust 结合；(3) 用 multi-seed + normalisation probe 给出 bounded recommendation 而非 single-seed claim。

---

### Q13: 如果给你更多时间，你会优先做什么？

**回答**：
> 三件事按优先级：(1) 10+ seeds 提高统计力度；(2) Rolling window 跨 market regime 验证；(3) Per-component loss logger 替代 diagnostics-estimated scale ratios，让 normalisation probe 更 rigorous。

---

## 💡 Q&A 应答技巧

1. **30 秒规则**：先用一句话给核心答案，再展开细节
2. **不确定时**：先 paraphrase 问题确认理解，再回答
3. **承认 limitation**：坦诚说 "这是 bounded claim"，比硬撑更得分
4. **引用数字**：回答中带具体数字（Sharpe 0.92, CV 0.18）显示熟悉度
5. **连接 RQ**：尽量把回答 link 回三个 research questions
