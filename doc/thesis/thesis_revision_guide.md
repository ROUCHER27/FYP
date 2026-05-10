# Thesis 修改总结指南

**用途**：这份文档从 `final_report_requirements_gap_analysis.md` 中提取最可执行的部分，直接回答“现有 thesis 哪些地方要改”。它不是新的长分析，而是 final report 改稿路线图。

**总原则**：现在不要优先做更多实验。优先把已有 thesis 改成一篇独立、完整、学校格式合规、实验口径一致的 final report。

---

## 1. 最终目标结构

当前 `doc/thesis` 的章节结构仍带有 interim/phase report 痕迹。最终建议改成：

1. `Chapter 1: Introduction`
2. `Chapter 2: Literature Review`
3. `Chapter 3: Data`
4. `Chapter 4: Methodology`
5. `Chapter 5: Empirical Results and Discussion`
6. `Chapter 6: Conclusion`
7. `References`

需要新增或拆分的文件：

| 建议文件                                       | 来源/作用                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| `front_matter.md` 或 LaTeX front matter     | 封面、空白页、abstract、keywords、中文摘要、acknowledgements                              |
| `chapter3_data.md`                         | 从当前 `chapter3_methodology.md` 拆出 Data 和 feature engineering                 |
| `chapter4_methodology.md`                  | 由当前 `chapter3_methodology.md` 的方法部分 + `chapter5_loss_function_design.md` 整合 |
| `chapter5_empirical_results_discussion.md` | 由当前 `chapter6_experimental_results.md` 改名/重构                                |
| `chapter6_conclusion.md`                   | 新写 conclusion                                                               |
| `references.bib` 或 `references.md`         | 新建 bibliography                                                             |

---

## 2. 必须先修的全局问题

### 2.1 不要让 final report 像补充报告

需要删除或改写这些 framing：

- `Semester 1 progress`
- `Knowledge Gained and Skills Required`
- `Progress in Semester 1`
- “之前报告已经说明，这里补充……”
- 把 Phase 2.5 当成论文主结构

正确做法：

- Phase 1.5/2/2.2/2.5 只作为 methodology 或 empirical results 里的实验流程。
- 主线改成：研究问题 -> 文献缺口 -> 数据 -> 方法 -> 实证结果与讨论 -> 结论。

### 2.2 统一最终实验口径

当前 thesis 中有几类冲突必须统一：

| 问题 | 当前冲突 | 建议 |
| --- | --- | --- |
| 测试窗口 | 早期 baseline 使用 1995-01 到 1995-06；Phase 1.5/2.1b/2.2 主结果使用 1995-01 到 1996-12 | final report 主实证口径统一为 24 个月；6 个月只标为 preliminary baseline sanity check |
| 模型结构 | `chapter6` 写 LSTM；诊断材料支持 MLP `[64, 32, 16]` | final report 统一为实际 runner/config 的 MLP |
| batch/epoch | `chapter6` 写 batch 256、50 epochs；当前实验口径多为 batch 1024、20 epochs | 以实际实验配置为准 |
| seeds | 有 `{42,123,456}`，也有 `{42,52,62}` | 按 phase 列出 seed set，不写一个全局 seed set |
| CV | `chapter6` 多处写 `gamma07 CV=0.0356` | 以 integrated analysis 为主：`gamma07 CV=0.1808` |
| normalization | `chapter6` 写 normalization failed across all losses | 改成：不是通用修复；gamma07 normalized 有改善，但 gamma10/alpha06 下降 |

### 2.3 统一 final conclusion

最终报告的主结论建议固定为：

- 主推荐：`m2_robust_gamma07`，Mean Sharpe `0.9156`，CV `0.1808`，Sharpe 和稳定性平衡最好。
- 高收益备选：`m2_robust_gamma10`，Mean Sharpe `1.0043`，但 CV `0.5613`，seed 波动大。
- 稳定 fallback：`imadl_m2_alpha06`，Mean Sharpe `0.6895`，CV `0.2443`。
- 不建议主推：IMADL+GMADL blend、adaptive hybrid。
- normalization 结论：不是通用修复；可以作为 gamma07 的 optional enhancement，但不能说它全面解决 scale imbalance。

---

## 3. 按现有文件逐项修改

## 3.1 `chapter1_introduction.md`

### 保留

- 1.1 Background and Motivation 的整体方向可以保留。
- 1.2 Research Gap 可以保留。
- 1.3 Research Objectives 基本可保留。
- 1.4 Research Questions 基本可保留。

### 必须修改

1. **删除旧 thesis structure。**

当前 1.6 仍写：

- Chapter 4: Progress in Semester 1
- Chapter 5: Knowledge Gained and Skills Required
- Chapter 6: Hybrid Loss Function Design
- Chapter 7: Experimental Results
- Chapter 8: Conclusion and Future Work

这些不符合 final report 要求。改为：

- Chapter 1: Introduction
- Chapter 2: Literature Review
- Chapter 3: Data
- Chapter 4: Methodology
- Chapter 5: Empirical Results and Discussion
- Chapter 6: Conclusion

2. **Scope and Limitations 里的实验设置要统一。**

当前写：

- main test period: January 1995 to December 1996
- 24-month controlled static evaluation
- 6-month static sanity check only for early baseline diagnostics

需要改成更严谨的阶段性表述。例如：

> The empirical analysis uses a controlled static training protocol. The model is trained on 1990-1994 and evaluated out-of-sample. Some early baseline checks use a shorter 1995-01 to 1995-06 window, while the main Phase 1.5-2.2 comparisons use the 1995-01 to 1996-12 window. Each table reports the exact evaluation window used.

3. **减少 “Semester 1” 叙事。**

可以说 “preliminary analysis”，但不要让 final report 依赖之前报告。建议把 “Semester 1 analysis identified...” 改成 “Preliminary analysis in this project identified...”。

---

## 3.2 `chapter2_literature_review.md`

### 保留

- 文献综述整体结构可用。
- Algorithmic investment、testing framework、loss functions、GMADL、robust losses 都是相关内容。

### 必须修改

1. **修正 testing protocol 口径。**

当前 chapter2 里有类似：

- train on 1990-1994; main empirical tables test on 1995-01 to 1996-12
- early baseline sanity checks may test on 1995-01 to 1995-06, but must be labelled preliminary
- seeds 42, 123, 456

这些要和最终 methodology 保持一致。建议只在 literature review 中概括“controlled static evaluation”，不要写太细的 test window 和 seed，避免和 methodology 冲突。

2. **补强 robust loss 与 heavy-tailed returns 的连接。**

最终结果主推 `m2_robust_gamma07`，所以文献综述应明确说明金融收益 heavy-tailed/outlier 问题为什么需要 Huber/MedSE/robust components。

3. **把 GMADL 定位写得更谨慎。**

不要写成“GMADL 已经被严格复现后失败”。应写成：

> GMADL motivates the use of directional objectives, but the current project identifies limitations in its reward/penalty structure and extends the idea through hybrid directional-robust loss functions.

---

## 3.3 `chapter3_methodology.md`

### 总体处理

这个文件不建议直接作为最终 Chapter 3。建议拆成两部分：

- 数据相关内容迁移到 `chapter3_data.md`。
- 方法相关内容改成 `chapter4_methodology.md`。

### 迁移到 `chapter3_data.md`

建议迁移：

- `3.2 Data Description and Preprocessing`
- `3.3 Feature Engineering`
- 数据来源、样本筛选、feature set、train/test split、preprocessing。

新 Data chapter 应包含：

1. Data source: CRSP monthly US equities。
2. Sample period: 原始样本期与实际实验样本期分开写。
3. Asset filtering: 股票筛选、缺失值、return/price/turnover 等处理。
4. Feature construction: X1/X2/X3 或最终真实使用的 feature set。
5. Train/test split: 按 phase 说明。
6. Data limitations: US equities、monthly frequency、无 transaction costs、无 liquidity/market impact。

### 保留到 `chapter4_methodology.md`

建议保留并重写：

- Research design
- Model architecture
- Loss functions
- Training protocol
- Portfolio construction
- Evaluation metrics
- Experimental design

### 必须修正

1. **测试窗口。**

当前多处写 `1995-01 to 1995-06`。最终主结果表应统一到 `1995-01 to 1996-12`；6 个月窗口只能作为 early/preliminary baseline sanity check 出现。

2. **seed set。**

当前写 “most experiments use seeds 42, 123, 456”。这不准确。应按 phase 写：

- Phase 1 baseline: seed 42。
- Phase 1.5 robustness: `{42, 52, 62}`。
- Phase 2/2.2/normalization: 按实际结果文件确认，一般为 `{42, 123, 456}`。

3. **模型配置。**

Methodology 里必须和 results 一致。最终建议写 MLP `[64, 32, 16]`，tanh，dropout 0.0，Adam lr 0.001，batch size/epochs 以实际 runner 为准。

4. **loss function section 要和 `chapter5_loss_function_design.md` 合并。**

如果 final report 保留完整 Methodology，就不要让 loss design 散在另一个独立 chapter 里。可以在 Methodology 中设置：

- Baseline losses
- Directional losses
- Hybrid M1/M2
- IMADL-M2 blend
- M2 robustness family
- Normalized variants

5. **experimental design 需要加入 Phase 2.5。**

当前 methodology 到 Phase 2.2-fix1 为止。需要补：

- Phase 2.5 alignment diagnostics
- 作用：确定哪些 cross-phase claims 安全，哪些不能直接比较。

---

## 3.4 `chapter5_loss_function_design.md`

### 总体处理

这个文件内容有价值，但不建议在 final report 中继续作为 Chapter 5。原因是老师要求 `methodology` 完整，而 Chapter 5 应该留给 `Empirical Results and Discussion`。

建议：

- 把核心公式和设计逻辑并入 `chapter4_methodology.md`。
- 删除过强的 “Semester 1 Findings” framing。
- 把 “Supervisor Guidance Integration” 改成正式论文语言，例如 “Design Motivation” 或 “Loss Design Rationale”。

### 必须修改

1. **章节编号要重排。**

如果最终结构里 Chapter 5 是 results，那么当前 `Chapter 5: Loss Function Design` 不能继续存在。

2. **“Chapter 8 future work” 引用要删。**

当前有 “discussed in Chapter 8” 这类旧结构引用。最终只有 Chapter 6 Conclusion，应改成 “discussed in the conclusion” 或删除。

3. **normalization 相关表述要谨慎。**

如果提到 batch normalization / scale balance，要和最终 normalization experiment 结论一致：不是通用修复。

4. **避免把所有 loss design 写成最终都成功。**

应明确说明这些是候选设计，最终 empirical results 会筛选出：

- 成功方向：M2 + robustness。
- 稳定 fallback：IMADL-M2 alpha06。
- 失败方向：IMADL+GMADL blend、adaptive hybrid。

---

## 3.5 `chapter6_experimental_results.md`

### 总体处理

这是当前最重要的 empirical 材料来源，但需要改名/重构为：

`chapter5_empirical_results_discussion.md`

老师说 discussion 基本就是 empirical 部分，所以这里不只是放结果，而要明确讨论哪个好、哪个差、为什么。

### 必须修改

1. **标题改成 `Empirical Results and Discussion`。**

不要只叫 `Experimental Results`。

2. **删除或修正错误 setup。**

当前 6.1.2 写：

- Main testing period: 1995-01 to 1996-12
- Early baseline sanity-check period: 1995-01 to 1995-06, only when explicitly labelled
- Seeds `{42,123,456}`
- Model: 3-layer LSTM with 128 hidden units
- Batch size 256
- Epochs 50 with early stopping

需要改成与实际实验一致的配置。尤其 LSTM 必须改掉。

3. **修正 Phase 2.2 gamma refinement 数字。**

当前多处写：

- `m2_robust_gamma07` CV `0.0356`
- `gamma10` CV `0.1151`
- `gamma07` 稳定性极强

建议改成：

- `m2_robust_gamma07`: Mean Sharpe `0.9156`, CV `0.1808`
- `m2_robust_gamma10`: Mean Sharpe `1.0043`, CV `0.5613`
- `imadl_m2_alpha06`: Mean Sharpe `0.6895`, CV `0.2443`

由此结论改成：

> `gamma07` is preferred not because it has near-zero variance, but because it provides the best balance between high Sharpe and materially lower seed sensitivity than `gamma10`.

4. **修正 normalization 结论。**

当前写 “Normalization failed across all losses” 和 “Scale imbalance is a feature, not a bug”。这太绝对。

建议改成：

> Normalization is not a universal improvement. It improves the gamma07 variant in the authored analysis, but degrades gamma10 and imadl_m2_alpha06. Therefore, the original gamma07 remains the main recommendation, while normalized gamma07 can be presented as an optional variant rather than the central result.

5. **加入 Phase 2.5 alignment diagnostics。**

建议在最后加一个 subsection：

`5.x Alignment Diagnostics and Claim Boundaries`

内容包括：

- Phase 2 是新 variants 的内部比较，不是 Phase 1.5 exact replication。
- M2 cross-phase comparison 受 `lambda_dir` 不一致影响。
- IMADL 存在命名/公式碰撞。
- GMADL 偏差不视为 blocking bug，但不能 claim strict numerical replication。
- 因此最终结论应聚焦 Phase 2 内部排序。

6. **每个结果段落都要明确 “哪个更好/更差”。**

建议每个 subsection 最后加一句：

- Phase 1.5：M2 有潜力但 seed sensitivity 高。
- Phase 2：M2+robustness 是最有前景方向。
- Variant 1：稳定 fallback。
- Variant 2：失败，不作为主线。
- Variant 4：不稳定，不作为主线。
- Phase 2.2：gamma07 是主推荐，gamma10 是高 Sharpe 备选。
- Normalization：不是通用修复。

---

## 4. 需要新增的内容

## 4.1 Front Matter

学校要求必须有：

- 英文题目
- 中文题目
- student name
- student ID
- submission date
- supervisor name
- blank page
- English abstract
- 2-6 English keywords
- Chinese abstract
- acknowledgements
- contents
- list of figures/tables

这是格式硬要求，不是内容优化。

## 4.2 `chapter3_data.md`

必须新增或拆出。老师明确说 final report 包含 data。

建议标题：

`# Chapter 3: Data`

建议小节：

1. Data Source
2. Sample Construction
3. Train-Test Split
4. Feature Variables
5. Data Preprocessing
6. Data Limitations

## 4.3 `chapter4_methodology.md`

建议重构为完整方法论。

建议小节：

1. Research Design
2. Model Architecture
3. Loss Function Families
4. Training Protocol
5. Portfolio Construction
6. Evaluation Metrics
7. Experimental Design
8. Reproducibility and Claim Boundaries

## 4.4 `chapter6_conclusion.md`

必须新增。

建议包含：

1. 回答研究问题。
2. 总结核心发现：loss choice affects portfolio performance。
3. 推荐 `m2_robust_gamma07`。
4. 说明 `gamma10` 和 `imadl_m2_alpha06` 的角色。
5. 说明失败方向。
6. 写限制：static sanity check、未计交易成本、US equities only、非 rolling backtest。
7. 写 future work：rolling backtest、transaction costs、regime expansion、strict reproduction if required。

## 4.5 References

必须新增。

建议至少包括：

- Gu et al. 2020 / empirical asset pricing with machine learning
- MADL/GMADL 相关论文
- Huber loss / robust loss 文献
- Sharpe ratio / portfolio evaluation
- CRSP/data source 或数据说明
- 其他 chapter2 已引用的来源

---

## 5. 最快改稿顺序

### 第一步：先改结构

1. 改 `chapter1_introduction.md` 的 thesis structure。
2. 新建 `chapter3_data.md`。
3. 把当前 `chapter3_methodology.md` 改造成 `chapter4_methodology.md`。
4. 把当前 `chapter6_experimental_results.md` 改造成 `chapter5_empirical_results_discussion.md`。
5. 新建 `chapter6_conclusion.md`。

### 第二步：统一口径

统一：

- test window
- model architecture
- batch size
- epochs
- seed sets
- gamma07/gamma10/alpha06 Sharpe 和 CV
- normalization conclusion

### 第三步：补学校必需内容

补：

- front page
- blank page
- abstract + keywords
- Chinese abstract
- acknowledgements
- references

### 第四步：最后 polish

检查：

- 每章编号是否连续。
- 图表编号是否连续。
- 文内引用是否有 references。
- final report 是否能独立阅读。
- 是否还残留 “Chapter 8”“Semester 1 progress”“Knowledge Gained” 等旧结构。

---

## 6. 一句话版本

现有 thesis 不是缺 empirical results，而是需要从“阶段性实验记录”改成“独立 final report”：重排章节，拆出 Data，整合完整 Methodology，把 Experimental Results 改成 Results and Discussion，补 Conclusion/References/front matter，并把所有实验口径和最终数字统一。
