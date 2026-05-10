# Final Report 完整要求与当前缺口分析

**生成目的**：根据老师最新指导、学校 `FYP Final Report.pdf` 要求，以及当前 `doc/thesis` 草稿，整理最终论文必须满足的完整要求、当前已经做到的部分、还没有做到的部分和优先修改顺序。

**核心结论**：当前 empirical results 的材料量基本足够，主要风险不在“还要不要做更多实验”，而在 final report 结构、方法论完整性、data 独立呈现、front matter、conclusion/references，以及实验口径一致性。最需要立刻处理的是：把论文从“阶段性补充报告”改成“独立完整 final report”。

---

## 1. 信息来源

本分析基于以下材料：

1. 老师指导：
   - final report 格式基本包含 `introduction`, `literature review`, `methodology`, `data`, `empirical results`。
   - methodology 需要完整的方法论，包含所有内容。
   - final report 应该独立成章，不是之前报告的补充。
   - discussion 基本就是 empirical 部分，需要讨论实证测试结果哪个好、哪个差。
   - 当前 empirical 部分感觉已经足够。
2. 学校 PDF 要求：`FYP Final Report.pdf`。
3. 当前论文草稿：`doc/thesis/`。
4. 当前实验总结与诊断材料：
   - `doc/phase2.5/executive_summary.md`
   - `doc/phase2.5/*.md`
   - `doc/phase2-fix/*.md`
   - `phase2.5-对齐失败完整诊断报告.md`

---

## 2. 学校正式要求清单

### 2.1 提交要求

| 项目 | 学校要求 | 当前状态 | 需要做什么 |
| --- | --- | --- | --- |
| 截止时间 | 2026-05-13 16:00 前 | 需要用户确认提交安排 | 在最终打包前预留 PDF 编译和检查时间 |
| 提交方式 | Learning Mall softcopy | 未涉及 | 最终交 PDF |
| 文件命名 | `StudentID ForenameSurname 2026.pdf` | 未完成 | 导出时按学校格式命名 |
| 补充材料 | 多文件需压缩为同名 ZIP | 未完成 | 如提交 LaTeX、代码、数据或结果，打包 ZIP |
| 占比 | final report 占最终成绩 70% | 已知 | 优先保障正式报告完整性 |

### 2.2 页面顺序与格式要求

| 页面/部分 | 学校要求 | 当前 `doc/thesis` 状态 | 缺口 |
| --- | --- | --- | --- |
| Front Page | 英文题目、中文题目、姓名、学号、日期、导师 | 未见正式 front page 文件 | 缺失 |
| Blank Page | 封面后必须有空白页 | 未见 | 缺失 |
| Abstract Page | 英文 abstract、2-6 个英文 keywords、中文 abstract | 未见 | 缺失 |
| Acknowledgements | 模板包含，可写 | 未见 | 建议补 |
| Contents | 目录 | 未见最终编译入口 | 缺失 |
| List of Figures/Tables | 模板包含 | 未见最终编译入口 | 需要由 LaTeX/Word 自动生成或手工检查 |
| Main Body | 约 6000+ words，不含 abstract、目录、references、tables、公式、代码 | 当前章节正文约 15963 words | 字数足够，但结构和口径需修 |
| References | 必须包含 | 当前没有独立 references/bibliography 文件 | 缺失 |

### 2.3 内容要求

学校要求 final report 至少包含：

1. Introduction
2. Background / Literature Review
3. Description of Methodology
4. Main Results and Discussion
5. Conclusions
6. References

老师进一步要求把 `Data` 明确作为一个部分呈现。因此本项目最终结构应至少包含：

1. Introduction
2. Literature Review
3. Data
4. Methodology
5. Empirical Results and Discussion
6. Conclusion
7. References

---

## 3. 老师指导转化成论文写作要求

### 3.1 Final report 要独立成章

当前论文不能让读者感觉是在读“Phase 1.5/Phase 2/Phase 2.5 的补充说明”。Phase 可以保留，但应该放在 methodology 或 empirical results 中作为实验设计与实验流程，而不是作为论文结构的主线。

需要避免的表述：

- “Semester 1 report 已经说过，所以这里补充……”
- “Phase 2.5 是对齐失败诊断，因此本报告只补充诊断……”
- “Chapter 4: Progress in Semester 1”
- “Chapter 5: Knowledge Gained and Skills Required”

更合适的写法：

- 论文主线是：研究背景 -> 文献缺口 -> 数据 -> 方法论 -> loss function design -> 实证比较 -> 结论。
- Phase 1.5 到 2.5 是实验流程，用来支持 empirical results，而不是读者理解论文的前置条件。

### 3.2 Methodology 必须完整

老师说 methodology 要“完整的方法论，包含所有内容”。这意味着 methodology 不能只写模型或 loss formula，而要覆盖从数据到结果评估的全流程。

建议 methodology 必须包含：

1. 研究设计：controlled static sanity-check，而不是 production rolling backtest。
2. 数据来源与样本构造：CRSP monthly US equities，训练期/测试期，筛选规则。
3. 特征构造：使用哪些 feature set，为什么固定特征以隔离 loss function effect。
4. 模型结构：MLP `[64, 32, 16]`、activation、dropout、optimizer、batch size、epoch 等，以当前实验真实配置为准。
5. Loss functions：baseline losses、MADL/GMADL/IMADL、M1/M2、robust M2、normalization variants。
6. 训练 protocol：seed、early stopping 或固定 epoch、checkpoint、相同配置控制。
7. Portfolio construction：long-short、cap、equal/signal/capped 组合口径。
8. Evaluation metrics：Sharpe、cumulative return、volatility、R2、MSE、CV。
9. Experimental phases：Phase 1.5 到 2.5 是实验组织方式，说明每一阶段解决什么问题。
10. Reproducibility：代码版本、runner、结果文件位置、限制条件。

### 3.3 Data 需要单独可读

老师明确提到 `data`。当前 `chapter3_methodology.md` 已经有 data description，但它嵌在 methodology 里。最终报告最好单独成章或至少单独一级章节，让老师一眼看到数据来源、样本期、过滤规则和变量构造。

建议做法：

- 新增 `Chapter 3: Data`。
- 把当前 `chapter3_methodology.md` 的数据和 feature engineering 内容拆出来。
- 原 methodology 变成 `Chapter 4: Methodology`，重点写实验设计、模型、loss、训练和评估。

### 3.4 Discussion 可以并入 empirical results

老师说 discussion 基本就是 empirical 部分。因此最终可以写成：

`Chapter 5: Empirical Results and Discussion`

这一章不只是放表格，还必须回答：

- 哪个 loss 最好？
- 哪个 loss 最差？
- 为什么 `m2_robust_gamma07` 是主推荐？
- 为什么 `gamma10` Sharpe 更高但不是主推荐？
- 为什么 IMADL+GMADL blend 和 adaptive hybrid 不作为最终方案？
- normalization 为什么不是通用修复？
- Phase 2.5 诊断如何限制结论的表述边界？

---

## 4. 当前 `doc/thesis` 草稿状态

当前已有文件：

| 文件 | 当前内容 | 状态 |
| --- | --- | --- |
| `chapter1_introduction.md` | 背景、research gap、objectives、scope、structure | 部分可用，但 thesis structure 过时 |
| `chapter2_literature_review.md` | 文献综述、loss function、GMADL、定位 | 基本可用，但需统一实验口径 |
| `chapter3_methodology.md` | 数据、特征、模型、loss、训练、portfolio、metrics、实验设计 | 内容多，但应拆 Data，并修正配置 |
| `chapter5_loss_function_design.md` | hybrid loss design 详细说明 | 可并入 methodology 或作为 methodology 子章 |
| `chapter6_experimental_results.md` | 实证结果与阶段性实验 | 材料足够，但需改成 Results and Discussion 并修正数据 |

当前缺失文件/章节：

| 缺失项 | 重要性 |
| --- | --- |
| 封面/front page | 必须 |
| 空白页 | 必须 |
| 英文 abstract + keywords | 必须 |
| 中文 abstract | 必须 |
| acknowledgements | 建议 |
| contents/list of figures/list of tables | 必须或模板要求 |
| 独立 Data chapter | 老师明确要求 |
| Conclusion chapter | 学校明确要求 |
| References/Bibliography | 学校明确要求 |
| 最终编译入口 | 必须 |

---

## 5. 当前最重要缺口

### Gap 1：论文结构仍像 interim/phase report

**严重程度：高。**

当前 `chapter1_introduction.md` 的 thesis structure 仍写：

- Chapter 4: Progress in Semester 1
- Chapter 5: Knowledge Gained and Skills Required
- Chapter 6: Hybrid Loss Function Design
- Chapter 7: Experimental Results
- Chapter 8: Conclusion and Future Work

这与老师“final report 独立成章，不是之前补充”的要求冲突。

**建议修改**：

最终结构改为：

1. Chapter 1: Introduction
2. Chapter 2: Literature Review
3. Chapter 3: Data
4. Chapter 4: Methodology
5. Chapter 5: Empirical Results and Discussion
6. Chapter 6: Conclusion
7. References

Phase 1.5、Phase 2、Phase 2.2、Phase 2.5 只作为 Chapter 4/5 的实验设计和结果分段。

### Gap 2：缺少 final report front matter

**严重程度：高。**

学校明确要求 front page、blank page、abstract page、keywords 和中文 abstract。当前 `doc/thesis` 没有看到这些内容。

**建议补齐**：

1. 英文题目。
2. 中文题目。
3. student name/student ID/submission date/supervisor。
4. 英文 abstract，约 200-300 words。
5. 2-6 个 keywords。
6. 中文摘要。
7. acknowledgements。

### Gap 3：Data 没有独立呈现

**严重程度：高。**

当前 data 在 `chapter3_methodology.md` 内部，老师明确要求 report 包含 data。为了降低结构风险，建议单独成章。

**建议新建**：`chapter3_data.md`

内容应包括：

- 数据来源：CRSP monthly US equities。
- 样本区间：说明完整原始样本区间和本实验实际使用训练/测试窗口。
- 资产筛选：普通股、缺失值、价格/市值/return 过滤。
- 变量定义：return、momentum、turnover、feature set。
- 训练/测试划分：必须与真实实验一致。
- 数据限制：月度数据、US equities、无交易成本、无流动性和市场冲击。

### Gap 4：Methodology 需要合并并完整化

**严重程度：高。**

当前 methodology 内容不少，但分散在 `chapter3_methodology.md` 与 `chapter5_loss_function_design.md`。老师希望完整方法论，因此需要整合为一个自洽章节。

**建议处理**：

- `chapter5_loss_function_design.md` 不一定保留为单独 chapter，可以整合进 methodology 的 `Loss Function Design` 部分。
- 如果保留 loss design 作为单独章，也要确保老师要求的 methodology 章节已经完整覆盖所有流程。
- 最终不要出现“方法论不完整，loss design 另见后文但没有整体流程”的感觉。

### Gap 5：缺少 Conclusion

**严重程度：高。**

学校明确要求 conclusions。当前没有独立 conclusion chapter。

**建议写法**：

Conclusion 不需要很长，但必须独立收束：

- 研究问题回扣。
- 最主要发现：loss function choice materially affects portfolio performance。
- 当前主推荐：`m2_robust_gamma07`。
- 备选：`m2_robust_gamma10` 高 Sharpe 但波动高，`imadl_m2_alpha06` 稳定 fallback。
- 负面发现：IMADL+GMADL blend/adaptive 不够稳，normalization 不是通用修复。
- 限制：static sanity check、未计交易成本、非 rolling backtest、limited market regimes。
- future work：rolling window、transaction cost、market regime expansion、strict replication if required。

### Gap 6：缺少 References/Bibliography

**严重程度：高。**

学校明确要求 references。当前 chapter 里有 `[1]`, `[7, 8]` 等引用口径，但没有看到独立 bibliography。

**建议补齐**：

- 建立 `references.bib` 或最终 markdown/LaTeX bibliography。
- 统一 chapter2 中引用格式。
- 确保所有文中编号都能在 bibliography 中找到。
- 至少包含：Gu et al. 2020、MADL/GMADL 相关论文、CRSP/data source、robust loss/Huber、portfolio evaluation/Sharpe 等。

---

## 6. 实验口径一致性问题

这些是当前草稿最容易被导师或评审发现的问题，优先级很高。

### 6.1 测试窗口冲突

当前草稿多处写：

- training: 1990-01 to 1994-12
- main testing: 1995-01 to 1996-12
- 24-month controlled static evaluation
- early baseline sanity check: 1995-01 to 1995-06, only when explicitly labelled

但 Phase 1.5/2.1b/Phase 2.5 诊断材料显示，多数当前关键实验使用的是：

- testing: 1995-01 to 1996-12
- 24-month test window

**建议**：

- 不要简单全局替换。
- 按实验阶段写清楚：
  - 哪些早期 baseline 使用 6-month。
  - 哪些 Phase 1.5/2.1b/2.2 结果使用 24-month。
  - 最终主结果采用哪一个 test window。
- 最终 empirical tables 应以 24-month 结果为主；methodology 和 empirical results 中统一为 24-month，并把 6-month 描述限制在 early/preliminary baseline sanity check 语境。

### 6.2 模型结构冲突

当前 `chapter6_experimental_results.md` 写：

- 3-layer LSTM with 128 hidden units
- batch size 256
- 50 epochs with early stopping

但 Phase 2.5 诊断与当前实验总结更支持：

- MLP `[64, 32, 16]`
- `tanh`
- dropout `0.0`
- Adam learning rate `0.001`
- batch size `1024`
- 20 epochs

**建议**：

- 以实际 runner/config 为准修正所有章节。
- 如果历史上确实有 LSTM 表述，应删除或解释为早期计划而非最终实验。
- final report 不能同时出现 LSTM 和 MLP 作为“all experiments”的配置。

### 6.3 Seed 设置冲突

当前 `chapter6_experimental_results.md` 写 robustness seeds `{42, 123, 456}`。

但已有材料显示：

- Phase 1.5 robustness 常见 seeds 是 `{42, 52, 62}`。
- normalized experiments 使用 `{42, 123, 456}`。

**建议**：

- 在 methodology 中按 phase 列出 seed，而不是写一个统一 seed set。
- Results tables 旁边标注该表使用的 seed。

### 6.4 CV 和 Sharpe 数字冲突

当前 `chapter6` 概览写：

- `gamma07` optimal: Sharpe 0.92, CV 0.04
- normalization failed across all losses

但 Phase 2.2 integrated analysis 给出的主结果是：

- `m2_robust_gamma07`: mean Sharpe `0.9156`, CV `0.1808`
- `m2_robust_gamma10`: mean Sharpe `1.0043`, CV `0.5613`
- `imadl_m2_alpha06`: mean Sharpe `0.6895`, CV `0.2443`

Normalization 的更细结论是：

- `gamma07_normalized` 改善约 `+5.96%`。
- `gamma10` 和 `imadl_m2_alpha06` normalized 后下降。
- 因此 normalization 不是通用修复，但可以作为 `gamma07` optional enhancement。

**建议**：

- 用 `doc/phase2-fix/reports/phase2_2_integrated_analysis.md` 作为主数字来源。
- 修正所有 CV。
- 把 “normalization failed across all losses” 改成 “normalization is not a universal fix; it helps gamma07 but hurts gamma10 and alpha06”。

### 6.5 Phase 2.5 对齐诊断的表述边界

Final report 需要吸收 Phase 2.5 结论，否则容易过度 claim。

安全表述：

- Phase 2 是新 loss variants 的 controlled comparison，不是 Phase 1.5 exact replication。
- Phase 2 内部比较有效。
- `m2_robust_gamma07` 是 Phase 2 内部最稳妥主推荐。

避免表述：

- “Phase 2 M2 直接超越 Phase 1.5 M2。”
- “IMADL 在 Phase 2 中严格复现并提升。”
- “Phase 2 runner 完全数值复现 Phase 1.5。”

原因：

- M2 有 `lambda_dir` 口径差异：Phase 1.5 `lambda=5.0`，Phase 2.1b default `lambda=1.0`，Phase 2 常用 `lambda=2.0`。
- IMADL 有命名/公式碰撞：Phase 1.5 simple tanh-based IMADL vs Phase 2 rebalanced IMADL。
- GMADL 偏差不再视为阻塞 bug，但也不是强到可以 claim exact replication。

---

## 7. 推荐最终章节结构

### Chapter 1: Introduction

保留当前 chapter1 的主体，但修改：

- 删除/改写 `Chapter 4: Progress in Semester 1` 等 interim-style structure。
- 明确 final report 是完整研究，不是补充报告。
- Scope 中的 test period、model、seed 要与最终实验一致。
- Research objectives 可以保留，但要和最后 results 对齐。

### Chapter 2: Literature Review

当前 chapter2 基本可用，建议补强：

- loss function design 在 financial prediction 中为什么重要。
- robust loss/Huber/median loss 为什么适用于 heavy-tailed returns。
- Directional loss 的优点与局限。
- 本研究和 MADL/GMADL 的关系：不是简单复现，而是诊断后设计 hybrid loss。

### Chapter 3: Data

建议新建。内容来自当前 methodology 的 data/feature 部分，但要更像正式论文：

- data source
- sample construction
- train/test split
- feature definitions
- preprocessing
- descriptive limitations

### Chapter 4: Methodology

建议由当前 `chapter3_methodology.md` 和 `chapter5_loss_function_design.md` 整合而成。

核心结构：

1. Research design
2. Fixed model architecture
3. Loss functions
4. Training protocol
5. Portfolio construction
6. Evaluation metrics
7. Experimental phase design
8. Reproducibility and limitations

### Chapter 5: Empirical Results and Discussion

当前 `chapter6_experimental_results.md` 是主要材料来源。保留大部分实验表格和分析，但要：

- 修正配置。
- 修正 CV/Sharpe。
- 加入“哪个好/哪个差”的明确 discussion。
- 把 Phase 2.5 的结论作为限制和 robustness discussion，而不是另开一个不相关的诊断报告。

推荐 narrative：

1. Baseline and Phase 1.5 show robust/directional tradeoff。
2. Phase 2 shows IMADL+GMADL blend and adaptive variants are not reliable。
3. M2+robustness family is strongest。
4. Gamma refinement shows `gamma07` is best balance, `gamma10` is high-return/high-variance。
5. Normalization is not universal。
6. Alignment diagnostics constrain cross-phase claims but do not invalidate Phase 2 internal comparison。

### Chapter 6: Conclusion

新建。约 800-1200 words 比较稳妥，或者按学校模板 200-300 words 最低要求写短版。建议不要太短，因为 final report 需要收束研究贡献。

### References

新建并统一引用。

---

## 8. 当前已经做到的地方

可以保留并继续使用：

1. Introduction 的研究动机比较完整，已经清楚说明 loss function 对 trading objective 的意义。
2. Literature review 已经覆盖 algorithmic investment、testing framework、loss functions、GMADL 和本研究定位。
3. Methodology 草稿已经包含大量必要内容，只是需要拆分 data、统一配置、整合 loss design。
4. Empirical results 的材料量足够，符合老师“当前 empirical 这部分感觉已经足够”的判断。
5. Phase 1.5 到 Phase 2.5 的实验链条完整，能形成清楚的研究进展：
   - lambda sweep
   - multi-seed robustness
   - four hybrid families
   - gamma refinement
   - loss-scale diagnostics
   - alignment diagnostics
6. 已经有明确主结论：
   - 主推荐 `m2_robust_gamma07`
   - 高收益备选 `m2_robust_gamma10`
   - 稳定 fallback `imadl_m2_alpha06`

---

## 9. 还没有做到的地方：按优先级排序

### P0：必须在提交前完成

1. 补 front page、blank page、abstract、keywords、中文摘要。
2. 建立最终 report 结构，不再沿用 interim structure。
3. 新增或拆出 Data chapter。
4. 完整 Methodology chapter，覆盖数据、模型、loss、训练、portfolio、metrics、phases。
5. 修正所有实验配置冲突：test window、MLP/LSTM、batch size、epoch、seed。
6. 修正 `gamma07`、`gamma10`、`imadl_m2_alpha06` 的 Sharpe/CV 数字。
7. 新增 Conclusion。
8. 新增 References/Bibliography。

### P1：强烈建议完成

1. 把 `chapter6_experimental_results.md` 改名或标题改为 `Empirical Results and Discussion`。
2. 每个结果 subsection 都加一句明确判断：best/worst/why。
3. 在 results/discussion 中加入 Phase 2.5 的 safe-claim boundary。
4. 将 normalization 结论改成更细的版本：不是通用修复，但 gamma07 normalized 有改善。
5. 对所有图表编号、表格编号和引用进行统一。

### P2：有时间再优化

1. 精简过长的 phase 叙事，避免像实验日志。
2. 增加一张 final loss comparison summary table。
3. 增加一张 final methodology flowchart。
4. 增加 appendix 列出完整 experimental runs。
5. 补 AI-assisted technology disclosure，与导师确认是否需要正式声明。

---

## 10. 现阶段不建议继续投入太多时间的事

根据老师反馈“当前 empirical 这部分感觉已经足够”，不建议把主要时间继续放在新实验上，除非导师明确要求。

不优先做：

1. 大规模新增 experiments。
2. 重新严格复现 GMADL，除非导师要求 exact replication。
3. 把 normalization 扩展到所有 loss families。
4. 新增复杂 portfolio optimization。
5. 加入 transaction cost model 后重跑所有表格。

更应该做：

1. 把已有结果写得自洽、完整、可提交。
2. 修正口径冲突。
3. 完成学校要求的 front matter 和 references。
4. 把 methodology 写成完整独立章节。

---

## 11. 建议的最后写作顺序

### Step 1：锁定最终章节结构

先确定最终报告目录：

1. Introduction
2. Literature Review
3. Data
4. Methodology
5. Empirical Results and Discussion
6. Conclusion
7. References

### Step 2：修 thesis structure

修改 Chapter 1 的 `Thesis Structure`，删除 interim-style chapters。

### Step 3：拆 Data 与 Methodology

从当前 methodology 中拆出 Data chapter，然后把 loss design 合并回 Methodology。

### Step 4：修 empirical results 口径

重点修：

- LSTM -> MLP
- 6-month/24-month 统一
- seed sets
- CV values
- normalization conclusion
- Phase 2.5 safe claims

### Step 5：写 front matter 和 conclusion

补学校必需内容。

### Step 6：整理 references 与最终编译

确认所有引用都有 bibliography，最终导出 PDF 并按学校命名。

---

## 12. 可以直接向老师说明的当前状态

可以这样汇报：

> I have already completed the main empirical part of the project. The experiments cover lambda tuning, multi-seed robustness checks, four hybrid loss families, gamma refinement, normalization diagnostics, and alignment diagnostics. The strongest current result is that `m2_robust_gamma07` provides the best balance between Sharpe and stability, while `gamma10` achieves a higher Sharpe but with substantially higher seed sensitivity. The empirical evidence is sufficient for the final report, so my remaining work is mainly to reorganize the thesis into a standalone final report, complete the methodology and data sections, fix configuration consistency, and add the required abstract, conclusion, and references.

中文版本：

> 我目前实证部分的主要材料已经足够，已经完成了从参数搜索、多 seed 鲁棒性检查、四类 hybrid loss 设计、gamma 精调、normalization 诊断到 Phase 2.5 对齐诊断的完整实验链条。当前最稳妥的主结论是 `m2_robust_gamma07` 在 Sharpe 和稳定性之间取得最好平衡，`gamma10` 虽然 Sharpe 更高但 seed 波动明显更大。接下来主要工作不是继续大量做新实验，而是把论文整理成独立完整的 final report，补齐 data、methodology、conclusion、references 和学校要求的 front matter，并修正实验配置口径不一致的问题。

---

## 13. 最终提交前检查表

- [ ] 文件名符合 `StudentID ForenameSurname 2026.pdf`。
- [ ] 封面有英文题目、中文题目、姓名、学号、日期、导师。
- [ ] 封面后有空白页。
- [ ] 有英文 abstract。
- [ ] 有 2-6 个 English keywords。
- [ ] 有中文摘要。
- [ ] 有目录。
- [ ] 有 list of figures/list of tables，如文档包含图表。
- [ ] 正文超过约 6000 words。
- [ ] 有 Introduction。
- [ ] 有 Literature Review。
- [ ] 有 Data。
- [ ] 有完整 Methodology。
- [ ] 有 Empirical Results and Discussion。
- [ ] 有 Conclusion。
- [ ] 有 References。
- [ ] 所有实验配置一致。
- [ ] 所有表格数字与最终结果文件一致。
- [ ] 没有把 final report 写成 interim report 的补充。
- [ ] AI-assisted technology use 已按学校政策与导师确认。
