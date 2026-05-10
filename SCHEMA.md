## principles
1. final report 独立成章. 可以参考doc/2253235_YirongYu_2025.pdf 中期报告现有内容，可以重复或直接复用原文，但是不要被它干扰，不要做出对它的引用性表达。
2. 仅需把已有 thesis 改成一篇独立、完整、学校格式合规、实验口径一致的 final report。
3. `SCHEMA.md` 是后续写 report 的最高层准则。任何 agent 在写 final report 前都必须先读本文件，再读 `doc/thesis/*` 草稿和对应实验结果。
4. 不允许把阶段性诊断文档、Colab 运行说明、旧 6-month sanity check 当作 final report 的主实证结果。主文只引用通过同口径验证的 final evidence。
5. 当前不需要管`front_matter.md` 或 LaTeX front matter，仅使用md格式以及符合obsidian中数学块，代码块的格式即可；最后收尾我会再转到latex编排

## 1. report structure
当前 `doc/thesis` 的章节结构仍带有 interim/phase report 痕迹。最终改成：

1. `Chapter 1: Introduction`
2. `Chapter 2: Literature Review`
3. `Chapter 3: Methodology`
4. `Chapter 4: Data`
5. `Chapter 5: Empirical Results and Discussion`
6. `Chapter 6: Conclusion`

| 建议文件                                       | 来源/作用                                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `chapter4_data.md`                         | 从当前 `chapter3_methodology.md` 拆出 Data, 此外还有额外对`chapter5_empirical_results部分需要用到的数据进行整理，介绍，用于支撑的最终论证。 |
| `chapter3_methodology.md`                  | 由当前 `chapter3_methodology.md` 的方法部分 + `chapter5_loss_function_design.md` 整合                          |
| `chapter5_empirical_results_discussion.md` | 由当前 `chapter6_experimental_results.md` 改名/重构                                                         |
| `chapter6_conclusion.md`                   | 新写 conclusion                                                                                        |
| `references.bib` 或 `references.md`         | 新建 bibliography                                                                                      |


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
| normalization | `chapter6` 写 normalization failed across all losses | 改成：不是通用修复；gamma07 normalized 基本持平或略降，gamma10/alpha06 明显下降 |
| baseline evidence | 旧草稿混用 6-month 和 24-month 数字 | 使用 `doc/final_report_all_24m_evidence/` 作为同口径 24-month single-seed baseline/Phase 1.5 evidence |

### 2.3 统一 final conclusion

最终报告的主结论建议固定为：

- 主推荐：`m2_robust_gamma07`，Mean Sharpe `0.9156`，CV `0.1808`，Sharpe 和稳定性平衡最好。
- 高收益备选：`m2_robust_gamma10`，Mean Sharpe `1.0043`，但 CV `0.5613`，seed 波动大。
- 稳定 fallback：`imadl_m2_alpha06`，Mean Sharpe `0.6895`，CV `0.2443`。

`m2_robust_gamma07`/`gamma10` 的 gamma-refinement source 是 `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`。`imadl_m2_alpha06` fallback 的 Mean Sharpe/CV 来自 `phase2.2-fix` 分支的 integrated summary：`git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`。不要用本地 5-row gamma refinement 表去支撑 `imadl_m2_alpha06` 的 CV。

- normalization 结论：不是通用修复；当前证据中 gamma07 normalized 与 original 基本持平但略低，gamma10/alpha06 normalized 明显下降，不能说它全面解决 scale imbalance。

### 2.4 final evidence gate

任何进入 final report 主表、主图、abstract、conclusion 的数字都必须满足：

1. 结果文件有明确来源路径：baseline/Phase 1.5 single-seed 表优先使用 `doc/final_report_all_24m_evidence/`；multi-seed/robustness 表优先使用 `doc/phase2-fix/*/reports/`、`doc/phase2-fix/*/results/`；`doc/final_report_24m_baselines/` 只作为旧 MSE/MedSE 交叉核对材料。
2. 主实证窗口必须是 `1995-01` 到 `1996-12`，共 24 行月度结果；早期 6-month 结果只能标注为 preliminary sanity check。
3. 训练窗口必须是 `1990-01` 到 `1994-12`，除非表标题明确声明不同设置。
4. 结果对应的 command/manifest/log/run_spec 必须能说明 seed、loss、batch size、epoch、portfolio cap、commit 或 runner。
5. 如果两个表来自不同 phase，不能默认可直接比较；必须先检查 loss formula、lambda/gamma/alpha、seed set、portfolio construction、feature set、model config 是否一致。
6. Phase 2 是新 variants 的内部比较，不是 Phase 1.5 exact replication。跨 phase 的 improvement claim 必须谨慎写成 controlled comparison 或 motivation chain。

### 2.5 latest 24-month same-window evidence

`doc/final_report_all_24m_evidence/` 当前已验证完整同口径 24-month evidence，在 `main` 提交 `6c0fbde` 下运行完成：

- Baseline losses: `mse`, `medse`, `madl`, `gmadl`, `imadl`, `hybrid_mul_m1`, `hybrid_mul_m2`.
- Phase 1.5 variants: `A1`-`A5` (`hybrid_add_a1`-`hybrid_add_a5`) and `M1`-`M4` (`hybrid_mul_m1`-`hybrid_mul_m4`).
- All runs: train `1990-01..1994-12`, test `1995-01..1996-12`, `test_months=24`, seed `42`, batch size `1024`, max epochs `20`.
- All runs have verification manifests with 24 rows from `1995-01` through `1996-12`.

这批结果可以作为 final report 开始正式撰写 baseline 和 Phase 1.5 同口径表的主要证据。注意：它是 single-seed 同口径 evidence，不是 multi-seed robustness evidence；multi-seed 稳定性仍应引用 Phase 2.2 / phase2-fix 的 grouped summaries。旧草稿中 “MedSE Sharpe 2.68 dominates MSE 0.37” 不能作为 24-month final headline。

## 3. 按现有文件逐项修改

## 3.1 `chapter1_introduction.md`

### 保留

- 1.1 Background and Motivation 的整体方向可以保留。
- 1.2 Research Gap 可以保留。
- 1.3 Research Objectives 基本可保留。
- 1.4 Research Questions 基本可保留。

### 必须修改

1. **删除旧 thesis structure。**
2. 不用删除thesis文档，只需新建final report文档在其中体现出来即可

当前 1.6 仍写：
- Chapter 4: Progress in Semester 1
- Chapter 5: Knowledge Gained and Skills Required
- Chapter 6: Hybrid Loss Function Design
- Chapter 7: Experimental Results
- Chapter 8: Conclusion and Future Work
和我们的structure不符合，且当前描述过于冗长


## 4. 需要新增的内容
## 4.2 `chapter4_data.md`

必须新增或拆出。老师明确说 final report 包含 data。

建议标题：

`# Chapter 4: Data`

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
