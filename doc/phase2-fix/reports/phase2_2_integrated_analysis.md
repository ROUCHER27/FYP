# Phase 2.2 Integrated Analysis

生成日期：2026-05-03

## 结论摘要

这轮 Phase 2.2 实验已经跑完整，结果文件也完整下载到了本地。技术完成度上，Phase 2.1b alignment 为 9/9 successful，loss-scale diagnostics 为 3/3，gamma refinement 为 15/15 successful。

实验结论要分两层看：

1. **在 Phase 2 runner 内部比较中，`m2_robust_gamma07` 是当前最好的综合选择。** 它的 mean Sharpe 为 0.9156，CV 为 0.1808，明显比 `gamma10` 更稳定。`gamma10` 的 mean Sharpe 最高，为 1.0043，但 CV 为 0.5613，seed 间波动更大。
2. **暂时不能直接声称 Phase 2 已经严格超越 Phase 1.5。** Alignment 对比中 3 个 baseline 的均值偏差都超过 15% 阈值：IMADL +17.64%，GMADL +62.38%，hybrid_mul -28.13%。这不一定立刻说明 runner 有 bug，因为 criteria 已经要求先核对 Phase 1.5 target 的来源和 loss 配置是否严格等价；但在核对完成前，Phase 2.2 只能作为内部探索结果使用。

## 数据范围

| 模块 | 路径 | 完成情况 |
| --- | --- | --- |
| Phase 2 P0 baseline | `doc/phase2-fix/reports/phase2_grouped_summary.csv` | 16 losses x 3 seeds = 48 runs |
| Phase 2.1b alignment | `doc/phase2-fix/phase2_1b/reports/phase21b_vs_phase15_grouped.csv` | 3 losses x 3 seeds = 9 runs |
| Phase 2.2 loss-scale diagnostics | `doc/phase2-fix/phase2_2/loss_scale/reports/loss_scale_summary.csv` | 3 losses |
| Phase 2.2 gamma refinement | `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv` | 5 gamma values x 3 seeds = 15 runs |

`doc/phase2-fix/phase2_2/latest_status.json` 记录了 alignment summaries = 9、diagnostic summaries = 3、gamma refinement summaries = 15。

## 1. Phase 2.1b Alignment

Alignment 的技术运行是成功的，但统计对齐没有通过 15% 阈值。

| Loss | Phase 1.5 target Sharpe | Phase 2.1b observed mean | Deviation | 判断 |
| --- | ---: | ---: | ---: | --- |
| `imadl` | 0.464 | 0.5458 | +17.64% | 超过 15% |
| `gmadl` | 0.307 | 0.4985 | +62.38% | 明显超过 15% |
| `hybrid_mul` | 0.914 | 0.6569 | -28.13% | 超过 15% |

需要注意两点：

- `hybrid_mul` 的 seed 52 和 seed 62 实际接近 target，分别偏差 -3.56% 和 +2.83%；均值失败主要来自 seed 42 的 Sharpe 只有 0.1495。
- `imadl` 和 `gmadl` 的三个 seed 波动也很大，说明这组 3-seed alignment 本身对单个 seed 很敏感。

我的判断：**alignment 目前是黄灯，不是红灯。** 它证明了 Phase 2 runner 能完整跑完 baseline，但还没有证明与 Phase 1.5 是同口径复现。下一步应该先核对 Phase 1.5 target 的数据来源、cap 设置、seed 设置、`hybrid_mul` / M2 的 `lambda_dir` 参数，以及 train/test window 是否完全一致。只有这些确认一致后，偏差 >15% 才能判定为 runner bug。

## 2. Gamma Refinement

Gamma refinement 的核心结果如下：

| Loss | Mean Sharpe | Std | CV | Mean cumulative return |
| --- | ---: | ---: | ---: | ---: |
| `m2_robust_gamma03` | 0.3234 | 0.3418 | 1.0570 | 0.0818 |
| `m2_robust_gamma05` | 0.7054 | 0.1488 | 0.2109 | 0.2392 |
| `m2_robust_gamma07` | 0.9156 | 0.1655 | 0.1808 | 0.2799 |
| `m2_robust_gamma10` | 1.0043 | 0.5638 | 0.5613 | 0.2368 |
| `m2_robust_gamma15` | 0.8163 | 0.3724 | 0.4562 | 0.2277 |

趋势很清楚：Sharpe 从 gamma=0.3 到 1.0 上升，gamma=1.5 回落。按 Phase 2.2 criteria，`gamma15` 没有超过 `gamma10`，所以不需要继续扩到 gamma=2.0 / 2.5。真正值得关注的是 gamma=0.7，因为它牺牲了少量 mean Sharpe，但换来了非常大的稳定性提升：

- `gamma07`: Sharpe 0.9156，CV 0.1808，cumulative return 0.2799
- `gamma10`: Sharpe 1.0043，CV 0.5613，cumulative return 0.2368

如果论文叙事优先强调“稳定、鲁棒、可复现”，我建议把 **`m2_robust_gamma07` 作为主推荐**，把 `m2_robust_gamma10` 作为高 Sharpe 但高波动的备选。`gamma05` 可以作为保守备选，因为 Sharpe 已超过 0.70 且 CV 低，但它明显弱于 `gamma07`。

## 3. Loss-Scale Diagnostics

Diagnostics 跑通了，但结果显示三个代表 loss 都有 severe imbalance：

| Loss | Scale ratio | Largest component | Smallest component | 判断 |
| --- | ---: | --- | --- | --- |
| `imadl_m2_alpha06` | 34.36x | `r2` | `medse` | severe imbalance |
| `m2_robust_gamma01` | 113.07x | `directional_accuracy` | `medse` | severe imbalance |
| `m2_robust_gamma10` | 113.69x | `directional_accuracy` | `medse` | severe imbalance |

这里要谨慎解释：当前 diagnostics 的 `diagnostic_type` 是 `metric_proxy`，它不是训练时 loss term 的精确逐项分解，而是用输出指标做量级代理。因此它足够说明“不同目标信号的量级差异很大”，但不能直接证明某个 loss term 在反向传播中一定被完全淹没。

尽管如此，按照 Phase 2.2 criteria，>10x 已经触发“应考虑归一化”的规则。我的建议是：

- 不要马上否定当前 gamma refinement，因为内部比较仍然有效。
- 下一阶段应该加一个真正的 loss-component logger，记录训练时方向项、幅度项、robustness penalty 的 batch-level mean/std。
- 如果真实 loss term 也呈现 >10x 失衡，再做 normalized variant，不要只根据 metric proxy 改公式。

## 4. 与 Phase 2 P0 的关系

P0 阶段已经确定的 shortlist 是：

| Loss | P0 Mean Sharpe | P0 CV | 角色 |
| --- | ---: | ---: | --- |
| `m2_robust_gamma10` | 1.0043 | 0.5613 | 最高 Sharpe |
| `m2_robust_gamma01` | 0.7470 | 0.5270 | 早期低 gamma 代表 |
| `imadl_m2_alpha06` | 0.6895 | 0.2443 | 稳定 alpha 混合代表 |

Phase 2.2 改变了这个排序：`gamma07` 没有超过 `gamma10` 的 mean Sharpe，但在稳定性上显著更好，也超过了原先的 `gamma01`。因此 shortlist 应更新为：

1. `m2_robust_gamma07`：主推荐，Sharpe 与稳定性最均衡。
2. `m2_robust_gamma10`：最高 Sharpe 备选，适合强调上限。
3. `imadl_m2_alpha06`：保留作为机制不同、稳定性较高的组合 loss 对照组。

`m2_robust_gamma01` 可以降级为历史对照，不再作为主候选。

## 5. 是否达到 Phase 2.2 Criteria

| Criteria | 结果 |
| --- | --- |
| Gamma 至少 1 个新值 Sharpe > 0.70 | 通过：`gamma05`, `gamma07`, `gamma15` |
| Gamma 找到趋势 | 通过：0.3 -> 1.0 上升，1.5 回落 |
| Gamma 至少 1 个新值 Sharpe > 1.00 | 未通过：最高新值是 `gamma07` 0.9156 |
| 找到 Sharpe-CV 更优平衡点 | 通过：`gamma07` Sharpe 0.9156 / CV 0.1808 |
| Loss-scale 记录 top losses | 通过 |
| Loss-scale ratio < 10x | 未通过：三个都 > 30x |
| Alignment 9 runs 完成 | 通过 |
| Alignment 偏差 < 15% | 未通过，需要核对口径后判断原因 |

总体评价：**Phase 2.2 对 gamma 选择是成功的，对 diagnostics 和 alignment 暴露了下一步必须收口的问题。**

## 6. 最终建议

短期先做两个动作：

1. **把 `m2_robust_gamma07` 加入最终候选，并作为当前主推荐。** 论文可以写成：gamma=0.7 在 Sharpe 与 stability 之间取得最优折中，而 gamma=1.0 代表收益上限但 seed variability 更高。
2. **做一次 alignment cleanup，不要直接进入最终结论。** 先核对 Phase 1.5 target 来源和配置等价性，特别是 `hybrid_mul` seed 42 的异常低 Sharpe。如果发现配置不等价，修正后重跑 9 个 alignment；如果配置等价，再把它当 runner/数据切分 bug 处理。

中期可以加一个小实验：

- 增加真实 loss-component logging，而不是 metric proxy。
- 只对 `m2_robust_gamma07`, `m2_robust_gamma10`, `imadl_m2_alpha06` 记录训练时各 loss term 的 mean/std。
- 若真实 term ratio 仍 >10x，再设计 normalized version；否则保留当前实现，避免无谓改公式。

## 可直接写入论文的表述

在第二阶段的稳健性惩罚精调中，`m2_robust_gamma07` 在 3 个随机种子下取得 mean Sharpe 0.9156、CV 0.1808、mean cumulative return 0.2799，表现出比 `gamma10` 更稳定的风险收益特征。`gamma10` 的 mean Sharpe 最高，为 1.0043，但 CV 达到 0.5613，说明其对随机种子更敏感。因此，若以稳健性和可复现性为优先目标，`gamma=0.7` 是更合适的最终参数；若强调收益上限，`gamma=1.0` 可作为备选。

同时，alignment 实验显示 Phase 2 runner 与 Phase 1.5 target 尚未完全同口径对齐，三个 baseline 的均值偏差均超过 15%。因此，Phase 2.2 的结论当前应被表述为 runner 内部比较结果；在核对 Phase 1.5 target 来源和 loss 配置等价性后，才能进一步声称其相对 Phase 1.5 的确定提升。
