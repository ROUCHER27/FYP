# Phase 2.2 Success Criteria

**定义日期**: 2026-04-30

## 实验目标

Phase 2.2 的核心目标是在 Phase 2 P0 成功的基础上，通过精调和诊断进一步优化 top 3 losses。

## Gamma 精调成功标准

### 最低标准（必须达到）
- ✅ 至少 1 个新 gamma 值的 Sharpe > 0.70
- ✅ 所有新 gamma 值无 NaN/Inf 错误
- ✅ 找到 Sharpe vs gamma 的趋势（单调递增/先增后减/平稳）

### 目标标准（期望达到）
- 🎯 至少 1 个新 gamma 值的 Sharpe > 1.00（超越 gamma10）
- 🎯 或者找到 Sharpe-CV 更优的平衡点（例如 Sharpe 0.90, CV 0.35）
- 🎯 明确识别最优 gamma 值（峰值点）

### 理想标准（最佳情况）
- 🌟 找到 Sharpe > 1.10 且 CV < 0.50 的 gamma 值
- 🌟 理解 gamma 对 Sharpe 和 CV 的影响机制
- 🌟 给出最终推荐的 gamma 值及理论解释

## Loss-scale Diagnostics 成功标准

### 最低标准（必须达到）
- ✅ 成功记录 top 3 losses 的分量量级
- ✅ 计算量级比率（max_component / min_component）
- ✅ 无技术错误（脚本运行成功）

### 目标标准（期望达到）
- 🎯 **imadl_m2_alpha06**: 量级比率 < 10x（证明分量平衡）
- 🎯 **m2_robust_gamma10/01**: 理解 robustness penalty 的实际量级
- 🎯 如果发现失衡（比率 > 10x），明确是否需要归一化

### 理想标准（最佳情况）
- 🌟 量级比率 < 5x（完美平衡）
- 🌟 解释为什么 alpha=0.6 是最优（通过量级分析）
- 🌟 如果需要归一化，实现并验证归一化版本

## Phase 2.1b Alignment 成功标准

**重要说明**: Phase 2.1b alignment 发现 Phase 2 M2 (λ_dir=2.0) 与 Phase 1.5 M2 (λ_dir=5.0) 是不同的 loss variants，不应强行对齐。

### 最低标准（必须达到）
- ✅ 成功运行 IMADL, GMADL, hybrid_mul × 3 seeds
- ✅ 无技术错误（9 runs 全部完成）

### 目标标准（期望达到）
- 🎯 **IMADL 和 GMADL 对齐误差 < 15%**（前提：loss 配置严格等价）:
  - IMADL Sharpe: 0.464 ± 15% = [0.39, 0.53]
  - GMADL Sharpe: 0.307 ± 15% = [0.26, 0.35]
- 🎯 证明 Phase 2 runner 与 Phase 1.5 一致

**M2 对齐说明**：
- Phase 1.5 M2 使用 λ_dir=5.0，Sharpe=0.914
- Phase 2 M2 使用 λ_dir=2.0，这是一个新的 loss variant
- Phase 2.1b hybrid_mul 使用 λ_dir=1.0（默认值）
- **这三个是不同的 loss functions，不应直接比较**

**重要**：在判断 alignment 失败前，必须先确认：
1. Phase 1.5 target 数字的来源（lambda sweep / robustness / cap/no-cap）
2. Phase 2.1b 跑的 loss 配置与 Phase 1.5 严格等价（例如 M2 的 lambda_dir 参数）
3. 只有确认同口径、同配置后，偏差 >15% 才判定为 runner bug

### 理想标准（最佳情况）
- 🌟 IMADL 和 GMADL 对齐误差 < 10%（几乎完美复现）
- 🌟 CV 也在合理范围内（与 Phase 1.5 接近）

## 失败标准（触发重新评估）

### Gamma 精调失败
- ❌ 所有新 gamma 值的 Sharpe < 0.70
- ❌ CV > 1.0（不稳定）
- ❌ 无法找到明确的 Sharpe vs gamma 趋势

**后果**: 停止 gamma 精调，使用 gamma=1.0 作为最终推荐

### Diagnostics 失败
- ❌ 量级比率 > 20x（严重失衡）
- ❌ 脚本运行失败，无法获取诊断数据

**后果**: 必须实现归一化版本，重跑实验

### Alignment 失败
- ❌ 确认同口径、同配置后，IMADL 或 GMADL 对齐误差仍 > 15%

**后果**: 
1. 先检查 Phase 1.5 target 来源和 loss 配置是否严格等价
2. 如果配置不等价，修正 alignment 实验配置，重跑 Phase 2.1b
3. 如果配置等价但仍偏差 >15%，则判定为 runner bug，修复后重跑所有 Phase 2 实验

**M2 对齐失败说明**：
- Phase 2.1b 发现 Phase 2 M2 (λ_dir=2.0) 与 Phase 1.5 M2 (λ_dir=5.0) 参数不同
- 这不是 runner bug，而是不同的 loss variants
- Phase 2 M2 应被视为新的 loss 探索，不应强行对齐 Phase 1.5 M2

## 决策规则

### Gamma 精调后的决策
```
IF gamma=1.5 Sharpe > gamma=1.0 Sharpe:
    → 继续测试 gamma=2.0, 2.5（寻找峰值）
ELSE IF gamma=0.5-0.7 有更好的 Sharpe-CV 平衡:
    → 采用该 gamma 值作为最终推荐
ELSE IF gamma=1.0 是峰值:
    → 确认 gamma=1.0 为最优，Phase 2 收尾
```

### Diagnostics 后的决策
```
IF 量级比率 < 5x:
    → 无需归一化，当前结果可信
ELSE IF 量级比率 5-10x:
    → 建议归一化，但可选（对比实验）
ELSE IF 量级比率 > 10x:
    → 必须归一化，重跑实验
```

### Alignment 后的决策
```
IF IMADL 和 GMADL 对齐误差 < 15% (且配置严格等价):
    → Runner 可信，继续 Phase 2.2
ELSE IF 配置不等价（例如 lambda_dir 不同）:
    → 修正 alignment 配置，重跑 Phase 2.1b
ELSE IF 配置等价但偏差 > 15%:
    → 修复 runner，重跑所有 Phase 2 实验
```

**Phase 2.2 结果使用原则**：
- Gamma refinement 和 loss-scale diagnostics 可以继续跑（内部可比性有效）
- 但在 alignment cleanup 完成前，Phase 2.2 结果只能作为"Phase 2 runner 内部探索"
- 只有 alignment 通过后，才能声称"相对 Phase 1.5 的确定提升"

**M2 特殊说明**：
- Phase 2 M2 (λ_dir=2.0) 是新的 loss variant，不需要对齐 Phase 1.5 M2 (λ_dir=5.0)
- Phase 2.2 的 m2_robust_gamma07 结果有效，但应描述为"新 M2 variant 的探索"
- 不应声称"Phase 2 M2 超越 Phase 1.5 M2"（因为参数不同）

## 时间预算

- **Gamma 精调**: 4 gamma × 3 seeds = 12 runs, ~8 hours
- **Diagnostics**: 3 losses × 1 run = 3 runs, ~1 hour
- **Alignment**: 3 losses × 3 seeds = 9 runs, ~9 hours
- **总计**: 24 runs, ~18 hours

## 最终交付物

Phase 2.2 完成后，应产出：
1. **Gamma 精调报告**: 最优 gamma 值及 Sharpe vs gamma 曲线
2. **Diagnostics 报告**: Top 3 losses 的量级分析
3. **Alignment 报告**: Phase 2 runner 与 Phase 1.5 的对齐验证
4. **最终推荐**: Phase 2 最佳 loss 函数及参数配置

## 论文贡献

Phase 2.2 成功后，论文可以声称：
- ✅ 设计了有效的 robustness penalty（Variant 3）
- ✅ 发现了 IMADL+M2 的最优组合比例（alpha=0.6）
- ✅ 通过系统性实验找到了最优 gamma 值
- ✅ 验证了实验基础设施的可靠性（alignment）
- ✅ 理解了组合 loss 的内部机制（diagnostics）

**核心贡献**: 不仅提出了新 loss，还通过严格的实验验证了其有效性和稳定性。
