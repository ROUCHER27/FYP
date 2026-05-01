# Phase 2 Shortlist (Frozen for Phase 2.2)

**冻结日期**: 2026-04-30

## Top 3 Candidates

### 1. m2_robust_gamma10
- **Sharpe**: 1.0043 ± 0.5638
- **CV**: 0.5613
- **Cumulative Return**: 23.68%
- **配置**: M2 + variance penalty (gamma=1.0)
- **优势**: 最高 Sharpe，显著超越 Phase 1.5 所有 baseline
- **劣势**: CV 较高，seed 间波动较大

### 2. m2_robust_gamma01
- **Sharpe**: 0.7470 ± 0.3937
- **CV**: 0.5270
- **Cumulative Return**: 27.18%
- **配置**: M2 + variance penalty (gamma=0.1)
- **优势**: Sharpe 和 CV 都较好，更稳定
- **劣势**: Sharpe 略低于 gamma10

### 3. imadl_m2_alpha06
- **Sharpe**: 0.6895 ± 0.1685
- **CV**: 0.2443 ⭐ 最稳定
- **Cumulative Return**: 30.42%
- **配置**: 60% IMADL + 40% M2
- **优势**: CV 最低，非常稳定，收益率最高
- **劣势**: Sharpe 略低于前两者

## Dropped Variants

### Variant 2: IMADL+GMADL (全部失败)
- **imadl_gmadl_beta03**: Sharpe -0.0345
- **imadl_gmadl_beta05**: Sharpe 0.0406
- **imadl_gmadl_beta07**: Sharpe -0.0020
- **原因**: 所有 beta 值均接近 0 或负 Sharpe，R2 极差（-300 到 -600），组合无效

### Variant 4: Adaptive Hybrid (不稳定)
- **adaptive_lambda10**: Sharpe 0.4938, CV 1.5426
- **adaptive_lambda50**: Sharpe 0.2763, CV 0.5780
- **adaptive_lambda100**: Sharpe 0.0955, CV 0.3591
- **原因**: lambda10 虽然 Sharpe 接近 0.5，但 CV > 1.5 极不稳定；其他 lambda 值 Sharpe 过低

### Variant 1: IMADL+M2 低 alpha 值
- **imadl_m2_alpha02**: Sharpe 0.1788, CV 6.47
- **imadl_m2_alpha03**: Sharpe 0.2159, CV 0.93
- **imadl_m2_alpha04**: Sharpe 0.3540, CV 0.19
- **原因**: alpha < 0.5 时 Sharpe 显著低于 alpha06，不如直接使用 alpha06

## Rationale

### 为什么只保留 Top 3？
1. **专注有效机制**: Variant 3 (robustness penalty) 和 Variant 1 (IMADL+M2, alpha≥0.5) 已验证有效
2. **避免算力浪费**: Variant 2 和 Adaptive 已证明无效或不稳定
3. **保持可解释性**: Top 3 都有清晰的机制解释，便于论文撰写

### Phase 2.2 计划
- **Gamma 精调**: 围绕 m2_robust 测试 gamma=0.3, 0.5, 0.7, 1.5
- **Loss-scale diagnostics**: 验证 imadl_m2_alpha06 的分量平衡
- **不再测试**: IMADL+GMADL, Adaptive, 低 alpha 值的 IMADL+M2

## 与 Phase 1.5 对比

| Loss | Phase 1.5 Sharpe | Phase 2 Sharpe | 提升 |
|------|------------------|----------------|------|
| IMADL | 0.464 | - | - |
| M2 (hybrid_mul) | 0.914 | - | - |
| **m2_robust_gamma10** | - | **1.0043** | **+10%** |
| **m2_robust_gamma01** | - | **0.7470** | - |
| **imadl_m2_alpha06** | - | **0.6895** | - |

**结论**: Phase 2 P0 修复成功，m2_robust_gamma10 超越 Phase 1.5 最佳结果。
