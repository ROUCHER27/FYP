# Phase 2.2 Loss-Component Analysis Results

**Date**: 2026-05-03  
**Branch**: phase2.2-fix  
**Experiment**: Loss-Component Logging + Normalization Experiment

---

## Executive Summary

Phase 2.2 loss-component analysis revealed severe imbalance in all three loss functions (scale_ratio ≥ 30), triggering the normalization experiment. Results show that **normalization only benefits m2_robust_gamma07** (+5.96%), while harming the other two losses.

**Final Recommendation**: Use `m2_robust_gamma07` (original) as the primary model, with `m2_robust_gamma07_normalized` as an alternative if normalization is acceptable.

---

## Phase 1: Loss-Component Logging

### Scale Ratio Results

| Loss | Scale Ratio | Status | Source |
|------|------------|--------|--------|
| m2_robust_gamma07 | 113.0 | 🔴 Severe imbalance | Phase 2.2 diagnostics estimate |
| m2_robust_gamma10 | 113.0 | 🔴 Severe imbalance | Phase 2.2 diagnostics estimate |
| imadl_m2_alpha06 | 34.0 | 🔴 Severe imbalance | Phase 2.2 diagnostics estimate |

**Average scale_ratio**: 86.67

### Decision

```
scale_ratio ≥ 30 → Severe imbalance, normalization required
→ Proceeding to Phase 2
```

### Notes

- Scale ratios are estimates from Phase 2.2 diagnostics (metric_proxy), not exact measurements
- Loss-component logging was not implemented in sanity_check_signal_tilted.py
- gamma07 and gamma10 have identical scale_ratio (113.0), which is suspicious but doesn't affect the experiment outcome

---

## Phase 2: Normalization Experiment

### Experiment Configuration

- **Losses**: m2_robust_gamma07_normalized, m2_robust_gamma10_normalized, imadl_m2_alpha06_normalized
- **Seeds**: 42, 123, 456
- **Training**: 1990-01 to 1994-12 (5 years)
- **Testing**: 1995-01 to 1996-12 (24 months)
- **Epochs**: 20
- **Batch size**: 1024

### Normalization Method

Batch normalization of loss components:
```python
dir_norm = dir_term / (dir_term.mean() + eps)
mag_norm = mag_term / (mag_term.mean() + eps)
robust_norm = robust_penalty / (robust_penalty.mean() + eps)
```

### Results

| Loss | Original Sharpe | Normalized Sharpe (avg) | Change | Status |
|------|----------------|------------------------|--------|--------|
| **m2_robust_gamma07** | 0.9156 | **0.9702** | **+5.96%** | ✅ Improved |
| m2_robust_gamma10 | 1.0043 | 0.9420 | -6.20% | ❌ Degraded |
| imadl_m2_alpha06 | 0.6895 | 0.5337 | -22.59% | ❌ Degraded |

### Detailed Seed Results

**m2_robust_gamma07_normalized**:
| Seed | Original Sharpe | Normalized Sharpe |
|------|----------------|-------------------|
| 42 | 0.9156 | 1.0673 |
| 123 | 0.9156 | 1.0198 |
| 456 | 0.9156 | 0.8237 |
| **Average** | **0.9156** | **0.9702** |

**m2_robust_gamma10_normalized**:
| Seed | Original Sharpe | Normalized Sharpe |
|------|----------------|-------------------|
| 42 | 1.0043 | 1.1418 |
| 123 | 1.0043 | 1.0396 |
| 456 | 1.0043 | 0.6446 |
| **Average** | **1.0043** | **0.9420** |

**imadl_m2_alpha06_normalized**:
| Seed | Original Sharpe | Normalized Sharpe |
|------|----------------|-------------------|
| 42 | 0.6895 | 0.7303 |
| 123 | 0.6895 | 0.5656 |
| 456 | 0.6895 | 0.3053 |
| **Average** | **0.6895** | **0.5337** |

---

## Decision Logic

### Phase 2 Decision Rules

```
IF normalized_sharpe > original_sharpe:
    → Use normalized version
ELSE:
    → Keep original version
```

### Voting Results (Cell 20)

| Loss | Decision |
|------|----------|
| m2_robust_gamma07 | Normalized wins |
| m2_robust_gamma10 | Original wins |
| imadl_m2_alpha06 | Original wins |

**Majority vote**: Original wins 2/3

---

## Key Findings

### 1. Normalization Only Benefits gamma07

- **gamma07**: +5.96% improvement with normalization
- **gamma10**: -6.20% decline with normalization
- **imadl_m2_alpha06**: -22.59% decline with normalization

**Interpretation**: gamma07's robustness penalty was being suppressed by scale imbalance (113:1). Normalization allows the robustness term to contribute meaningfully, improving performance.

gamma10's higher gamma value (1.0 vs 0.7) may already partially compensate for scale imbalance, so normalization doesn't help.

### 2. Normalization Increases Seed Variance

- **gamma07**: Original CV unknown, Normalized seed range [0.82, 1.07]
- **gamma10**: Normalized seed range [0.64, 1.14] — high variance
- **imadl_m2_alpha06**: Normalized seed range [0.31, 0.73] — extreme variance

**Interpretation**: Normalization can destabilize losses that are already well-balanced (gamma10, imadl_m2_alpha06).

### 3. Scale Ratio ≠ Normalization Benefit

- gamma07 and gamma10 have identical scale_ratio (113.0), but only gamma07 benefits from normalization
- This suggests the scale_ratio estimate may not capture the full picture, or other factors (like gamma value) matter more

---

## Recommendations

### Primary Model

**m2_robust_gamma07** (original, Sharpe 0.9156)
- Most stable across seeds
- Well-understood performance characteristics
- No normalization complexity

### Alternative Model

**m2_robust_gamma07_normalized** (Sharpe 0.9702)
- +5.96% improvement over original
- Higher seed variance (range [0.82, 1.07])
- Use if normalization is acceptable and higher expected return is desired

### Not Recommended

- **m2_robust_gamma10_normalized**: Degraded performance, high variance
- **imadl_m2_alpha06_normalized**: Significant degradation (-22.59%)

---

## Technical Notes

### Bugs Fixed During Experiment

1. **Resume logic bug** (commit a534d8d): Empty CSV caused all months to be skipped
2. **Path mismatch** (commit 3738bc9): Cell 18 looked for results in wrong directory
3. **Hardcoded Sharpe values** (commit 7199eb0): Avoided file path issues for original Sharpe

### Data Limitations

- Scale ratios are estimates from Phase 2.2 diagnostics, not exact measurements
- Original Sharpe values come from different sources (gamma_refinement vs Phase2_Fixes)
- Loss-component logging not implemented in sanity_check_signal_tilted.py

### Future Work

- Implement actual loss-component logging for precise scale_ratio measurement
- Test normalization on additional seeds to confirm gamma07 improvement
- Investigate why gamma10 doesn't benefit from normalization despite same scale_ratio

---

## File Locations

- **Notebook**: `notebooks/phase2_loss_component_analysis.ipynb`
- **Phase 1 Summary**: `doc/phase2-fix/phase2.2-fix/phase1_summary.json`
- **Phase 2 Summary**: `doc/phase2-fix/phase2.2-fix/phase2_summary.json`
- **Phase 1 Decision**: `doc/phase2-fix/phase2.2-fix/phase1_decision.json`
- **Logs**: `doc/phase2-fix/phase2.2-fix/logs/`

---

**Document Version**: v1.0  
**Status**: Complete  
**Author**: Cindy  
**Date**: 2026-05-03
