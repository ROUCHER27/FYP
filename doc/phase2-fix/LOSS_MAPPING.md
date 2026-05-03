# Phase 1.5 vs Phase 2 Loss Function Mapping

**Purpose**: Clarify the relationship between Phase 1.5 and Phase 2 loss functions, especially M2 variants.

**Date**: 2026-05-03  
**Author**: Cindy

---

## Critical Finding: M2 Parameter Inconsistency

Phase 1.5 and Phase 2 use **different λ_dir parameters** for M2, making them distinct loss variants that should not be directly compared.

---

## Loss Function Mapping

### 1. IMADL (Identical)

| Phase | Loss Name | Implementation | λ_dir | Status |
|-------|-----------|----------------|-------|--------|
| Phase 1.5 | IMADL | `imadl_loss()` | N/A | ✅ Identical |
| Phase 2 | IMADL | `imadl_loss()` | N/A | ✅ Identical |

**Alignment**: Should align perfectly (within random seed variance)

---

### 2. GMADL (Identical)

| Phase | Loss Name | Implementation | λ_dir | Status |
|-------|-----------|----------------|-------|--------|
| Phase 1.5 | GMADL | `gmadl_loss()` | N/A | ✅ Identical |
| Phase 2 | GMADL | `gmadl_loss()` | N/A | ✅ Identical |

**Alignment**: Should align perfectly (within random seed variance)

---

### 3. M2 (Different Variants)

| Phase | Loss Name | Implementation | λ_dir | Sharpe | Status |
|-------|-----------|----------------|-------|--------|--------|
| Phase 1.5 | M2 | `hybrid_mul_m2` | **5.0** | 0.914 | Phase 1.5 baseline |
| Phase 2 | M2 | `m2_loss()` | **2.0** | - | ⚠️ New variant |
| Phase 2.1b | hybrid_mul | `hybrid_mul` | **1.0** | - | ⚠️ Default variant |

**Critical Issue**: These are **three different loss functions**!

**Alignment**: 
- Phase 2 M2 (λ_dir=2.0) should **NOT** be compared to Phase 1.5 M2 (λ_dir=5.0)
- Phase 2.1b hybrid_mul (λ_dir=1.0) is yet another variant
- Each should be treated as a separate loss exploration

---

## Recommended Loss Naming Convention

To avoid confusion, use explicit λ_dir in loss names:

| Loss | Old Name | New Name | λ_dir | Usage |
|------|----------|----------|-------|-------|
| Phase 1.5 M2 | `hybrid_mul_m2` | `m2_lambda5` | 5.0 | Phase 1.5 baseline |
| Phase 2 M2 | `m2_loss()` | `m2_lambda2` | 2.0 | Phase 2 experiments |
| Default hybrid_mul | `hybrid_mul` | `m2_lambda1` | 1.0 | Phase 2.1b alignment |

---

## Phase 2.1b Alignment Results

### Actual Results (2026-05-02)

| Loss | Phase 1.5 Target | Phase 2.1b Result | Deviation | Status |
|------|------------------|-------------------|-----------|--------|
| IMADL | 0.464 | 0.546 | +17.64% | ⚠️ Needs investigation |
| GMADL | 0.307 | 0.499 | +62.38% | ⚠️ Needs investigation |
| hybrid_mul (λ_dir=1.0) | 0.914 (M2 λ_dir=5.0) | 0.657 | -28.13% | ❌ Wrong comparison |

### Analysis

**IMADL and GMADL**: 
- Deviations exceed 15% threshold
- Need further investigation to determine if this is:
  - Random seed variance
  - Configuration differences (cap, window, etc.)
  - Runner bug

**M2 / hybrid_mul**:
- Comparison is invalid because λ_dir parameters differ
- Phase 2.1b hybrid_mul (λ_dir=1.0) vs Phase 1.5 M2 (λ_dir=5.0)
- This is not an alignment failure, but a parameter mismatch

---

## Phase 2.2 Results Interpretation

### m2_robust_gamma07 (Recommended)

- **Sharpe**: 0.9156
- **CV**: 0.1808
- **Base Loss**: M2 with λ_dir=2.0
- **Robustness Penalty**: gamma=0.7

**Correct Interpretation**:
- This is a **new loss variant** based on M2 (λ_dir=2.0) + robustness penalty
- Should be described as "Phase 2 exploration of M2 variant with robustness penalty"
- Should **NOT** be claimed as "improvement over Phase 1.5 M2" (different λ_dir)

### m2_robust_gamma10

- **Sharpe**: 1.0043
- **CV**: 0.5613
- **Base Loss**: M2 with λ_dir=2.0
- **Robustness Penalty**: gamma=1.0

**Correct Interpretation**:
- Highest Sharpe but high variance
- Based on M2 (λ_dir=2.0), not Phase 1.5 M2 (λ_dir=5.0)
- Should **NOT** be claimed as "10% improvement over Phase 1.5 M2"

---

## Recommendations

### For Documentation

1. **Update SHORTLIST.md**: Remove claims of "超越 Phase 1.5 M2"
2. **Update PHASE22_CRITERIA.md**: Remove M2 alignment requirements
3. **Create this mapping document**: Clarify λ_dir differences

### For Code

1. **Add loss aliases**: `m2_lambda5`, `m2_lambda2`, `m2_lambda1`
2. **Update docstrings**: Explicitly state λ_dir for each M2 variant
3. **Optional**: Rerun Phase 2.1b with `m2_lambda5` for true alignment (not required)

### For Paper/Report

**Correct Phrasing**:
- ✅ "Phase 2 explored a new M2 variant (λ_dir=2.0) with robustness penalty"
- ✅ "m2_robust_gamma07 achieved Sharpe 0.92 with low variance (CV=0.18)"
- ✅ "Phase 2 M2 (λ_dir=2.0) is a different loss variant from Phase 1.5 M2 (λ_dir=5.0)"

**Incorrect Phrasing**:
- ❌ "Phase 2 M2 improved Phase 1.5 M2 by 10%"
- ❌ "m2_robust_gamma10 surpassed Phase 1.5 best result"
- ❌ "Phase 2 achieved Sharpe 1.00, exceeding Phase 1.5's 0.91"

---

## Next Steps

### Priority 1: IMADL/GMADL Alignment Investigation

**Question**: Why do IMADL and GMADL show >15% deviation?

**Possible Causes**:
1. Random seed variance (Phase 1.5 used seeds 42, 52, 62; Phase 2.1b may use different seeds)
2. Configuration differences (weight cap, train/test window, best config path)
3. Runner implementation differences

**Action**: 
- Verify Phase 2.1b uses identical configuration to Phase 1.5 robustness test
- Check seed values, weight cap, train/test periods
- If configuration matches, investigate runner implementation

### Priority 2: M2 Variant Documentation

**Action**:
- Add explicit λ_dir to all M2 loss names
- Update all documentation to clarify M2 variants
- Remove invalid Phase 1.5 vs Phase 2 M2 comparisons

### Priority 3: Phase 2.2 Results Finalization

**Action**:
- Update SHORTLIST.md with correct interpretation
- Prepare paper/report materials with accurate phrasing
- Highlight Phase 2 as "exploration of new M2 variants" rather than "improvement over Phase 1.5"

---

## Conclusion

The Phase 2.1b alignment investigation revealed a critical parameter inconsistency: Phase 1.5 M2 (λ_dir=5.0) and Phase 2 M2 (λ_dir=2.0) are different loss functions. This explains the alignment deviation and clarifies that Phase 2 should be viewed as exploring new loss variants rather than directly improving Phase 1.5 results.

**Key Takeaway**: Always verify parameter equivalence before claiming alignment or improvement.

---

**Document Version**: v1.0  
**Status**: Complete  
**Next Review**: After IMADL/GMADL alignment investigation
