# Phase 2 Experiment Plan - Loss Function Combinations

> **Objective:** Design and evaluate novel loss function combinations that balance stability (IMADL) with high-return potential (M2), addressing robustness issues identified in Phase 1.5.

**Date:** 2026-04-26  
**Status:** Planning  
**Previous Phase:** Phase 1.5 Robustness Testing (completed)

---

## Executive Summary

Phase 1.5 revealed that:
- **IMADL** is most stable (CV=0.892, Sharpe=0.464) but has moderate returns
- **M2** has highest average Sharpe (0.914) but extreme variance (CV=1.396, 33% failure rate)
- **M1** has unacceptable failure rate (66.7%) and should be abandoned

**Phase 2 Goal:** Combine the strengths of IMADL (stability) and M2 (high returns) through four novel loss function variants.

---

## Four Loss Function Variants

### Variant 1: IMADL + M2 Linear Combination
**File:** `Loss-Variant-1-IMADL-M2-Linear.md`

**Formula:**
```
loss = α * IMADL + (1-α) * hybrid_mul_m2
```

**Rationale:** Linear interpolation between stable (IMADL) and aggressive (M2) losses.

**Parameters:** α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

**Expected Outcome:** Find optimal balance point between stability and returns.

---

### Variant 2: IMADL + GMADL Weighted Combination
**File:** `Loss-Variant-2-IMADL-GMADL-Weighted.md`

**Formula:**
```
loss = β * IMADL + (1-β) * GMADL
```

**Rationale:** Both are MAD-based with similar structure, may complement each other.

**Parameters:** β ∈ {0.3, 0.5, 0.7}

**Expected Outcome:** Improve IMADL's returns while maintaining stability.

---

### Variant 3: Robustness-Enhanced M2
**File:** `Loss-Variant-3-M2-Robustness-Enhanced.md`

**Formula:**
```
loss = hybrid_mul_m2 + γ * robustness_penalty
robustness_penalty = Var(monthly_returns)
```

**Rationale:** Add explicit robustness constraint to M2 to reduce variance.

**Parameters:** γ ∈ {0.01, 0.1, 1.0}

**Expected Outcome:** Reduce M2's variance, improve cross-seed stability.

---

### Variant 4: Adaptive Hybrid
**File:** `Loss-Variant-4-Adaptive-Hybrid.md`

**Formula:**
```
loss = IMADL * exp(-λ * |y_true|) + M2 * (1 - exp(-λ * |y_true|))
```

**Rationale:** Use IMADL for small returns (stable), M2 for large returns (aggressive).

**Parameters:** λ ∈ {1.0, 5.0, 10.0}

**Expected Outcome:** Dynamically adjust loss weighting based on sample importance.

---

## Experiment Matrix

### Phase 2.1: Initial Screening (39 runs)

**Configuration:**
- **Variants:** 4 types × varying parameters = 13 total loss functions
  - Variant 1: 7 losses (α = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
  - Variant 2: 3 losses (β = 0.3, 0.5, 0.7)
  - Variant 3: 3 losses (γ = 0.01, 0.1, 1.0)
  - Variant 4: 3 losses (λ = 1.0, 5.0, 10.0)
- **Seeds:** 42, 52, 62 (3 seeds)
- **Weight Cap:** 0.05 (consistent with Phase 1.5)
- **Training Period:** 1990-01 to 1994-12 (5 years)
- **Testing Period:** 1995-01 to 1996-12 (24 months)
- **Total Runs:** 13 losses × 3 seeds = 39 runs

**Evaluation Metrics:**
1. Average Sharpe ratio (primary)
2. Sharpe standard deviation (stability)
3. Failure rate (Sharpe < 0)
4. Maximum drawdown
5. Cumulative return
6. Coefficient of variation (CV = std/mean)

**Success Criteria:**
- Average Sharpe > 0.5
- Sharpe std < 0.5 (CV < 1.0)
- Failure rate < 20%
- Better than IMADL baseline (Sharpe 0.464, CV 0.892)

---

### Phase 2.2: Extended Validation (36 runs)

**Selection:** Top 2-3 performers from Phase 2.1

**Configuration:**
- **Variants:** Best 3 losses from Phase 2.1
- **Seeds:** 42, 52, 62, 72, 82, 92 (6 seeds)
- **Weight Caps:** 0.05, None (2 configurations)
- **Training Period:** 1990-01 to 1994-12 (5 years)
- **Testing Period:** 1995-01 to 1998-12 (48 months, extended)
- **Total Runs:** 3 losses × 6 seeds × 2 caps = 36 runs

**Additional Analysis:**
1. Monthly return distribution
2. Performance by market regime (bull/bear/sideways)
3. Drawdown analysis
4. Comparison with Phase 1.5 baselines

---

## Timeline

**Phase 2.1: Initial Screening**
- Week 1: Code implementation and testing
- Week 2: Run experiments on Colab
- Week 3: Analyze results, select top performers

**Phase 2.2: Extended Validation**
- Week 4: Run extended experiments
- Week 5: Deep analysis and comparison
- Week 6: Write results section for thesis

**Total Duration:** 6 weeks

---

## Code Implementation Guidelines

### Step 1: Define New Loss Functions in `losses.py`

Add to `EXPERIMENT_LOSS_NAMES`:
```python
EXPERIMENT_LOSS_NAMES = (
    # ... existing losses ...
    "imadl_m2_linear_02",  # α=0.2
    "imadl_m2_linear_03",  # α=0.3
    "imadl_m2_linear_04",  # α=0.4
    "imadl_m2_linear_05",  # α=0.5
    "imadl_m2_linear_06",  # α=0.6
    "imadl_m2_linear_07",  # α=0.7
    "imadl_m2_linear_08",  # α=0.8
    "imadl_gmadl_weighted_03",  # β=0.3
    "imadl_gmadl_weighted_05",  # β=0.5
    "imadl_gmadl_weighted_07",  # β=0.7
    "m2_robust_001",  # γ=0.01
    "m2_robust_01",   # γ=0.1
    "m2_robust_10",   # γ=1.0
    "adaptive_hybrid_10",  # λ=1.0
    "adaptive_hybrid_50",  # λ=5.0
    "adaptive_hybrid_100", # λ=10.0
)
```

Add implementations in `get_experiment_loss_fn()`:
```python
# Variant 1: IMADL + M2 Linear
if name_lower == "imadl_m2_linear_02":
    return lambda y_true, y_pred: (
        0.2 * imadl_loss(y_true, y_pred) + 
        0.8 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    )

# Variant 2: IMADL + GMADL Weighted
if name_lower == "imadl_gmadl_weighted_03":
    return lambda y_true, y_pred: (
        0.3 * imadl_loss(y_true, y_pred) + 
        0.7 * gmadl_loss(y_true, y_pred)
    )

# Variant 3: Robustness-Enhanced M2
if name_lower == "m2_robust_001":
    return lambda y_true, y_pred: (
        hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0) +
        0.01 * robustness_penalty(y_true, y_pred)
    )

# Variant 4: Adaptive Hybrid
if name_lower == "adaptive_hybrid_10":
    return lambda y_true, y_pred: adaptive_hybrid_loss(
        y_true, y_pred, lambda_param=1.0
    )
```

**Note:** Detailed implementation for each variant is in the individual variant files.

### Step 2: Create Runner Scripts

Create `run_sanity_check_<variant>.py` for each new loss function (13 total).

### Step 3: Update `run_all_experiments.py`

Add new loss names to `RUNNER_BY_LOSS` dictionary.

### Step 4: Create Phase 2 Robustness Script

Create `run_phase2_robustness.py` similar to `run_phase15_robustness.py` but with:
- New loss function list
- Phase 2.1 and Phase 2.2 configurations
- Extended test period option (48 months)

---

## Baseline Comparisons

All Phase 2 results will be compared against Phase 1.5 baselines:

| Loss | Avg Sharpe | Std Dev | CV | Failure Rate |
|------|-----------|---------|-----|--------------|
| IMADL | 0.464 | 0.414 | 0.892 | 0% |
| M2 | 0.914 | 1.042 | 1.396 | 33% |
| GMADL | 0.307 | 0.358 | 1.168 | 0% |

**Target Performance:**
- Sharpe > 0.6 (better than IMADL)
- CV < 1.0 (better than M2)
- Failure rate < 10%

---

## Expected Contributions

### 1. Methodological Innovation
- Novel loss function combination strategies
- Robustness-aware loss design
- Adaptive weighting based on sample importance

### 2. Empirical Findings
- Optimal balance between stability and returns
- Cross-seed generalization of hybrid losses
- Impact of robustness penalties on trading performance

### 3. Theoretical Insights
- Why certain loss combinations work better
- Trade-offs between directional accuracy and magnitude prediction
- Role of lambda parameters in cross-seed robustness

---

## Risk Mitigation

**Risk 1: All variants fail to improve over IMADL**
- Mitigation: Extend to Phase 2.3 with different parameter ranges
- Fallback: Use IMADL as final recommendation, focus thesis on "why IMADL is robust"

**Risk 2: Variants show high variance like M2**
- Mitigation: Increase robustness penalty weights (γ)
- Fallback: Ensemble multiple seeds instead of single-seed optimization

**Risk 3: Computational resource constraints**
- Mitigation: Run Phase 2.1 first, only proceed to 2.2 if promising results
- Fallback: Reduce number of seeds in Phase 2.2 (4 instead of 6)

---

## Next Steps

1. **Review and approve this plan** - Confirm experiment design and parameters
2. **Create detailed variant documents** - One file per variant with full specifications
3. **Implement loss functions** - Add to `losses.py`
4. **Set up Colab environment** - Prepare for Phase 2.1 experiments
5. **Run Phase 2.1** - Initial screening (39 runs)
6. **Analyze and select** - Choose top 2-3 performers for Phase 2.2
7. **Run Phase 2.2** - Extended validation (36 runs)
8. **Write thesis chapter** - Document findings and contributions

---

**Document Version:** v1.0  
**Created:** 2026-04-26  
**Author:** Yirong Yu  
**Status:** Planning Phase
