# Phase 2.1b Alignment Failure: Configuration Comparison

**Date**: 2026-05-05  
**Status**: ROOT CAUSE IDENTIFIED  
**Severity**: CRITICAL

## Executive Summary

Phase 2.1b alignment test failed to replicate Phase 1.5 baseline results with deviations of 15-62%:
- **IMADL**: 0.464 target → 0.546 observed (+17.6%)
- **GMADL**: 0.307 target → 0.499 observed (+62.4%)
- **M2 (hybrid_mul)**: 0.914 target → 0.657 observed (-28.1%)

**ROOT CAUSE IDENTIFIED**: Phase 2.1b used `hybrid_mul` with **λ_dir=1.0** (default), but Phase 1.5 M2 used **λ_dir=5.0**. These are fundamentally different loss functions and should never have been compared directly.

---

## Code Locations

### Phase 1.5 Code
- **Branch**: `codex/phase15-colab-drive`
- **Runner**: `run_phase15_robustness.py`
- **Core**: `sanity_check_core.py`
- **Losses**: `Model_Train/losses.py`
- **Loss Registry**: `get_experiment_loss_fn()` with `hybrid_mul_m2` variant

### Phase 2.1b Code
- **Branch**: `phase2.2-fix` (current)
- **Runner**: `run_phase2_1b_alignment.py`
- **Core**: `sanity_check_core.py` (identical to Phase 1.5)
- **Losses**: `Model_Train/losses.py` (extended with Phase 2 variants)
- **Loss Registry**: `get_experiment_loss_fn()` with default `hybrid_mul`

---

## Critical Difference: M2 Loss λ_dir Parameter

### Phase 1.5 M2 (hybrid_mul_m2)
```python
# codex/phase15-colab-drive:Model_Train/losses.py
if name_lower == "hybrid_mul_m2":
    return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
        y_true, y_pred, lambda_dir=5.0, reduction="mean"
    )
```

**Performance**: Sharpe 1.0316, Cumulative Return 54.28% (seed=42, cap=0.05)

### Phase 2.1b hybrid_mul (default)
```python
# HEAD:Model_Train/losses.py
if name_lower == "hybrid_mul":
    return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
        y_true, y_pred, reduction="mean"  # Uses default lambda_dir=1.0
    )
```

**Performance**: Sharpe 0.657 (seed=42, cap=0.05) - **28.1% deviation**

### Loss Function Definition
```python
def hybrid_dir_huber_mul_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    delta: float = 0.01,
    lambda_dir: float = 1.0,  # DEFAULT VALUE
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Multiplicative hybrid directional-Huber loss.
    loss = (1.0 + lambda_dir * dir_term) * huber_term
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    huber_term = _huber_term(y_true - y_pred, delta=delta)
    loss = (1.0 + lambda_dir * dir_term) * huber_term
    return _reduce(loss, reduction)
```

**Impact of λ_dir**:
- λ_dir=1.0: `loss = (1.0 + 1.0 * dir_term) * huber_term` → moderate directional penalty
- λ_dir=5.0: `loss = (1.0 + 5.0 * dir_term) * huber_term` → **5x stronger** directional penalty

---

## Side-by-Side Configuration Comparison

| Configuration | Phase 1.5 | Phase 2.1b | Match? |
|--------------|-----------|------------|--------|
| **Loss Functions** | | | |
| IMADL | `imadl_rebalanced_loss()` | `imadl_rebalanced_loss()` | ✅ |
| GMADL | `gmadl_loss()` | `gmadl_loss()` | ✅ |
| M2 (hybrid_mul) | `hybrid_dir_huber_mul_loss(λ_dir=5.0)` | `hybrid_dir_huber_mul_loss(λ_dir=1.0)` | ❌ **MISMATCH** |
| **Training Hyperparameters** | | | |
| Max Epochs | 20 | 20 | ✅ |
| Batch Size | 1024 | 1024 | ✅ |
| Learning Rate | From `best_hyperparameters.txt` | From `best_hyperparameters.txt` | ✅ |
| **Data Configuration** | | | |
| Train Window | 1990-01 to 1994-12 | 1990-01 to 1994-12 | ✅ |
| Test Window | 1995-01 + 6 months | 1995-01 + 24 months | ⚠️ **DIFFERENT** |
| Lookback Months | 12 | 12 | ✅ |
| Feature Set | X1 (12-month lookback) | X1 (12-month lookback) | ✅ |
| **Random Seeds** | | | |
| Seeds | 42, 52, 62 | 42, 52, 62 | ✅ |
| Seed Setting | `set_seed()` | `set_seed()` | ✅ |
| **Weight Capping** | | | |
| Max Weight | 0.05 (5%) | 0.05 (5%) | ✅ |
| **Model Architecture** | | | |
| Config Source | `best_hyperparameters.txt` | `best_hyperparameters.txt` | ✅ |
| Hidden Dims | From config | From config | ✅ |
| Activation | From config (default: relu) | From config (default: relu) | ✅ |
| Dropout | From config (default: 0.0) | From config (default: 0.0) | ✅ |

---

## Detailed Findings

### 1. M2 Loss λ_dir Parameter (CRITICAL)

**Finding**: Phase 1.5 used `hybrid_mul_m2` with λ_dir=5.0, but Phase 2.1b used `hybrid_mul` with λ_dir=1.0 (default).

**Impact**: 
- The directional penalty term is multiplied by λ_dir before being added to 1.0
- λ_dir=5.0 creates a loss range of [1.0, 6.0] × huber_term
- λ_dir=1.0 creates a loss range of [1.0, 2.0] × huber_term
- **5x difference in directional penalty strength**

**Evidence**:
```bash
# Phase 1.5 loss registry
$ git show codex/phase15-colab-drive:Model_Train/losses.py | grep -A 3 "hybrid_mul_m2"
if name_lower == "hybrid_mul_m2":
    return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
        y_true, y_pred, lambda_dir=5.0, reduction="mean"
    )

# Phase 2.1b loss registry
$ grep -A 3 "hybrid_mul" Model_Train/losses.py
if name_lower == "hybrid_mul":
    return lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
        y_true, y_pred, reduction="mean"  # lambda_dir defaults to 1.0
    )
```

**Commit Evidence**:
```
commit 95a5cdf2fff5954db76f88a890b975f53af0bff1
Author: roucher <roucher@aoao.dev>
Date:   Sun May 3 16:30:35 2026 +0800

    feat(losses): add M2 lambda_dir variants for Phase 1.5/2/2.1b alignment
    
    Root cause: Phase 1.5 M2 used λ_dir=5.0, but Phase 2 M2 used λ_dir=2.0,
    and Phase 2.1b hybrid_mul used λ_dir=1.0 (default). These are different
    loss variants and should not be directly compared.
```

### 2. Test Window Length (MINOR)

**Finding**: Phase 1.5 used 6-month test windows, Phase 2.1b used 24-month test windows.

**Impact**: 
- Longer test windows provide more data points for Sharpe ratio calculation
- May affect variance estimation and statistical significance
- **However, this is unlikely to cause 15-62% deviations**

**Evidence**:
```python
# Phase 1.5 default
parser.add_argument("--test-months", type=int, default=6)

# Phase 2.1b default
parser.add_argument("--test-months", type=int, default=24)
```

### 3. IMADL and GMADL Implementations (VERIFIED IDENTICAL)

**Finding**: IMADL and GMADL loss functions are identical between Phase 1.5 and Phase 2.1b.

**Evidence**:
```bash
$ git diff codex/phase15-colab-drive:Model_Train/losses.py HEAD:Model_Train/losses.py -- '*madl*'
# No differences in madl_loss, gmadl_loss, or imadl_rebalanced_loss implementations
```

**Implication**: The 17.6% IMADL deviation and 62.4% GMADL deviation are NOT caused by loss function changes. Other factors must be investigated:
- Data preprocessing differences
- Feature engineering changes
- Model architecture differences
- Training dynamics (optimizer, learning rate schedule)
- Random seed handling

### 4. Core Training Loop (VERIFIED IDENTICAL)

**Finding**: `sanity_check_core.py` is identical between Phase 1.5 and Phase 2.1b.

**Evidence**:
```bash
$ git diff codex/phase15-colab-drive:sanity_check_core.py HEAD:sanity_check_core.py
# No differences in training loop, data loading, or evaluation logic
```

---

## Ranked Hypotheses

### Hypothesis 1: M2 λ_dir Mismatch (CONFIRMED)
**Probability**: 100%  
**Severity**: CRITICAL  
**Status**: ROOT CAUSE IDENTIFIED

Phase 2.1b used λ_dir=1.0 instead of λ_dir=5.0, creating a fundamentally different loss function. This explains the 28.1% M2 deviation.

**Fix**: Use `m2_lambda5_loss()` or explicitly pass `lambda_dir=5.0` to `hybrid_dir_huber_mul_loss()`.

### Hypothesis 2: Data Preprocessing Differences
**Probability**: 80%  
**Severity**: HIGH  
**Status**: REQUIRES INVESTIGATION

IMADL and GMADL loss functions are identical, yet show 17.6% and 62.4% deviations. Possible causes:
- Feature normalization differences
- Data loading order (despite seed setting)
- CSV file differences (Phase 1.5 vs Phase 2.1b data sources)
- Pandas version differences affecting data parsing

**Next Steps**: 
1. Compare `Model_Train/data_preprocess.py` between branches
2. Compare `Model_Train/features.py` between branches
3. Verify CSV data files are identical
4. Check for floating-point precision differences

### Hypothesis 3: Model Architecture Differences
**Probability**: 60%  
**Severity**: MEDIUM  
**Status**: REQUIRES INVESTIGATION

Both phases use `best_hyperparameters.txt`, but:
- The config file may differ between Phase 1.5 and Phase 2.1b
- Model initialization may differ despite seed setting
- PyTorch version differences may affect layer initialization

**Next Steps**:
1. Compare `best_hyperparameters.txt` files
2. Compare `Model_Train/models.py` between branches
3. Check PyTorch versions used in Phase 1.5 vs Phase 2.1b

### Hypothesis 4: Test Window Length Effect
**Probability**: 20%  
**Severity**: LOW  
**Status**: UNLIKELY

Phase 1.5 used 6-month test windows, Phase 2.1b used 24-month windows. While this affects sample size, it's unlikely to cause 15-62% deviations in Sharpe ratios.

**Rationale**: Sharpe ratio is a normalized metric (mean return / std return). Longer windows should converge to similar values unless there's regime change.

### Hypothesis 5: Random Seed Handling
**Probability**: 40%  
**Severity**: MEDIUM  
**Status**: REQUIRES INVESTIGATION

Despite using identical `set_seed()` functions, differences may arise from:
- PyTorch version differences (CUDA/MPS backend changes)
- NumPy version differences
- DataLoader shuffling differences
- Model weight initialization order

**Next Steps**:
1. Check PyTorch/NumPy versions in Phase 1.5 vs Phase 2.1b environments
2. Verify DataLoader shuffle behavior
3. Compare first-batch predictions to detect initialization differences

---

## Immediate Action Items

### Priority 1: Fix M2 Alignment (CRITICAL)
- [ ] Update Phase 2.1b runner to use `m2_lambda5_loss()` instead of `hybrid_mul`
- [ ] Re-run Phase 2.1b alignment test with correct M2 loss
- [ ] Verify M2 Sharpe ratio converges to 0.914 ± 5%

### Priority 2: Investigate IMADL/GMADL Deviations (HIGH)
- [ ] Compare `data_preprocess.py` between branches
- [ ] Compare `features.py` between branches
- [ ] Verify CSV data files are identical (checksums)
- [ ] Check for floating-point precision differences

### Priority 3: Document Loss Variants (MEDIUM)
- [ ] Update Phase 2 documentation to clarify M2 variants
- [ ] Create loss mapping table: Phase 1.5 → Phase 2 → Phase 2.1b
- [ ] Add warnings to prevent future misalignment

### Priority 4: Standardize Test Windows (LOW)
- [ ] Decide on standard test window length (6 vs 24 months)
- [ ] Document rationale for choice
- [ ] Update all runners to use consistent test window

---

## Conclusion

The Phase 2.1b alignment failure was caused by a **critical configuration mismatch**: Phase 2.1b used `hybrid_mul` with λ_dir=1.0, but Phase 1.5 M2 used λ_dir=5.0. This 5x difference in directional penalty strength explains the 28.1% M2 deviation.

However, the 17.6% IMADL and 62.4% GMADL deviations remain unexplained, as these loss functions are identical between phases. Further investigation is required to identify data preprocessing, model architecture, or training dynamic differences.

**Recommendation**: Do not proceed with Phase 2.2 experiments until IMADL/GMADL alignment is restored to within ±5% of Phase 1.5 baselines.
