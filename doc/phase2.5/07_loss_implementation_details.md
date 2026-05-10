# Loss Implementation Details: Phase 1.5 vs Phase 2

**Investigation Date**: 2026-05-05  
**Investigator**: Research Agent  
**Purpose**: Identify subtle implementation differences that could explain IMADL (+17.6%) and GMADL (+62.4%) deviations

---

## Executive Summary

**CRITICAL FINDING**: IMADL implementations are **fundamentally different** between Phase 1.5 and Phase 2.

- **Phase 1.5 IMADL**: Uses `madl_loss()` - simple tanh-based directional loss
- **Phase 2 IMADL**: Uses `imadl_rebalanced_loss()` - complex normalized directional + magnitude loss
- **GMADL**: Identical implementation across both phases

**Conclusion**: The IMADL deviation (+17.6%) is NOT a bug or configuration issue. It's comparing two completely different loss functions that happen to share the same name.

---

## Detailed Analysis

### 1. IMADL Implementation Comparison

#### Phase 1.5 IMADL (commit 640e10f, 906ba2a)

**File**: `Model_Train/losses.py`  
**Function**: `madl_loss()` (aliased as `imadl_loss()` in Phase 2 initial)

```python
def madl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Differentiable approximation of MADL using a smoothed sign via tanh.
    """
    prod = y_true * y_pred
    alignment = torch.tanh(temperature * prod)
    loss = -alignment * torch.abs(y_true)
    return _reduce(loss, reduction)
```

**Formula**: `L = -tanh(25 * y * ŷ) * |y|`

**Characteristics**:
- Simple directional alignment via tanh
- No magnitude error term
- No normalization
- Temperature parameter: 25.0 (fixed)

---

#### Phase 2 IMADL (commit eb26783 onwards)

**File**: `Model_Train/losses.py`  
**Function**: `imadl_rebalanced_loss()`

```python
def imadl_rebalanced_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    lambda_dir: float = 1.0,
    lambda_mag: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Rebalanced Improved MADL that combines directional pressure with magnitude error.
    """
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    mag_term = (y_true - y_pred) ** 2
    loss = lambda_dir * dir_term + lambda_mag * mag_term
    return _reduce(loss, reduction)
```

**Helper Function**:
```python
def _normalized_direction_term(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    product = a * y_true * y_pred
    dir_penalty = 1.0 - torch.sigmoid(product)
    weight = torch.abs(y_true) ** b
    mean_weight = weight.mean() + eps
    normalized_weight = weight / mean_weight
    return dir_penalty * normalized_weight
```

**Formula**: 
```
dir_term = (1 - sigmoid(100 * y * ŷ)) * (|y|^2 / mean(|y|^2))
mag_term = (y - ŷ)^2
L = lambda_dir * dir_term + lambda_mag * mag_term
```

**Characteristics**:
- Complex two-component loss (directional + magnitude)
- Sigmoid-based directional penalty (not tanh)
- Normalized by batch mean weight
- Includes MSE magnitude term
- Multiple hyperparameters: a=100, b=2, lambda_dir=1.0, lambda_mag=1.0

---

### 2. Key Differences Between IMADL Implementations

| Aspect | Phase 1.5 IMADL | Phase 2 IMADL |
|--------|-----------------|---------------|
| **Base Function** | `madl_loss()` | `imadl_rebalanced_loss()` |
| **Directional Mechanism** | tanh(25 * y * ŷ) | 1 - sigmoid(100 * y * ŷ) |
| **Magnitude Term** | None | (y - ŷ)^2 |
| **Normalization** | None | Normalized by mean(|y|^b) |
| **Weight Scaling** | |y| | |y|^2 / mean(|y|^2) |
| **Parameters** | temperature=25 | a=100, b=2, λ_dir=1, λ_mag=1 |
| **Epsilon** | None | 1e-8 for numerical stability |

---

### 3. GMADL Implementation Comparison

#### Phase 1.5 GMADL (commit 640e10f)

```python
def gmadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    GMADL loss based on a scaled sigmoid of a * y_true * y_pred and |y_true|^b.
    """
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    loss = -(sigmoid - 0.5) * torch.abs(y_true) ** b
    return _reduce(loss, reduction)
```

#### Phase 2 GMADL (current)

```python
def gmadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    GMADL loss based on a scaled sigmoid of a * y_true * y_pred and |y_true|^b.
    """
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    loss = -(sigmoid - 0.5) * torch.abs(y_true) ** b
    return _reduce(loss, reduction)
```

**Status**: ✅ **IDENTICAL** - Character-for-character match

**Formula**: `L = -(sigmoid(100 * y * ŷ) - 0.5) * |y|^2`

---

### 4. Experiment Dispatcher Analysis

#### Phase 2 Initial (commit 906ba2a)

```python
def imadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    IMADL (Improved MADL) loss - alias for madl_loss for clarity.
    """
    return madl_loss(y_true, y_pred, temperature, reduction)
```

At this point, IMADL was just an alias for the original `madl_loss()`.

---

#### Phase 2 After Restructure (commit eb26783)

```python
def get_experiment_loss_fn(name: str) -> ExperimentLossFn:
    name_lower = name.lower()
    # ...
    if name_lower == "imadl":
        return lambda y_true, y_pred: imadl_rebalanced_loss(
            y_true, y_pred, reduction="mean"
        )
```

**CRITICAL CHANGE**: The dispatcher now routes "imadl" to `imadl_rebalanced_loss()` instead of `madl_loss()`.

---

### 5. Timeline of Changes

| Commit | Date | Change |
|--------|------|--------|
| 640e10f | Initial | Phase 1.5 baseline: `madl_loss()` and `gmadl_loss()` |
| 906ba2a | Phase 2 start | Added `imadl_loss()` as alias for `madl_loss()` |
| eb26783 | Apr 25, 2026 | **BREAKING**: Introduced `imadl_rebalanced_loss()` and routed "imadl" to it |
| 161ca64 | Phase 2 fixes | Added Phase 2 loss variants (M2, combinations) |
| Current | May 5, 2026 | "imadl" still routes to `imadl_rebalanced_loss()` |

---

## Hypothesis: Why GMADL Shows +62.4% Deviation

Despite identical implementations, GMADL shows a large deviation. Possible causes:

### 1. Numerical Stability Differences

**Phase 1.5 Environment**:
- PyTorch version: Unknown (need to check)
- CUDA version: Unknown
- Floating point precision: Unknown

**Phase 2 Environment**:
- PyTorch version: Need to check
- CUDA version: Need to check
- Floating point precision: Need to check

**Hypothesis**: Different PyTorch versions may have slightly different sigmoid implementations, especially for extreme values (100 * y * ŷ can be very large).

---

### 2. Reduction Method Differences

Both use `reduction="mean"`, but let me verify the `_reduce()` function:

**Phase 1.5**:
```python
def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    # ...
```

**Phase 2**:
```python
def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    # ...
```

**Status**: ✅ Identical

---

### 3. Batch Size or Data Differences

**Hypothesis**: If Phase 1.5 and Phase 2 use different:
- Batch sizes
- Data preprocessing
- Train/test splits
- Lookback windows

This could cause different loss landscapes even with identical formulas.

---

### 4. Gradient Accumulation or Optimizer Differences

**Hypothesis**: Different optimizer settings or gradient accumulation could cause divergence even with identical loss functions.

---

## Recommendations

### Priority 1: Fix IMADL Naming Confusion

**Problem**: "IMADL" refers to two completely different loss functions.

**Solution**:
1. Rename Phase 2 `imadl_rebalanced_loss()` to something distinct (e.g., `imadl_v2_loss()` or `hybrid_imadl_loss()`)
2. Restore original `imadl_loss()` as alias for `madl_loss()` for Phase 1.5 compatibility
3. Update all documentation to clarify the difference

**Code Change**:
```python
# Phase 1.5 compatible IMADL
def imadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 25.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Original IMADL from Phase 1.5 - simple tanh-based directional loss."""
    return madl_loss(y_true, y_pred, temperature, reduction)

# Phase 2 new variant
def imadl_v2_loss(  # or imadl_rebalanced_loss
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    lambda_dir: float = 1.0,
    lambda_mag: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Phase 2 rebalanced IMADL with normalized directional + magnitude terms."""
    dir_term = _normalized_direction_term(y_true, y_pred, a=a, b=b)
    mag_term = (y_true - y_pred) ** 2
    loss = lambda_dir * dir_term + lambda_mag * mag_term
    return _reduce(loss, reduction)
```

---

### Priority 2: Investigate GMADL Deviation

**Action Items**:
1. Check PyTorch versions in Phase 1.5 vs Phase 2 environments
2. Verify CUDA versions
3. Compare batch sizes and data preprocessing
4. Check optimizer settings (learning rate, weight decay, etc.)
5. Verify train/test split consistency
6. Check for any data normalization differences

**Test**:
Run a minimal test with identical data, identical hyperparameters, and identical random seed to isolate the cause.

---

### Priority 3: Update Documentation

**Files to Update**:
1. `doc/phase2-fix/LOSS_MAPPING.md` - Add IMADL implementation difference
2. `doc/phase2.5/02_lambda_dir_check.md` - Note IMADL is not comparable
3. `Model_Train/losses.py` - Add clear docstrings explaining the difference

**Correct Phrasing**:
- ✅ "Phase 2 introduced a new IMADL variant (imadl_rebalanced_loss) with normalized directional + magnitude terms"
- ✅ "Phase 1.5 IMADL (madl_loss) and Phase 2 IMADL (imadl_rebalanced_loss) are different loss functions"
- ❌ "Phase 2 IMADL improved Phase 1.5 IMADL by 17.6%"
- ❌ "IMADL alignment test shows +17.6% deviation"

---

## Conclusion

The investigation revealed a **critical naming collision**: "IMADL" refers to two fundamentally different loss functions in Phase 1.5 and Phase 2.

**Phase 1.5 IMADL**:
- Simple tanh-based directional loss
- Formula: `L = -tanh(25 * y * ŷ) * |y|`
- No magnitude term, no normalization

**Phase 2 IMADL**:
- Complex hybrid loss with normalized directional + magnitude terms
- Formula: `L = λ_dir * (1 - sigmoid(100 * y * ŷ)) * (|y|^2 / mean(|y|^2)) + λ_mag * (y - ŷ)^2`
- Includes MSE term, batch normalization, multiple hyperparameters

**GMADL** implementations are identical, but the +62.4% deviation suggests environmental differences (PyTorch version, CUDA, batch size, data preprocessing, or optimizer settings).

**Key Takeaway**: The IMADL "deviation" is not a deviation at all - it's a comparison between two different loss functions. Phase 2 should be viewed as exploring new loss variants, not aligning with Phase 1.5.

---

**Document Version**: v1.0  
**Status**: Complete  
**Next Steps**: 
1. Rename Phase 2 IMADL to avoid confusion
2. Investigate GMADL environmental differences
3. Update all documentation to reflect findings
