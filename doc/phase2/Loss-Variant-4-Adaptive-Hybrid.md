# Loss Variant 4: Adaptive Hybrid

> **Objective:** Dynamically adjust loss weighting based on sample importance - use stable IMADL for small returns, aggressive M2 for large returns.

**Variant Type:** Adaptive Weighting  
**Base Losses:** IMADL (small returns) + M2 (large returns)  
**Weighting Function:** Exponential decay based on |y_true|  
**Parameter:** λ (decay rate)

---

## Mathematical Definition

### Formula

```
loss = IMADL * w_imadl + M2 * w_m2

where:
  w_imadl = exp(-λ * |y_true|)
  w_m2 = 1 - exp(-λ * |y_true|)
  λ ∈ {1.0, 5.0, 10.0}
```

**Full Expression:**
```
loss = imadl_loss(y_true, y_pred) * exp(-λ * |y_true|) + 
       hybrid_mul_m2(y_true, y_pred) * (1 - exp(-λ * |y_true|))
```

### Weighting Function Behavior

**For small |y_true| (e.g., 0.01):**
- w_imadl ≈ 1.0 (IMADL dominates)
- w_m2 ≈ 0.0 (M2 has minimal influence)
- **Rationale:** Small returns are common, use stable IMADL

**For large |y_true| (e.g., 0.10):**
- w_imadl ≈ 0.0 (IMADL has minimal influence)
- w_m2 ≈ 1.0 (M2 dominates)
- **Rationale:** Large returns are rare but important, use aggressive M2

### Weight Curves by λ

| |y_true| | λ=1.0 | λ=5.0 | λ=10.0 |
|---------|-------|-------|--------|
| 0.01 | w_imadl=0.99, w_m2=0.01 | w_imadl=0.95, w_m2=0.05 | w_imadl=0.90, w_m2=0.10 |
| 0.05 | w_imadl=0.95, w_m2=0.05 | w_imadl=0.78, w_m2=0.22 | w_imadl=0.61, w_m2=0.39 |
| 0.10 | w_imadl=0.90, w_m2=0.10 | w_imadl=0.61, w_m2=0.39 | w_imadl=0.37, w_m2=0.63 |
| 0.20 | w_imadl=0.82, w_m2=0.18 | w_imadl=0.37, w_m2=0.63 | w_imadl=0.14, w_m2=0.86 |

**Interpretation:**
- **λ=1.0:** Slow transition, IMADL dominates even for large returns
- **λ=5.0:** Moderate transition, balanced weighting
- **λ=10.0:** Fast transition, M2 dominates for moderate-to-large returns

---

## Rationale

### Why Adaptive Weighting?

1. **Sample Heterogeneity:**
   - Small returns (|y| < 0.05): ~80% of samples, low signal-to-noise
   - Large returns (|y| > 0.10): ~10% of samples, high trading value
   - Different samples need different loss functions

2. **Loss Function Strengths:**
   - **IMADL:** Stable, good for noisy small returns
   - **M2:** Aggressive, good for capturing large moves
   - **Fixed combination:** Treats all samples equally (suboptimal)

3. **Adaptive Advantage:**
   - Automatically adjust based on sample importance
   - No need to manually tune α (like Variant 1)
   - Theoretically more principled than linear combination

### Expected Behavior by λ

| λ | Transition Speed | Expected Behavior |
|---|-----------------|-------------------|
| 1.0 | Slow | IMADL-dominant, conservative |
| 5.0 | Moderate | Balanced, adaptive |
| 10.0 | Fast | M2-dominant for moderate returns |

**Hypothesis:** λ=5.0 will provide optimal balance, smoothly transitioning from IMADL to M2 as return magnitude increases.

---

## Parameter Grid

### 3 Configurations

1. **adaptive_hybrid_10** (λ=1.0): Slow transition, IMADL-dominant
2. **adaptive_hybrid_50** (λ=5.0): Moderate transition, balanced
3. **adaptive_hybrid_100** (λ=10.0): Fast transition, M2-dominant

---

## Code Implementation

### Add to `losses.py`

```python
# In EXPERIMENT_LOSS_NAMES tuple
EXPERIMENT_LOSS_NAMES = (
    # ... existing losses ...
    "adaptive_hybrid_10",
    "adaptive_hybrid_50",
    "adaptive_hybrid_100",
)

# Helper function for adaptive weighting
def adaptive_hybrid_loss(y_true: torch.Tensor, 
                        y_pred: torch.Tensor, 
                        lambda_param: float) -> torch.Tensor:
    """
    Adaptive hybrid loss that weights IMADL and M2 based on |y_true|.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
        lambda_param: Decay rate for exponential weighting
    
    Returns:
        Weighted combination of IMADL and M2
    """
    # Compute base losses
    imadl = imadl_loss(y_true, y_pred)
    m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    
    # Compute adaptive weights
    abs_y = torch.abs(y_true)
    w_imadl = torch.exp(-lambda_param * abs_y)
    w_m2 = 1.0 - w_imadl
    
    # Weighted combination
    loss = imadl * w_imadl.mean() + m2 * w_m2.mean()
    return loss

# In get_experiment_loss_fn() function
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... existing loss implementations ...
    
    # Variant 4: Adaptive Hybrid
    if name_lower == "adaptive_hybrid_10":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=1.0
        )
    
    if name_lower == "adaptive_hybrid_50":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=5.0
        )
    
    if name_lower == "adaptive_hybrid_100":
        return lambda y_true, y_pred: adaptive_hybrid_loss(
            y_true, y_pred, lambda_param=10.0
        )
```

### Create Runner Scripts

Create 3 files: `run_sanity_check_adaptive_hybrid_10.py`, `run_sanity_check_adaptive_hybrid_50.py`, `run_sanity_check_adaptive_hybrid_100.py`

**Template:**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (Adaptive Hybrid λ=1.0)")
    args = parser.parse_args()
    run_sanity_check("adaptive_hybrid_10", args)

if __name__ == "__main__":
    main()
```

### Update `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... existing mappings ...
    "adaptive_hybrid_10": "run_sanity_check_adaptive_hybrid_10.py",
    "adaptive_hybrid_50": "run_sanity_check_adaptive_hybrid_50.py",
    "adaptive_hybrid_100": "run_sanity_check_adaptive_hybrid_100.py",
}
```

---

## Experiment Configuration

### Phase 2.1: Initial Screening

**Runs:** 3 losses × 3 seeds = 9 runs

| Loss Name | λ | Seeds | Weight Cap | Test Period |
|-----------|---|-------|------------|-------------|
| adaptive_hybrid_10 | 1.0 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| adaptive_hybrid_50 | 5.0 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| adaptive_hybrid_100 | 10.0 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |

---

## Expected Results

### Performance Predictions

**Low λ (1.0): Slow Transition**
- Expected Sharpe: 0.5-0.6
- Expected CV: 0.8-1.0
- Failure rate: 0-5%
- Behavior: Conservative, close to IMADL

**Medium λ (5.0): Moderate Transition**
- Expected Sharpe: 0.6-0.8
- Expected CV: 0.9-1.1
- Failure rate: 5-15%
- Behavior: Balanced, adaptive to sample importance

**High λ (10.0): Fast Transition**
- Expected Sharpe: 0.7-0.9
- Expected CV: 1.1-1.3
- Failure rate: 10-20%
- Behavior: Aggressive, closer to M2 for moderate returns

### Success Criteria

**Minimum Requirements:**
- Average Sharpe > 0.6
- CV < 1.1
- Failure rate < 15%

**Ideal Performance:**
- Average Sharpe > 0.7
- CV < 1.0
- Failure rate < 10%
- Better than both IMADL and M2 baselines

---

## Analysis Plan

### Key Questions

1. **Does adaptive weighting improve over fixed combination?**
   - Compare vs Variant 1 (IMADL+M2 linear)
   - Check if adaptive approach is more robust

2. **What is the optimal λ?**
   - Which decay rate provides best risk-adjusted returns?

3. **How do weights distribute in practice?**
   - Analyze actual weight distribution across samples
   - Verify that small returns use IMADL, large returns use M2

4. **Is the transition smooth?**
   - Check for discontinuities or instabilities
   - Verify gradient flow during training

### Visualization

1. **λ vs Sharpe Plot:** Show performance trend
2. **λ vs CV Plot:** Show stability trend
3. **Weight Distribution:** Histogram of w_imadl and w_m2 across samples
4. **Weight vs |y_true| Scatter:** Verify exponential decay pattern
5. **Sample-Level Analysis:** Show which samples use which loss

---

## Implementation Notes

### Alternative Weighting Functions

The current implementation uses exponential decay. Other options:

**Variant A: Sigmoid Weighting**
```python
def sigmoid_weighting(y_true, lambda_param):
    w_m2 = torch.sigmoid(lambda_param * (torch.abs(y_true) - 0.05))
    w_imadl = 1.0 - w_m2
    return w_imadl, w_m2
```

**Variant B: Threshold-Based**
```python
def threshold_weighting(y_true, threshold=0.05):
    w_imadl = (torch.abs(y_true) < threshold).float()
    w_m2 = 1.0 - w_imadl
    return w_imadl, w_m2
```

**Variant C: Polynomial Decay**
```python
def polynomial_weighting(y_true, power=2):
    abs_y = torch.abs(y_true)
    w_m2 = abs_y ** power
    w_imadl = 1.0 - w_m2
    return w_imadl, w_m2
```

**Recommendation:** Start with exponential decay (most principled). If results are promising, explore alternatives in Phase 2.2.

### Gradient Considerations

**Potential Issue:** Adaptive weights may cause gradient instability if weights change too rapidly.

**Solution:** Use smooth exponential function (already implemented) and monitor:
1. Loss curve smoothness during training
2. Gradient norms
3. Weight distribution stability

If instability occurs, consider:
- Reducing λ range
- Adding weight clipping
- Using moving average of weights

---

## Thesis Contribution

### Methodological Innovation

- **Adaptive loss weighting:** First sample-importance-based loss combination for trading
- **Principled approach:** Theoretically motivated by sample heterogeneity
- **Automatic tuning:** No need for manual α selection (unlike Variant 1)

### Expected Findings

1. **Optimal λ:** Likely λ=5.0 provides best balance
2. **Adaptive advantage:** 5-10% improvement over fixed combination
3. **Weight distribution:** Verify that small returns use IMADL, large returns use M2
4. **Robustness:** Better cross-seed stability than pure M2

### Theoretical Insights

- Why sample-adaptive weighting is superior to fixed weighting
- Role of return magnitude in loss function selection
- Trade-offs between simplicity (fixed α) and adaptivity (exponential λ)

---

## Comparison with Variant 1

| Aspect | Variant 1 (Linear) | Variant 4 (Adaptive) |
|--------|-------------------|---------------------|
| Weighting | Fixed α for all samples | Adaptive based on \|y_true\| |
| Parameters | 7 values (α=0.2-0.8) | 3 values (λ=1.0, 5.0, 10.0) |
| Complexity | Simple, interpretable | More complex, principled |
| Flexibility | One weight for all | Different weight per sample |
| Expected Performance | Good baseline | Potentially better |

**Hypothesis:** Variant 4 will outperform Variant 1 by 5-10% Sharpe due to adaptive weighting.

---

**Document Version:** v1.0  
**Created:** 2026-04-26  
**Author:** Yirong Yu  
**Status:** Ready for Implementation
