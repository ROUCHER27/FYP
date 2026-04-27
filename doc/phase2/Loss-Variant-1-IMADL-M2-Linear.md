# Loss Variant 1: IMADL + M2 Linear Combination

> **Objective:** Find optimal linear interpolation between stable IMADL and high-return M2 to balance risk and reward.

**Variant Type:** Linear Combination  
**Base Losses:** IMADL (stable) + M2 (aggressive)  
**Parameter:** α (IMADL weight)

---

## Mathematical Definition

### Formula

```
loss = α * IMADL + (1-α) * hybrid_mul_m2

where:
  IMADL = imadl_loss(y_true, y_pred)
  hybrid_mul_m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
  α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
```

### Component Breakdown

**IMADL (Inverse Mean Absolute Directional Loss):**
```python
def imadl_loss(y_true, y_pred, a=100.0, b=2.0, eps=1e-8):
    product = a * y_true * y_pred
    dir_penalty = 1.0 - torch.sigmoid(product)
    weight = torch.abs(y_true) ** b
    mean_weight = weight.mean() + eps
    normalized_weight = weight / mean_weight
    weighted_penalty = dir_penalty * normalized_weight
    mag_term = (y_true - y_pred) ** 2
    return (weighted_penalty + mag_term).mean()
```

**M2 (Hybrid Multiplicative with λ_dir=2.0):**
```python
def hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0, delta=1.0):
    dir_term = _normalized_direction_term(y_true, y_pred)
    mag_term = huber_loss(y_true, y_pred, delta=delta)
    return (dir_term * mag_term * lambda_dir).mean()
```

**Linear Combination:**
```python
def imadl_m2_linear_loss(y_true, y_pred, alpha):
    imadl = imadl_loss(y_true, y_pred)
    m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    return alpha * imadl + (1 - alpha) * m2
```

---

## Rationale

### Why This Combination?

1. **IMADL Strengths:**
   - Most stable across seeds (CV=0.892)
   - Consistent positive returns (Sharpe=0.464)
   - Low failure rate (0%)
   - Balanced directional and magnitude terms

2. **M2 Strengths:**
   - Highest average Sharpe (0.914)
   - Best-case performance (Sharpe=2.285 on seed=62)
   - Strong directional signal (λ_dir=2.0)

3. **Complementary Properties:**
   - IMADL: Additive structure (dir + mag)
   - M2: Multiplicative structure (dir × mag × λ)
   - Linear combination allows smooth interpolation

### Expected Behavior by α

| α | IMADL Weight | M2 Weight | Expected Behavior |
|---|-------------|-----------|-------------------|
| 0.2 | 20% | 80% | Aggressive, high variance, high potential return |
| 0.3 | 30% | 70% | Moderately aggressive |
| 0.4 | 40% | 60% | Balanced toward M2 |
| 0.5 | 50% | 50% | Equal weighting |
| 0.6 | 60% | 40% | Balanced toward IMADL |
| 0.7 | 70% | 30% | Moderately conservative |
| 0.8 | 80% | 20% | Conservative, low variance, stable |

**Hypothesis:** Optimal α will be in range [0.4, 0.6], balancing IMADL's stability with M2's high returns.

---

## Parameter Grid

### 7 Configurations

1. **imadl_m2_linear_02** (α=0.2): 20% IMADL, 80% M2
2. **imadl_m2_linear_03** (α=0.3): 30% IMADL, 70% M2
3. **imadl_m2_linear_04** (α=0.4): 40% IMADL, 60% M2
4. **imadl_m2_linear_05** (α=0.5): 50% IMADL, 50% M2
5. **imadl_m2_linear_06** (α=0.6): 60% IMADL, 40% M2
6. **imadl_m2_linear_07** (α=0.7): 70% IMADL, 30% M2
7. **imadl_m2_linear_08** (α=0.8): 80% IMADL, 20% M2

---

## Code Implementation

### Add to `losses.py`

```python
# In EXPERIMENT_LOSS_NAMES tuple
EXPERIMENT_LOSS_NAMES = (
    # ... existing losses ...
    "imadl_m2_linear_02",
    "imadl_m2_linear_03",
    "imadl_m2_linear_04",
    "imadl_m2_linear_05",
    "imadl_m2_linear_06",
    "imadl_m2_linear_07",
    "imadl_m2_linear_08",
)

# In get_experiment_loss_fn() function
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... existing loss implementations ...
    
    # Variant 1: IMADL + M2 Linear Combination
    if name_lower == "imadl_m2_linear_02":
        return lambda y_true, y_pred: (
            0.2 * imadl_loss(y_true, y_pred) + 
            0.8 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_03":
        return lambda y_true, y_pred: (
            0.3 * imadl_loss(y_true, y_pred) + 
            0.7 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_04":
        return lambda y_true, y_pred: (
            0.4 * imadl_loss(y_true, y_pred) + 
            0.6 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_05":
        return lambda y_true, y_pred: (
            0.5 * imadl_loss(y_true, y_pred) + 
            0.5 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_06":
        return lambda y_true, y_pred: (
            0.6 * imadl_loss(y_true, y_pred) + 
            0.4 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_07":
        return lambda y_true, y_pred: (
            0.7 * imadl_loss(y_true, y_pred) + 
            0.3 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
    
    if name_lower == "imadl_m2_linear_08":
        return lambda y_true, y_pred: (
            0.8 * imadl_loss(y_true, y_pred) + 
            0.2 * hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
        )
```

### Create Runner Scripts

Create 7 files: `run_sanity_check_imadl_m2_linear_02.py` through `run_sanity_check_imadl_m2_linear_08.py`

**Template:**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (IMADL+M2 Linear α=0.2)")
    args = parser.parse_args()
    run_sanity_check("imadl_m2_linear_02", args)

if __name__ == "__main__":
    main()
```

### Update `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... existing mappings ...
    "imadl_m2_linear_02": "run_sanity_check_imadl_m2_linear_02.py",
    "imadl_m2_linear_03": "run_sanity_check_imadl_m2_linear_03.py",
    "imadl_m2_linear_04": "run_sanity_check_imadl_m2_linear_04.py",
    "imadl_m2_linear_05": "run_sanity_check_imadl_m2_linear_05.py",
    "imadl_m2_linear_06": "run_sanity_check_imadl_m2_linear_06.py",
    "imadl_m2_linear_07": "run_sanity_check_imadl_m2_linear_07.py",
    "imadl_m2_linear_08": "run_sanity_check_imadl_m2_linear_08.py",
}
```

---

## Experiment Configuration

### Phase 2.1: Initial Screening

**Runs:** 7 losses × 3 seeds = 21 runs

| Loss Name | α | Seeds | Weight Cap | Test Period |
|-----------|---|-------|------------|-------------|
| imadl_m2_linear_02 | 0.2 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_03 | 0.3 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_04 | 0.4 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_05 | 0.5 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_06 | 0.6 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_07 | 0.7 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_m2_linear_08 | 0.8 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |

---

## Expected Results

### Performance Predictions

**Low α (0.2-0.3): M2-Dominant**
- Expected Sharpe: 0.7-1.0
- Expected CV: 1.2-1.5
- Failure rate: 20-30%
- Behavior: High variance, high potential return

**Medium α (0.4-0.6): Balanced**
- Expected Sharpe: 0.6-0.8
- Expected CV: 0.8-1.1
- Failure rate: 10-20%
- Behavior: Optimal balance point

**High α (0.7-0.8): IMADL-Dominant**
- Expected Sharpe: 0.5-0.6
- Expected CV: 0.7-0.9
- Failure rate: 0-10%
- Behavior: Conservative, stable

### Success Criteria

**Minimum Requirements:**
- Average Sharpe > 0.5 (better than IMADL baseline)
- CV < 1.0 (better than M2 baseline)
- Failure rate < 20%

**Ideal Performance:**
- Average Sharpe > 0.7
- CV < 0.9
- Failure rate < 10%
- Consistent performance across all 3 seeds

---

## Analysis Plan

### Metrics to Track

1. **Primary Metrics:**
   - Average Sharpe ratio across 3 seeds
   - Sharpe standard deviation
   - Coefficient of variation (CV)
   - Failure rate (Sharpe < 0)

2. **Secondary Metrics:**
   - Cumulative return
   - Maximum drawdown
   - Monthly return volatility
   - Directional accuracy

3. **Comparison Metrics:**
   - vs IMADL baseline (Sharpe=0.464, CV=0.892)
   - vs M2 baseline (Sharpe=0.914, CV=1.396)
   - Improvement percentage

### Visualization

1. **α vs Sharpe Plot:** Show how Sharpe changes with α
2. **α vs CV Plot:** Show stability trend
3. **Pareto Frontier:** Plot Sharpe vs CV to find optimal trade-off
4. **Seed Sensitivity:** Box plots showing variance across seeds for each α

---

## Thesis Contribution

### Methodological Innovation

- **Novel combination strategy:** First systematic exploration of linear IMADL+M2 interpolation
- **Parameter sweep:** Comprehensive α grid search to find optimal balance
- **Stability-return trade-off:** Quantify the Pareto frontier

### Expected Findings

1. **Optimal α identification:** Likely in range [0.4, 0.6]
2. **Smooth interpolation:** Linear combination provides continuous trade-off
3. **Robustness improvement:** Reduced variance compared to pure M2
4. **Performance boost:** Higher Sharpe than pure IMADL

### Theoretical Insights

- Why linear combinations work for loss functions
- Trade-offs between additive (IMADL) and multiplicative (M2) structures
- Role of α in controlling risk-return profile

---

**Document Version:** v1.0  
**Created:** 2026-04-26  
**Author:** Yirong Yu  
**Status:** Ready for Implementation
