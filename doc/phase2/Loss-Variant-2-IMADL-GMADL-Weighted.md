# Loss Variant 2: IMADL + GMADL Weighted Combination

> **Objective:** Combine two MAD-based losses with similar structure to improve returns while maintaining stability.

**Variant Type:** Weighted Combination  
**Base Losses:** IMADL + GMADL (both MAD-based)  
**Parameter:** β (IMADL weight)

---

## Mathematical Definition

### Formula

```
loss = β * IMADL + (1-β) * GMADL

where:
  IMADL = imadl_loss(y_true, y_pred)
  GMADL = gmadl_loss(y_true, y_pred)
  β ∈ {0.3, 0.5, 0.7}
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

**GMADL (Generalized Mean Absolute Directional Loss):**
```python
def gmadl_loss(y_true, y_pred, a=100.0, b=2.0, eps=1e-8):
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    loss = -(sigmoid - 0.5) * torch.abs(y_true) ** b
    return loss.mean()
```

**Key Differences:**
- **IMADL:** Includes magnitude term (MSE), normalized weighting
- **GMADL:** Direction-only, no magnitude term, unnormalized weighting
- **Both:** Use sigmoid for directional signal, |y|^b for importance weighting

---

## Rationale

### Why This Combination?

1. **Structural Similarity:**
   - Both use sigmoid(a * y_true * y_pred) for directional signal
   - Both use |y_true|^b for importance weighting
   - Similar mathematical foundation (MAD-based)

2. **Complementary Properties:**
   - IMADL: Adds magnitude term → better prediction accuracy
   - GMADL: Pure directional focus → stronger trading signal
   - IMADL: Normalized weights → stable across samples
   - GMADL: Unnormalized weights → emphasizes large returns

3. **Phase 1.5 Performance:**
   - IMADL: Sharpe=0.464, CV=0.892 (stable)
   - GMADL: Sharpe=0.307, CV=1.168 (moderate)
   - Both have 0% failure rate

### Expected Behavior by β

| β | IMADL Weight | GMADL Weight | Expected Behavior |
|---|-------------|--------------|-------------------|
| 0.3 | 30% | 70% | GMADL-dominant, stronger directional signal |
| 0.5 | 50% | 50% | Equal weighting, balanced approach |
| 0.7 | 70% | 30% | IMADL-dominant, more stable |

**Hypothesis:** β=0.5 will provide best balance, combining IMADL's stability with GMADL's directional strength.

---

## Parameter Grid

### 3 Configurations

1. **imadl_gmadl_weighted_03** (β=0.3): 30% IMADL, 70% GMADL
2. **imadl_gmadl_weighted_05** (β=0.5): 50% IMADL, 50% GMADL
3. **imadl_gmadl_weighted_07** (β=0.7): 70% IMADL, 30% GMADL

---

## Code Implementation

### Add to `losses.py`

```python
# In EXPERIMENT_LOSS_NAMES tuple
EXPERIMENT_LOSS_NAMES = (
    # ... existing losses ...
    "imadl_gmadl_weighted_03",
    "imadl_gmadl_weighted_05",
    "imadl_gmadl_weighted_07",
)

# In get_experiment_loss_fn() function
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... existing loss implementations ...
    
    # Variant 2: IMADL + GMADL Weighted Combination
    if name_lower == "imadl_gmadl_weighted_03":
        return lambda y_true, y_pred: (
            0.3 * imadl_loss(y_true, y_pred) + 
            0.7 * gmadl_loss(y_true, y_pred)
        )
    
    if name_lower == "imadl_gmadl_weighted_05":
        return lambda y_true, y_pred: (
            0.5 * imadl_loss(y_true, y_pred) + 
            0.5 * gmadl_loss(y_true, y_pred)
        )
    
    if name_lower == "imadl_gmadl_weighted_07":
        return lambda y_true, y_pred: (
            0.7 * imadl_loss(y_true, y_pred) + 
            0.3 * gmadl_loss(y_true, y_pred)
        )
```

### Create Runner Scripts

Create 3 files: `run_sanity_check_imadl_gmadl_weighted_03.py`, `run_sanity_check_imadl_gmadl_weighted_05.py`, `run_sanity_check_imadl_gmadl_weighted_07.py`

**Template:**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (IMADL+GMADL Weighted β=0.3)")
    args = parser.parse_args()
    run_sanity_check("imadl_gmadl_weighted_03", args)

if __name__ == "__main__":
    main()
```

### Update `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... existing mappings ...
    "imadl_gmadl_weighted_03": "run_sanity_check_imadl_gmadl_weighted_03.py",
    "imadl_gmadl_weighted_05": "run_sanity_check_imadl_gmadl_weighted_05.py",
    "imadl_gmadl_weighted_07": "run_sanity_check_imadl_gmadl_weighted_07.py",
}
```

---

## Experiment Configuration

### Phase 2.1: Initial Screening

**Runs:** 3 losses × 3 seeds = 9 runs

| Loss Name | β | Seeds | Weight Cap | Test Period |
|-----------|---|-------|------------|-------------|
| imadl_gmadl_weighted_03 | 0.3 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_gmadl_weighted_05 | 0.5 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| imadl_gmadl_weighted_07 | 0.7 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |

---

## Expected Results

### Performance Predictions

**Low β (0.3): GMADL-Dominant**
- Expected Sharpe: 0.35-0.45
- Expected CV: 1.0-1.2
- Failure rate: 0-10%
- Behavior: Stronger directional signal, moderate stability

**Medium β (0.5): Balanced**
- Expected Sharpe: 0.40-0.50
- Expected CV: 0.9-1.1
- Failure rate: 0-5%
- Behavior: Optimal balance of direction and magnitude

**High β (0.7): IMADL-Dominant**
- Expected Sharpe: 0.45-0.55
- Expected CV: 0.8-1.0
- Failure rate: 0%
- Behavior: Most stable, closer to IMADL baseline

### Success Criteria

**Minimum Requirements:**
- Average Sharpe > 0.4 (better than GMADL baseline)
- CV < 1.0
- Failure rate < 10%

**Ideal Performance:**
- Average Sharpe > 0.5 (better than IMADL baseline)
- CV < 0.9
- Failure rate = 0%
- Consistent performance across all 3 seeds

---

## Analysis Plan

### Key Questions

1. **Does combination improve over individual losses?**
   - Compare vs IMADL (Sharpe=0.464) and GMADL (Sharpe=0.307)
   
2. **What is the optimal β?**
   - Which weighting provides best risk-adjusted returns?
   
3. **Does magnitude term matter?**
   - IMADL has MSE term, GMADL doesn't
   - Does adding magnitude improve trading performance?

4. **Is normalization important?**
   - IMADL normalizes weights, GMADL doesn't
   - Impact on stability?

### Visualization

1. **β vs Sharpe Plot:** Show performance trend
2. **β vs CV Plot:** Show stability trend
3. **Component Analysis:** Decompose loss into directional and magnitude terms
4. **Seed Sensitivity:** Compare variance across seeds

---

## Thesis Contribution

### Methodological Innovation

- **MAD-based combination:** First exploration of combining two MAD-based losses
- **Structural analysis:** Understand impact of magnitude term and normalization
- **Complementary strengths:** Leverage IMADL's stability and GMADL's directional focus

### Expected Findings

1. **Optimal β:** Likely β=0.5 or β=0.7
2. **Magnitude term importance:** Quantify impact of MSE term on trading performance
3. **Normalization effect:** Understand role of weight normalization in stability
4. **Modest improvement:** Expect 5-10% Sharpe improvement over IMADL

### Theoretical Insights

- Why MAD-based losses are more stable than hybrid losses
- Role of magnitude term in balancing direction and accuracy
- Impact of weight normalization on cross-seed robustness

---

**Document Version:** v1.0  
**Created:** 2026-04-26  
**Author:** Yirong Yu  
**Status:** Ready for Implementation
