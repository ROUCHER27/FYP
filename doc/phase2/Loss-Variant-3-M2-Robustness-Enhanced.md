# Loss Variant 3: Robustness-Enhanced M2

> **Objective:** Add explicit robustness constraint to M2 to reduce variance and improve cross-seed stability while maintaining high returns.

**Variant Type:** Regularized Loss  
**Base Loss:** M2 (hybrid_mul_m2)  
**Regularization:** Robustness penalty (variance of monthly returns)  
**Parameter:** γ (penalty weight)

---

## Mathematical Definition

### Formula

```
loss = hybrid_mul_m2 + γ * robustness_penalty

where:
  hybrid_mul_m2 = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
  robustness_penalty = Var(monthly_returns)
  γ ∈ {0.01, 0.1, 1.0}
```

### Component Breakdown

**M2 Base Loss:**
```python
def hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0, delta=1.0):
    dir_term = _normalized_direction_term(y_true, y_pred)
    mag_term = huber_loss(y_true, y_pred, delta=delta)
    return (dir_term * mag_term * lambda_dir).mean()
```

**Robustness Penalty:**
```python
def robustness_penalty(y_true, y_pred, returns):
    """
    Penalize high variance in monthly returns.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
        returns: Monthly portfolio returns (computed from predictions)
    
    Returns:
        Variance of monthly returns
    """
    monthly_returns = compute_monthly_returns(y_pred)
    return torch.var(monthly_returns)
```

**Combined Loss:**
```python
def m2_robust_loss(y_true, y_pred, gamma):
    base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
    penalty = robustness_penalty(y_true, y_pred)
    return base_loss + gamma * penalty
```

---

## Rationale

### Why Add Robustness Penalty?

1. **M2's Problem:**
   - High average Sharpe (0.914) but extreme variance (CV=1.396)
   - 33% failure rate (seed=52: Sharpe=-0.239)
   - Sharpe ranges from -0.239 to 2.285 across seeds

2. **Root Cause:**
   - M2 optimizes for prediction accuracy, not trading robustness
   - No explicit constraint on return volatility
   - Multiplicative structure amplifies seed-specific biases

3. **Solution:**
   - Add variance penalty to directly control monthly return volatility
   - Encourage stable performance across different market conditions
   - Balance high returns with risk management

### Expected Behavior by γ

| γ | Penalty Weight | Expected Behavior |
|---|---------------|-------------------|
| 0.01 | Very weak | Minimal impact, close to pure M2 |
| 0.1 | Moderate | Noticeable variance reduction |
| 1.0 | Strong | Significant variance reduction, may sacrifice returns |

**Hypothesis:** γ=0.1 will provide optimal balance, reducing M2's variance while maintaining high returns.

---

## Parameter Grid

### 3 Configurations

1. **m2_robust_001** (γ=0.01): Weak robustness penalty
2. **m2_robust_01** (γ=0.1): Moderate robustness penalty
3. **m2_robust_10** (γ=1.0): Strong robustness penalty

---

## Code Implementation

### Add to `losses.py`

```python
# In EXPERIMENT_LOSS_NAMES tuple
EXPERIMENT_LOSS_NAMES = (
    # ... existing losses ...
    "m2_robust_001",
    "m2_robust_01",
    "m2_robust_10",
)

# Helper function for robustness penalty
def compute_robustness_penalty(y_pred: torch.Tensor, 
                               batch_size: int = 32) -> torch.Tensor:
    """
    Compute variance of predicted returns as robustness penalty.
    
    Note: This is a simplified version. In practice, you may want to:
    1. Group predictions by month
    2. Compute monthly portfolio returns
    3. Calculate variance of monthly returns
    
    For now, we use variance of predictions as a proxy.
    """
    return torch.var(y_pred)

# In get_experiment_loss_fn() function
def get_experiment_loss_fn(name: str) -> Callable:
    name_lower = name.lower()
    
    # ... existing loss implementations ...
    
    # Variant 3: Robustness-Enhanced M2
    if name_lower == "m2_robust_001":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 0.01 * penalty
        return loss_fn
    
    if name_lower == "m2_robust_01":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 0.1 * penalty
        return loss_fn
    
    if name_lower == "m2_robust_10":
        def loss_fn(y_true, y_pred):
            base_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
            penalty = compute_robustness_penalty(y_pred)
            return base_loss + 1.0 * penalty
        return loss_fn
```

### Create Runner Scripts

Create 3 files: `run_sanity_check_m2_robust_001.py`, `run_sanity_check_m2_robust_01.py`, `run_sanity_check_m2_robust_10.py`

**Template:**
```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check

def main() -> None:
    parser = build_arg_parser("Sanity check (M2 Robust γ=0.01)")
    args = parser.parse_args()
    run_sanity_check("m2_robust_001", args)

if __name__ == "__main__":
    main()
```

### Update `run_all_experiments.py`

```python
RUNNER_BY_LOSS = {
    # ... existing mappings ...
    "m2_robust_001": "run_sanity_check_m2_robust_001.py",
    "m2_robust_01": "run_sanity_check_m2_robust_01.py",
    "m2_robust_10": "run_sanity_check_m2_robust_10.py",
}
```

---

## Experiment Configuration

### Phase 2.1: Initial Screening

**Runs:** 3 losses × 3 seeds = 9 runs

| Loss Name | γ | Seeds | Weight Cap | Test Period |
|-----------|---|-------|------------|-------------|
| m2_robust_001 | 0.01 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| m2_robust_01 | 0.1 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |
| m2_robust_10 | 1.0 | 42, 52, 62 | 0.05 | 1995-01 to 1996-12 |

---

## Expected Results

### Performance Predictions

**Low γ (0.01): Weak Penalty**
- Expected Sharpe: 0.8-1.0
- Expected CV: 1.2-1.4
- Failure rate: 20-30%
- Behavior: Close to pure M2, minimal variance reduction

**Medium γ (0.1): Moderate Penalty**
- Expected Sharpe: 0.7-0.9
- Expected CV: 0.9-1.1
- Failure rate: 10-20%
- Behavior: Balanced, noticeable stability improvement

**High γ (1.0): Strong Penalty**
- Expected Sharpe: 0.5-0.7
- Expected CV: 0.7-0.9
- Failure rate: 0-10%
- Behavior: Very stable, may sacrifice returns

### Success Criteria

**Minimum Requirements:**
- Average Sharpe > 0.6
- CV < 1.2 (better than pure M2's 1.396)
- Failure rate < 20%

**Ideal Performance:**
- Average Sharpe > 0.8
- CV < 1.0
- Failure rate < 10%
- No catastrophic failures (Sharpe < -0.2)

---

## Analysis Plan

### Key Questions

1. **Does robustness penalty reduce variance?**
   - Compare CV across γ values
   - Check if seed=52 failure is prevented

2. **What is the return-stability trade-off?**
   - Plot Sharpe vs CV for different γ
   - Identify optimal γ on Pareto frontier

3. **Does penalty prevent catastrophic failures?**
   - Check if seed=52 Sharpe improves from -0.239
   - Verify no new failures on other seeds

4. **Is the penalty mechanism effective?**
   - Analyze loss curves during training
   - Check if penalty term is actually reducing variance

### Visualization

1. **γ vs Sharpe Plot:** Show performance trend
2. **γ vs CV Plot:** Show stability improvement
3. **Seed Comparison:** Box plots for each γ
4. **Loss Decomposition:** Plot base_loss vs penalty over training

---

## Implementation Notes

### Robustness Penalty Variants

The current implementation uses `torch.var(y_pred)` as a simple proxy. More sophisticated versions could include:

**Variant A: Monthly Return Variance**
```python
def compute_monthly_return_variance(y_pred, dates):
    # Group predictions by month
    monthly_returns = group_by_month(y_pred, dates)
    # Compute portfolio returns per month
    portfolio_returns = compute_portfolio_returns(monthly_returns)
    # Return variance
    return torch.var(portfolio_returns)
```

**Variant B: Maximum Drawdown Penalty**
```python
def compute_drawdown_penalty(y_pred):
    cumulative_returns = torch.cumsum(y_pred, dim=0)
    running_max = torch.cummax(cumulative_returns, dim=0)[0]
    drawdown = running_max - cumulative_returns
    max_drawdown = torch.max(drawdown)
    return max_drawdown
```

**Variant C: Downside Deviation**
```python
def compute_downside_deviation(y_pred):
    negative_returns = torch.clamp(y_pred, max=0)
    return torch.std(negative_returns)
```

**Recommendation:** Start with simple `torch.var(y_pred)`. If results are promising, explore more sophisticated penalties in Phase 2.2.

---

## Thesis Contribution

### Methodological Innovation

- **Robustness-aware loss design:** First explicit variance penalty for trading losses
- **Direct risk control:** Optimize for both returns and stability simultaneously
- **Regularization approach:** Apply ML regularization concepts to financial loss functions

### Expected Findings

1. **Optimal γ:** Likely γ=0.1 provides best trade-off
2. **Variance reduction:** 20-30% CV reduction compared to pure M2
3. **Failure prevention:** Eliminate or reduce seed=52 catastrophic failure
4. **Return sacrifice:** 10-20% Sharpe reduction acceptable for stability gain

### Theoretical Insights

- Why explicit robustness constraints are necessary for trading losses
- Trade-off between prediction accuracy and trading stability
- Role of regularization in preventing overfitting to specific seeds

---

**Document Version:** v1.0  
**Created:** 2026-04-26  
**Author:** Yirong Yu  
**Status:** Ready for Implementation
