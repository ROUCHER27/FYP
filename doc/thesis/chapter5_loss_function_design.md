# Chapter 5: Loss Function Design

## 5.1 Design Principles from Semester 1 Findings

The experimental results from Phase 1 (Chapter 4) revealed critical limitations in existing directional loss functions, particularly GMADL, while also highlighting the importance of robustness through MedSE's superior performance. These findings established the foundation for our loss function design principles.

### 5.1.1 GMADL Limitations

Phase 1 experiments identified three fundamental issues with GMADL:

1. **Symmetry Problem**: The reward and penalty magnitudes are equal. When a prediction is correct (same sign as actual), the reward magnitude equals the penalty for an incorrect prediction of the same absolute error. This fails to sufficiently penalize directional errors in financial forecasting where wrong-direction predictions can lead to catastrophic losses.

2. **Weak Signal at Critical Points**: As the prediction approaches zero (ŷ→0), the gradient signal weakens dramatically. The sigmoid-based directional term becomes less sensitive precisely when the model needs strong guidance to avoid sign errors. This creates a "dead zone" where the loss function provides insufficient feedback.

3. **Ignoring Prediction Precision**: GMADL only considers prediction direction and the magnitude of the actual return |y|, but completely ignores the prediction error magnitude |y-ŷ|. Two predictions with the same direction but vastly different errors receive similar loss values, failing to incentivize accurate magnitude estimation.

### 5.1.2 MedSE Robustness Finding

Phase 1 results showed MedSE achieving a Sharpe ratio of 2.68, dramatically outperforming MSE (0.37) and all directional losses. This 7.2× improvement demonstrated that robustness to outliers is critical for financial return prediction. The median-based approach effectively handles the heavy-tailed distribution of stock returns without sacrificing predictive accuracy.

### 5.1.3 Design Principles

Based on these findings, we established four core design principles for our hybrid loss functions:

1. **Address GMADL's Three Issues**: Any proposed loss must explicitly tackle the symmetry problem, strengthen signals at critical points, and incorporate prediction precision.

2. **Incorporate Robustness**: Following MedSE's success, robustness to outliers must be a first-class design consideration, not an afterthought.

3. **Maintain Computational Tractability**: Loss functions must remain O(n) per batch without requiring pairwise comparisons or complex sampling procedures that would significantly increase training time.

4. **Provide Clear Gradients**: The loss landscape must provide consistent, interpretable gradients for effective neural network training, avoiding pathological cases like vanishing or exploding gradients.

These principles guided all subsequent loss function design decisions in Phase 2.

## 5.2 Supervisor Guidance Integration

The pre-plan guidance from the interim report (Sections 3.2.1 and 3.2.2) provided specific directions for addressing GMADL's limitations:

### 5.2.1 Improved MADL Approach

The supervisor suggested enhancing MADL by adding a magnitude error term to address the precision issue (Issue 3). This led to the IMADL formulation:

```math
L_i = (1-\sigma(a \cdot y_i \hat{y}_i)) |y_i|^b + c |y_i - \hat{y}_i|^d
```

where a=100, b=2, c=1, d=2.

The first term `(1-σ(a·y·ŷ))|y|^b` provides directional penalty, while the second term `c|y-ŷ|^d` explicitly penalizes magnitude errors. This two-component structure directly addresses Issue 3 by making prediction precision a primary optimization target.

### 5.2.2 Directional Huber Approach

The supervisor also suggested combining directional penalties with the Huber loss, which is inherently robust to outliers. This led to the M2 (Directional Huber) formulation:

```math
L_i = (1 + \lambda_{\text{dir}} \cdot \text{dir\_term}_i) \times H_\delta(y_i - \hat{y}_i)
```

where `dir_term = 1 - sigmoid(100 * y * ŷ)` and `H_δ` is the Huber loss with δ=1.0.

This multiplicative structure amplifies the base Huber loss when predictions have the wrong direction, while the Huber component ensures robustness to outliers.

### 5.2.3 Key Insight: Combining Both Approaches

The supervisor's critical suggestion was to "consider combining both approaches." This guidance shaped the Phase 2 experimental design, leading to four hybrid variant families that blend IMADL's explicit magnitude penalty with M2's robust directional amplification.

## 5.3 Candidate Loss Functions

### 5.3.1 Baseline Losses

We established seven baseline losses from Phase 1 to provide comparison benchmarks:

**1. Mean Squared Error (MSE)**

The standard regression loss serving as our reference baseline:

```math
L_i = (y_i - \hat{y}_i)^2
```

MSE treats all errors equally and provides no directional bias, making it suitable as a neutral baseline for comparison.

**2. Median Squared Error (MedSE)**

Our robust baseline using median instead of mean:

```math
L = \text{median}_{i} (y_i - \hat{y}_i)^2
```

MedSE demonstrated exceptional performance in Phase 1 (Sharpe 2.68), validating the importance of robustness in financial return prediction.

**3. Mean Absolute Directional Loss (MADL)**

The original directional loss using tanh activation:

```math
L_i = -\tanh(25 \cdot y_i \cdot \hat{y}_i) \cdot |y_i|
```

MADL penalizes wrong-direction predictions but suffers from all three GMADL issues.

**4. Generalized Mean Absolute Directional Loss (GMADL)**

An improved version using sigmoid activation:

```math
L_i = -[\sigma(100 \cdot y_i \cdot \hat{y}_i) - 0.5] \times |y_i|^2
```

GMADL provides smoother gradients than MADL but still exhibits the three fundamental limitations identified in Section 5.1.1.

**5. Improved Mean Absolute Directional Loss (IMADL)**

The supervisor-suggested enhancement adding magnitude penalty:

```math
L_i = (1-\sigma(a \cdot y_i \hat{y}_i)) |y_i|^b + c |y_i - \hat{y}_i|^d
```

where a=100, b=2, c=1, d=2. This formulation directly addresses Issue 3 by incorporating the explicit magnitude error term `|y_i - ŷ_i|^d`.

**6. M1 (Directional MSE)**

A simple multiplicative combination of directional penalty with MSE:

```math
L_i = (1 + \lambda_{\text{dir}} \cdot \text{dir\_term}_i) \times (y_i - \hat{y}_i)^2
```

M1 served as an initial exploration of multiplicative directional amplification but lacks robustness to outliers.

**7. M2 (Directional Huber)**

The core hybrid loss combining directional penalty with robust Huber loss:

```math
L_i = (1 + \lambda_{\text{dir}} \cdot \text{dir\_term}_i) \times H_\delta(y_i - \hat{y}_i)
```

where `dir_term = 1 - sigmoid(100 * y * ŷ)` and `H_δ(e)` is the Huber loss with δ=1.0.

### 5.3.2 Directional Huber (M2) - Core Design

M2 represents our primary hybrid approach, warranting detailed examination of its design rationale and properties.

**Mathematical Formulation**

The Huber loss component is defined as:

```math
H_\delta(e) = \begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
```

where e = y_i - ŷ_i and δ=1.0. This provides quadratic loss for small errors (|e| ≤ δ) and linear loss for large errors (|e| > δ), achieving robustness to outliers while maintaining sensitivity to small errors.

The directional term is:

```math
\text{dir\_term}_i = 1 - \sigma(100 \cdot y_i \cdot \hat{y}_i)
```

This evaluates to approximately 0 when the prediction has the correct sign (y·ŷ > 0) and approximately 1 when the sign is wrong (y·ŷ < 0).

**Design Rationale**

M2 addresses all three GMADL issues and incorporates robustness:

1. **Addresses Issue 3 (Precision)**: The Huber term `H_δ(y_i - ŷ_i)` directly penalizes magnitude errors, ensuring the model optimizes for prediction accuracy, not just direction.

2. **Addresses Robustness**: Huber loss is inherently robust to outliers through its linear behavior for large errors, validating the MedSE finding from Phase 1.

3. **Multiplicative Structure**: Wrong-direction predictions amplify the loss by a factor of (1 + λ_dir), creating asymmetric penalties that strongly discourage sign errors.

4. **Addresses Issue 2 (Weak Signals)**: The Huber component provides consistent gradients even when ŷ→0, as the magnitude error term remains active regardless of prediction magnitude.

**Lambda Parameter Exploration**

The λ_dir parameter controls the strength of directional penalty amplification:

- **λ_dir = 1.0**: Wrong-direction predictions receive 2× penalty (baseline M2)
- **λ_dir = 2.0**: 3× penalty (Phase 2 M1 configuration)
- **λ_dir = 5.0**: 6× penalty (Phase 1.5 M2 exploration)

Phase 1.5 experiments explored this parameter space to identify optimal directional penalty strength (detailed results in Chapter 6).

### 5.3.3 Hybrid Variants (Phase 2 Design)

Following the supervisor's guidance to "combine both approaches," we designed four variant families that blend different loss components:

**Variant 1: IMADL + M2 Linear Combination**

```math
L = \alpha \cdot L_{\text{IMADL}} + (1-\alpha) \cdot L_{\text{M2}}
```

This variant uses alpha blending to combine IMADL's explicit magnitude penalty with M2's robust directional amplification. The parameter α ∈ [0,1] controls the balance between the two approaches.

**Design rationale**: Linear combination allows the model to benefit from both IMADL's dual-component structure and M2's robustness. Different α values emphasize different aspects of the hybrid loss.

**Variant 2: IMADL + GMADL Weighted**

```math
L = \alpha \cdot L_{\text{IMADL}} + (1-\alpha) \cdot L_{\text{GMADL}}
```

This variant attempts to combine IMADL with the original GMADL formulation.

**Design rationale**: Test whether IMADL's magnitude penalty can compensate for GMADL's limitations when blended. This variant ultimately failed (detailed in Chapter 6), as GMADL's fundamental issues dominated the blend regardless of α.

**Variant 3: M2 + Robustness Enhancements** ⭐

```math
L_i = (1 + \lambda_{\text{dir}} \cdot \text{dir\_term}_i) \times H_\delta(y_i - \hat{y}_i) + \gamma \cdot R_i
```

where R_i is a robustness enhancement term and γ controls its strength.

**Design rationale**: Extend M2's inherent robustness (from Huber loss) with additional robustness mechanisms. The R_i term can implement batch normalization, adaptive weighting, or other robustness-enhancing techniques.

**Implementation**: The robustness term R_i was implemented as a batch-normalized error component that adapts to the scale and distribution of errors within each training batch. This provides additional stability beyond Huber's outlier resistance.

**Significance**: This variant proved most promising in Phase 2 experiments, leading to the Phase 2.2 gamma refinement study (detailed in Chapter 6).

**Variant 4: Adaptive Hybrid**

```math
L_i = w_i(\hat{y}_i) \cdot L_{\text{IMADL},i} + (1-w_i(\hat{y}_i)) \cdot L_{\text{M2},i}
```

where w_i is a dynamic weighting function based on prediction confidence.

**Design rationale**: Adaptively blend IMADL and M2 based on the model's confidence in each prediction. High-confidence predictions use more directional penalty (IMADL), while low-confidence predictions rely more on robust Huber (M2).

**Implementation**: The weighting function w_i was based on prediction magnitude and variance estimates from the model's output distribution.

**Outcome**: This variant failed due to high variance and unstable training dynamics (detailed in Chapter 6). The adaptive weighting introduced additional complexity that destabilized the optimization process.

## 5.4 Why Phase 2 Approach Over Ranking-Aware Loss

The pre-plan (Section 3.3) mentioned ranking-aware losses as a potential research direction. However, we chose to focus on hybrid directional-robust losses instead. This section justifies that decision.

### 5.4.1 Computational Complexity Concerns

Ranking-aware losses typically require pairwise or listwise comparisons to optimize for relative ordering rather than absolute prediction accuracy. For a batch of size n, this introduces O(n²) or O(n log n) complexity compared to O(n) for pointwise losses.

**Impact on training time**: With our dataset containing ~500,000 training samples and typical batch sizes of 256-512, pairwise ranking losses would increase training time by an estimated 50-100×, making extensive hyperparameter exploration infeasible within the project timeline.

### 5.4.2 Direct Problem Addressing

The hybrid approach directly addresses the three identified GMADL issues:

1. **Issue 3 (Precision)**: Huber and magnitude penalty terms explicitly optimize for |y-ŷ|
2. **Issue 2 (Weak Signals)**: Magnitude-based components provide consistent gradients
3. **Issue 1 (Symmetry)**: Multiplicative directional amplification creates asymmetric penalties

Ranking losses, while potentially beneficial for portfolio construction, do not directly address these fundamental loss function design issues.

### 5.4.3 Supervisor Guidance Priority

The supervisor's guidance (Sections 3.2.1 and 3.2.2) explicitly prioritized:
1. Adding magnitude error terms to MADL (→ IMADL)
2. Combining directional penalties with Huber loss (→ M2)
3. Exploring combinations of both approaches (→ Phase 2 variants)

This guidance provided a clear research direction that we followed systematically. Ranking losses were mentioned as a potential extension but not prioritized in the immediate research plan.

### 5.4.4 Foundation First Philosophy

Establishing a strong hybrid baseline before exploring more complex ranking methods follows sound research methodology. The hybrid directional-robust approach provides:

1. **Interpretable components**: Each term has clear meaning and purpose
2. **Modular design**: Components can be analyzed and improved independently
3. **Baseline for comparison**: Future ranking-aware extensions can be compared against this foundation

### 5.4.5 Scope Management

Given the project timeline and the need for extensive hyperparameter exploration (103 total experimental runs), focusing on one research direction allowed for thorough investigation rather than superficial coverage of multiple approaches.

**Conclusion**: Ranking-aware losses represent a natural extension for future work (discussed in Chapter 8), but the hybrid directional-robust approach was the appropriate focus for this project given computational constraints, supervisor guidance, and research methodology considerations.

## 5.5 Hyperparameter Design

### 5.5.1 Lambda Ratios (λ_dir : λ_hub)

The λ_dir parameter controls the strength of directional penalty amplification in M1 and M2 losses. Phase 1.5 explored four values:

**λ_dir = 0.5**: Minimal directional penalty (1.5× amplification for wrong direction)
- Hypothesis: Gentle directional bias while maintaining robustness
- Expected behavior: Similar to base Huber with slight directional preference

**λ_dir = 1.0**: Moderate directional penalty (2× amplification)
- Hypothesis: Balanced approach between direction and magnitude
- Expected behavior: Significant directional bias without overwhelming magnitude optimization

**λ_dir = 2.0**: Strong directional penalty (3× amplification)
- Hypothesis: Aggressive directional enforcement while maintaining gradient stability
- Expected behavior: Strong preference for correct sign predictions

**λ_dir = 5.0**: Very strong directional penalty (6× amplification)
- Hypothesis: Maximum directional emphasis, potentially at cost of magnitude accuracy
- Expected behavior: Extreme directional bias, risk of overfitting to sign prediction

**Selection rationale**: The range [0.5, 5.0] spans from minimal to extreme directional bias, allowing identification of the optimal balance point. Phase 1.5 results (Chapter 6) showed λ_dir = 2.0 as the optimal balance for Phase 2 exploration.

### 5.5.2 Gamma for Robustness (Phase 2.2)

The γ parameter controls the strength of robustness enhancement in Variant 3 (M2 + Robustness). Phase 2.2 explored five values:

**γ = 0.5**: Minimal robustness enhancement
- Hypothesis: Slight improvement over base M2
- Expected behavior: Marginal stability gains

**γ = 0.7**: Moderate-low robustness enhancement
- Hypothesis: Balanced robustness without overwhelming directional signal
- Expected behavior: Improved stability with maintained performance

**γ = 1.0**: Equal weighting of robustness term
- Hypothesis: Significant robustness gains
- Expected behavior: Strong stability, potential performance improvement

**γ = 1.5**: High robustness enhancement
- Hypothesis: Maximum stability at potential cost of performance
- Expected behavior: Very stable but possibly over-regularized

**γ = 2.0**: Very high robustness enhancement
- Hypothesis: Excessive robustness may degrade signal
- Expected behavior: High stability but likely performance degradation

**Selection rationale**: The range [0.5, 2.0] explores from minimal to potentially excessive robustness enhancement. Phase 2.2 results (Chapter 6) identified γ = 0.7 as the optimal sweet spot balancing performance and stability.

### 5.5.3 Alpha for IMADL-M2 Blending

The α parameter controls the linear combination weight in Variant 1. Phase 2 explored four values:

**α = 0.2**: M2-dominant blend (80% M2, 20% IMADL)
- Hypothesis: Primarily robust Huber with IMADL refinement
- Expected behavior: Similar to M2 with slight IMADL influence

**α = 0.4**: M2-leaning blend (60% M2, 40% IMADL)
- Hypothesis: Balanced toward M2's robustness
- Expected behavior: Moderate influence from both components

**α = 0.6**: IMADL-leaning blend (60% IMADL, 40% M2)
- Hypothesis: Balanced toward IMADL's dual-component structure
- Expected behavior: Strong influence from both components

**α = 0.8**: IMADL-dominant blend (80% IMADL, 20% M2)
- Hypothesis: Primarily IMADL with M2 robustness enhancement
- Expected behavior: Similar to IMADL with slight M2 influence

**Selection rationale**: The range [0.2, 0.8] avoids extreme endpoints (pure IMADL or pure M2) to focus on true hybrid behavior. Phase 2 results (Chapter 6) showed α = 0.6 as optimal.

## 5.6 Implementation Considerations

### 5.6.1 Numerical Stability

All loss implementations incorporate numerical stability measures:

**Epsilon values**: Division operations include ε = 1e-8 to prevent division by zero:
```python
safe_denominator = denominator + 1e-8
```

**Sigmoid saturation handling**: The directional term uses sigmoid(100·y·ŷ), which can saturate for large products. PyTorch's built-in sigmoid handles this gracefully through numerically stable implementations.

**Gradient clipping consideration**: While gradient clipping was considered for extreme cases, it was not implemented in the final design. The Huber loss's linear behavior for large errors naturally prevents exploding gradients, and the multiplicative structure of M2 maintains bounded gradients.

### 5.6.2 Batch Normalization for Scale Balance

The robustness enhancement term R_i in Variant 3 uses batch normalization to adapt to error scale and distribution:

```python
batch_mean = errors.mean()
batch_std = errors.std() + 1e-8
normalized_errors = (errors - batch_mean) / batch_std
R_i = normalized_errors ** 2
```

This ensures the robustness term operates at a consistent scale across different batches and training phases, preventing scale mismatch between the directional and robustness components.

### 5.6.3 Computational Efficiency

All proposed losses maintain O(n) complexity per batch:

**Pointwise operations**: All loss components (sigmoid, Huber, magnitude penalties) operate element-wise on prediction-target pairs.

**No pairwise comparisons**: Unlike ranking losses, our hybrid losses do not require comparing predictions across samples, avoiding O(n²) complexity.

**GPU-friendly operations**: All operations (multiplication, addition, sigmoid, absolute value) are highly optimized in PyTorch and execute efficiently on GPU hardware.

**Training time impact**: Compared to MSE baseline, hybrid losses add negligible computational overhead (<5% increase in training time per epoch).

## 5.7 Summary

This chapter presented the systematic design of hybrid loss functions for financial return prediction:

1. **Foundation from Phase 1 findings**: Identified three critical GMADL limitations (symmetry, weak signals, ignoring precision) and validated the importance of robustness through MedSE's superior performance.

2. **Supervisor guidance integration**: Incorporated specific directions to enhance MADL with magnitude penalties (→ IMADL) and combine directional penalties with Huber loss (→ M2), with the key insight to explore combinations of both approaches.

3. **Four hybrid variant families**: Designed systematic variants blending IMADL and M2 through different mechanisms:
   - Variant 1: Linear combination (alpha blending)
   - Variant 2: IMADL + GMADL (failed due to GMADL's fundamental issues)
   - Variant 3: M2 + robustness enhancements (most promising)
   - Variant 4: Adaptive hybrid (failed due to high variance)

4. **Hyperparameter exploration strategy**: Established principled ranges for λ_dir (directional penalty strength), γ (robustness enhancement), and α (blend weight) based on theoretical considerations and preliminary experiments.

5. **Justified research focus**: Explained the decision to prioritize hybrid directional-robust losses over ranking-aware losses based on computational complexity, direct problem addressing, supervisor guidance, foundation-first philosophy, and scope management.

6. **Implementation considerations**: Addressed numerical stability, batch normalization for scale balance, and computational efficiency to ensure practical viability.

The design process followed a principled approach: identify problems (GMADL issues + robustness need) → incorporate guidance (IMADL + M2) → design variants (4 families) → plan exploration (hyperparameter ranges). This systematic methodology enabled comprehensive evaluation while maintaining computational tractability.

Variant 3 (M2 + robustness enhancements) emerged as the most promising direction, leading to focused gamma refinement in Phase 2.2. The next chapter presents experimental validation of these designs across 103 runs spanning five experimental phases.

