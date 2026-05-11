# Appendix A: Loss Function Definitions and Gradients

This appendix expands the loss-family definitions from Chapter 3 §3.3 with explicit closed-form expressions and derivatives with respect to the prediction $\hat y$. Every formula in Sections A.1–A.3.2 matches the implementation in `Model_Train/losses.py` on the current `main` branch. Section A.3.3 records the conceptual form used to interpret Phase 2 branch outputs; its exact closed form lives on the `phase2.2-fix` branch. Gradients are stated pointwise; the effective training gradient at each optimiser step is the batch average (or median for `MedSE`) of the pointwise gradient.

Throughout this appendix, $y$ is the scalar realised return, $\hat y$ is the scalar prediction, and $e = y - \hat y$ is the residual. Where a batch-normalisation factor appears, the batch-level mean is written $\mathbb{E}_{\mathcal{B}}[\cdot]$ and the small regulariser $\epsilon = 10^{-8}$ is included to avoid division by zero.

## A.1 Regression losses

### A.1.1 MSE

$$
L_{\mathrm{MSE}}(y, \hat y) = (y - \hat y)^2,
\qquad
\frac{\partial L_{\mathrm{MSE}}}{\partial \hat y} = -2\,(y - \hat y).
$$

Batch reduction: arithmetic mean.

### A.1.2 MedSE

$$
L_{\mathrm{MedSE}}(y, \hat y) = (y - \hat y)^2,
\qquad
\text{batch reduction} = \mathrm{median}.
$$

Because the median is non-decomposable, the gradient at each observation is zero at every observation *except* those that determine the batch median. In PyTorch the autograd routes gradient only through the median-ranked observation; this is why MedSE training is inherently noisier and why `reduction="median"` is declared explicitly in `medse_loss`.

### A.1.3 Huber (as magnitude backbone)

Used inside every hybrid loss, with $\delta = 0.01$ hard-coded in `_huber_term`:

$$
H_\delta(e) =
\begin{cases}
\tfrac12 e^2 & |e| \le \delta, \\
\delta\, (|e| - \tfrac12 \delta) & |e| > \delta,
\end{cases}
\qquad
\frac{\partial H_\delta}{\partial \hat y}
=
\begin{cases}
-e & |e| \le \delta, \\
-\delta \cdot \operatorname{sgn}(e) & |e| > \delta.
\end{cases}
$$

The gradient is continuous in $e$ (the one-sided derivatives agree at both boundaries $|e| = \delta$) and bounded by $\delta$ in absolute value.

## A.2 Directional losses

### A.2.1 MADL (differentiable)

With $a = 25$ fixed:

$$
L_{\mathrm{MADL}}(y, \hat y) = -\tanh(a \cdot y \cdot \hat y) \cdot |y|,
$$
$$
\frac{\partial L_{\mathrm{MADL}}}{\partial \hat y} = -a \cdot y \cdot |y| \cdot \operatorname{sech}^2(a \cdot y \cdot \hat y).
$$

The gradient peaks near $a \cdot y \cdot \hat y = 0$ and decays exponentially as $|a \cdot y \cdot \hat y|$ grows. For realistically small $|y|$ (monthly returns typically a few per cent), the factor $y \cdot |y| = y^2 \cdot \operatorname{sgn}(y)$ is of order $10^{-4}$, which is the "weak-gradient-near-zero" behaviour described in Chapter 2 §2.4.

### A.2.2 GMADL

With $a = 100$ and $b = 2$ fixed:

$$
L_{\mathrm{GMADL}}(y, \hat y) = -\big[\sigma(a \cdot y \cdot \hat y) - \tfrac12\big] \cdot |y|^b,
$$
$$
\frac{\partial L_{\mathrm{GMADL}}}{\partial \hat y}
= -a \cdot y \cdot |y|^b \cdot \sigma(a \cdot y \cdot \hat y) \cdot \big[1 - \sigma(a \cdot y \cdot \hat y)\big].
$$

The $|y|^b = y^2$ factor provides stronger magnitude weighting than MADL, but the sigmoid derivative $\sigma(1-\sigma)$ saturates at $|a y \hat y| \gg 1$. Together these explain why GMADL produces larger absolute prediction values (to push $|a y \hat y|$ into the saturation region) while preserving directional ranking — and why average R² values diverge under this loss (Chapter 5 §5.2).

### A.2.3 IMADL (rebalanced)

With $a = 100$, $b = 2$, $\lambda_{\mathrm{dir}} = \lambda_{\mathrm{mag}} = 1$ fixed:

$$
D(y, \hat y) = \big[1 - \sigma(a\, y\, \hat y)\big] \cdot \frac{|y|^b}{\mathbb{E}_{\mathcal{B}}[|y|^b] + \epsilon},
$$
$$
L_{\mathrm{IMADL}}(y, \hat y) = \lambda_{\mathrm{dir}} \cdot D(y, \hat y) + \lambda_{\mathrm{mag}} \cdot (y - \hat y)^2,
$$
$$
\frac{\partial L_{\mathrm{IMADL}}}{\partial \hat y}
= -\lambda_{\mathrm{dir}} \cdot a\, y \cdot \sigma(a\, y\, \hat y) \cdot \big[1 - \sigma(a\, y\, \hat y)\big] \cdot \frac{|y|^b}{\mathbb{E}_{\mathcal{B}}[|y|^b] + \epsilon}
- 2\,\lambda_{\mathrm{mag}}\,(y - \hat y).
$$

The batch-normalisation factor $\mathbb{E}_{\mathcal{B}}[|y|^b]$ decouples the scale of the directional gradient from the particular batch composition; it is treated as a constant during the backward pass through that observation (for gradients with respect to $\hat y$, the normalisation denominator is constant, matching the design intent).

## A.3 Hybrid losses

### A.3.1 Additive (A-series)

With Huber $\delta = 0.01$ fixed and $(\lambda_{\mathrm{dir}}, \lambda_{\mathrm{hub}})$ varied per variant (see Chapter 3 §3.3.3):

$$
L_{\mathrm{add}}(y, \hat y) = \lambda_{\mathrm{dir}} \cdot D(y, \hat y) + \lambda_{\mathrm{hub}} \cdot H_\delta(y - \hat y),
$$
$$
\frac{\partial L_{\mathrm{add}}}{\partial \hat y}
= \lambda_{\mathrm{dir}} \cdot \frac{\partial D}{\partial \hat y}
+ \lambda_{\mathrm{hub}} \cdot \frac{\partial H_\delta}{\partial \hat y}.
$$

Both components contribute additively to the gradient; the relative weighting is $(\lambda_{\mathrm{dir}}, \lambda_{\mathrm{hub}})$. The A-variants scan this 2-dimensional hyperparameter.

### A.3.2 Multiplicative (M-series)

$$
L_{\mathrm{mul}}(y, \hat y) = \big(1 + \lambda_{\mathrm{dir}} \cdot D(y, \hat y)\big) \cdot H_\delta(y - \hat y),
$$
$$
\frac{\partial L_{\mathrm{mul}}}{\partial \hat y}
= \lambda_{\mathrm{dir}} \cdot \frac{\partial D}{\partial \hat y} \cdot H_\delta(y - \hat y)
+ \big(1 + \lambda_{\mathrm{dir}} \cdot D(y, \hat y)\big) \cdot \frac{\partial H_\delta}{\partial \hat y}.
$$

The multiplicative form couples the two components: the Huber gradient is *gated* by the directional factor $1 + \lambda_{\mathrm{dir}} D$. When the direction is predicted correctly, $D \to 0$ and the gradient reduces to $\partial H_\delta / \partial \hat y$. When the direction is predicted wrongly, the sigmoid penalty $\sigma(ayŷ)$ approaches 0 so $[1-\sigma(\cdot)]$ approaches 1, and the Huber gradient is amplified by a factor of $1 + \lambda_{\mathrm{dir}} \cdot D$, where $D$ is proportional to the batch-normalised magnitude weight $|y|^b / (\mathbb{E}_{\mathcal{B}}[|y|^b]+\epsilon)$.

### A.3.3 M2-robust γ family

Conceptually, the M2-robust loss replaces the fixed-threshold Huber backbone with a γ-controlled saturating magnitude term $H^{\gamma}(e)$ and combines it with the same directional-gating structure:

$$
L_{\mathrm{M2\text{-}robust},\gamma}(y, \hat y) = \big(1 + \lambda_{\mathrm{dir}} \cdot D(y, \hat y)\big) \cdot H^{\gamma}(y - \hat y).
$$

The robustness schedule $H^{\gamma}(e)$ is a smooth monotone function that approaches $\tfrac12 e^2$ for small $|e|$ and saturates above a threshold controlled by $\gamma$. Small $\gamma$ flattens the loss surface and prioritises the directional component; large $\gamma$ recovers a nearly unbounded quadratic behaviour. The γ refinement of Chapter 5 §5.4 scans $\gamma \in \{0.03, 0.05, 0.07, 0.10, 0.15\}$. The exact closed form used in the Phase 2.2 runner lives on the `phase2.2-fix` branch together with the runner scripts; the empirically selected setting within the reported 24-month, 3-seed protocol is $\gamma = 0.07$ (see Chapter 5 §5.4 and Table 5.3).

## A.4 Notes on practical implementation

Two implementation points are worth repeating for completeness, both documented in `Model_Train/losses.py`:

1. **Reduction.** Every loss function uses `reduction="mean"` by default, except MedSE which uses `reduction="median"`. Passing `reduction="none"` returns the pointwise loss vector, which the tests in `tests/` consume.
2. **Numerical guards.** IMADL and the hybrid variants use the batch-mean normalisation $\mathbb{E}_{\mathcal{B}}[|y|^b] + \epsilon$ with $\epsilon = 10^{-8}$; the Huber term has no special numerical guard because it is smooth and bounded.

A LaTeX conversion pass can lift these formulas into a standard mathematical appendix. The `\mathrm{...}` wrappers used here are compatible with both Obsidian MathJax and LaTeX `amsmath`.
