# Chapter 2: Literature Review

The design space explored in this report sits at the intersection of three strands of literature: (i) machine learning for cross-sectional stock-return prediction, (ii) robust regression under heavy-tailed outcomes, and (iii) trading-aware loss functions that target portfolio rather than point-prediction objectives. This chapter reviews each strand in turn, then positions the report's contribution: a systematic, controlled comparison of loss functions, culminating in the M2-robust hybrid family.

## 2.1 Machine learning for cross-sectional stock-return prediction

Empirical asset pricing has historically relied on linear factor models in which the expected cross-sectional return is assumed to be a linear function of a small set of pre-specified characteristics. The three-factor model of Fama and French (1993) — market, size, and book-to-market — remains the workhorse benchmark, later extended to include momentum, profitability, and investment factors [2]. These models succeed as parsimonious explanations but impose a linear structure on a relationship that need not be linear.

Modern machine-learning approaches relax this linearity. Gu, Kelly, and Xiu (2020) construct a comprehensive horse race across regression trees, gradient boosting, random forests, and deep neural networks on a panel of 94 stock characteristics, and find that deeper non-linear models systematically outperform linear benchmarks out of sample [1]. Their key finding — that model capacity translates into incremental out-of-sample predictive power — has motivated a wave of MLP-, LSTM-, and transformer-based architectures applied to return prediction. Daniel and Moskowitz (2016) complement this architectural work on the feature side, isolating cumulative-return and turnover-based predictors that capture medium-term momentum while avoiding short-term reversal [3]. The feature set X1 used throughout this report (Chapter 4) follows their construction.

The pattern across this literature is informative: essentially every study iterates on architecture and feature construction while holding the training objective fixed, typically at MSE. Because the downstream decision — a long-short portfolio — depends on the cross-sectional *ranking* of predictions rather than their calibrated values, and because return distributions deviate substantially from the Gaussian noise assumption implicit in MSE, there is a clear opening to ask whether the fixed choice of MSE is the right one. That question is the starting point of this report.

## 2.2 Heavy-tailed returns and the case for robust losses

Equity returns are not Gaussian. Monthly cross-sections contain sporadic observations with returns in excess of $\pm 100\%$ (in the training-era panel used in Chapter 4, the realised monthly return distribution has a standard deviation of roughly $19.5\%$ but a maximum of $+2400\%$ for a single name-month). Four properties of this distribution challenge standard loss functions:

1. **Heavy tails.** Extreme returns occur far more often than a Gaussian model predicts, so the variance of a sample is dominated by a small number of tail observations.
2. **Skewness.** Return distributions are typically negatively skewed: large drawdowns are more common than positive returns of equivalent magnitude.
3. **Time-varying volatility.** Volatility clusters, violating the homoskedasticity assumption that underpins MSE's efficiency argument.
4. **Idiosyncratic shocks.** Company-specific events (earnings surprises, management changes, regulatory shocks) produce large returns that are unrelated to predictable factors and act as noise from the perspective of a factor-style model.

Under these conditions, a loss function that is quadratic in the residual gives disproportionate weight to a small number of outliers. The trained model fits the noise rather than the predictable component. This is the classical motivation for robust regression — a line of work that goes back at least to Huber (1964) and has been applied extensively to finance whenever heavy tails matter.

Huber loss offers a principled middle ground between quadratic and linear penalisation,
$$
L_{\delta}(y, \hat y) =
\begin{cases}
\tfrac{1}{2}(y - \hat y)^2 & \text{if } |y - \hat y| \le \delta, \\
\delta \, |y - \hat y| - \tfrac{1}{2}\delta^2 & \text{otherwise.}
\end{cases}
$$

For residuals below the threshold $\delta$ the loss behaves like MSE, preserving efficiency under near-Gaussian conditions; for residuals above $\delta$ it behaves like MAE, limiting the influence of any single observation. The resulting M-estimator retains desirable convexity and differentiability properties while being robust to a non-trivial fraction of contamination.

More aggressive robustness is provided by median-based losses. Median Squared Error (MedSE), $\mathrm{MedSE} = \mathrm{median}_i[(y_i - \hat y_i)^2]$, depends only on the central ordered residual and is therefore unaffected by contamination of up to half the sample. The price is computational: MedSE does not decompose across observations and has subgradient rather than gradient information at the median rank. Both properties matter for neural-network training.

These classical robustness ideas underpin the baseline comparison of Chapter 5, where MSE, MedSE, and the absolute-loss family (MADL/GMADL/IMADL — discussed in §2.4) are evaluated under identical protocol. The comparison provides the empirical grounding for the hybrid designs introduced in §2.5.

## 2.3 Traditional regression losses under portfolio objectives

The loss functions of §2.2 are regression losses: they target the conditional expectation $E[y_i \mid x_i]$ (MSE) or the conditional median $\text{med}[y_i \mid x_i]$ (MedSE), and their theoretical optimality is stated in terms of point-prediction error.

A cross-sectional long-short portfolio, however, depends only on the *ranks* of the predictions within each cross-section. Given the per-period prediction vector $\{\hat y_i\}_i$ for month $t$, the portfolio takes the top decile as long positions and the bottom decile as short positions. Two implications follow.

First, a prediction that is badly miscalibrated in magnitude but correctly ranked is indistinguishable, at the portfolio level, from a calibrated prediction with the same ranking. Conversely, a prediction that is correctly calibrated on average but poorly ranked in the tails produces an inferior portfolio. Optimising MSE therefore optimises a proxy of the quantity that matters for the portfolio.

Second, the R² of a model trained with a non-calibrated objective can be arbitrarily large in magnitude (as shown in Chapter 5, several variants reach average R² values below $-10^9$) while portfolio performance remains stable. R² is a scale-sensitive diagnostic and, in this setting, stops being a primary performance metric. This is a central empirical observation of the report and shapes how claims are made in Chapter 5.

The gap between point-prediction loss and portfolio objective has two standard responses in the literature. One response is to optimise a ranking loss directly (LambdaRank, ListNet, NDCG-based objectives). These methods target the ranking metric that matters for the portfolio, but they typically require pairwise or listwise comparisons within each cross-section, which is expensive when the cross-section is large. A second response is to construct *trading-aware* losses that explicitly reward correct directional predictions weighted by realised return magnitude. That second response is the family examined in §2.4, and it motivates the hybrid designs in §2.5.

## 2.4 Directional and trading-aware losses: MADL, GMADL, IMADL

The Mean Absolute Directional Loss (MADL) of Michańków, Ślepaczuk, and Bielak (2024) replaces point-prediction error with an explicit directional reward weighted by realised return magnitude [7]:
$$
L_{\mathrm{MADL}}(y, \hat y) = -\tanh\!\big(a \cdot y \cdot \hat y\big) \cdot |y|,
$$
with a typical scaling constant $a = 25$. The sign of $y \hat y$ encodes whether the prediction agrees with the realised return; the $\tanh$ wrapper provides a smooth, bounded translation of that sign into a training signal; and the $|y|$ factor weights the loss by how large the realised move was, so that correctly predicting a large return contributes more to training than correctly predicting a small one.

The same authors extend MADL into Generalised MADL (GMADL),
$$
L_{\mathrm{GMADL}}(y, \hat y) = -\big[\sigma(a \cdot y \cdot \hat y) - \tfrac12\big] \cdot |y|^b,
$$
with $a = 100$ and $b = 2$. The sigmoid replacement provides smoother gradients than $\tanh$ at large arguments, and the $|y|^b$ term magnifies the weighting of large returns. GMADL is the parent of the adaptive and hybrid families developed later in Phase 2 of this report.

An Inverse MADL (IMADL) variant has also been explored within the project codebase. IMADL is conceptually similar to MADL but re-parameterises the directional term so that the reward curve is steeper around the zero-prediction region, at the cost of making the loss less symmetric between correct and incorrect directions. The exact IMADL formulas that appear in the Phase 2 grouped summaries (IMADL-m2 $\alpha$ sweep and IMADL-GMADL $\beta$ sweep) are documented in Chapter 3 and in the loss implementation notes `doc/phase2.5/07_loss_implementation_details.md`.

The directional-loss family has three theoretical attractions that directly address the point-prediction–portfolio mismatch of §2.3:

1. **Directional alignment.** A model trained on MADL/GMADL is explicitly optimising for $\operatorname{sign}(\hat y) = \operatorname{sign}(y)$, which is the quantity that determines whether a name enters the long or short bucket.
2. **Magnitude weighting.** By multiplying the directional term by $|y|$ or $|y|^b$, the loss spends training signal on the moves that matter economically.
3. **Bounded loss.** Every term is bounded (the $\tanh$ and sigmoid factors saturate), which limits the influence of any single observation and acts as an implicit robustness mechanism.

Several limitations of the pure directional family have also been documented:

<!-- FIGURE PLACEHOLDER: Fig 2.1
  TYPE: four-panel didactic loss-shape comparison
    Each panel plots a 1D loss profile L(y, ŷ) as a function of (ŷ - y) in [-0.5, 0.5]
    with the realised return y held at a representative value (e.g. y = 0.05).
    Panel 1: MSE (quadratic) and Huber with delta=0.01 overlaid.
    Panel 2: MedSE (flat-in-median) illustrative shape.
    Panel 3: MADL (tanh-based) and GMADL (sigmoid-based with |y|^2) overlaid;
             mark the "weak-gradient-near-zero" region with a shaded band around ŷ=0.
    Panel 4: Hybrid multiplicative loss L_mul(y, ŷ) with lambda_dir = 2 (M1) overlaid on the Huber
             backbone; emphasise the gating effect when the directional penalty grows.
  DATA SOURCES:
    None (synthetic evaluation of the closed-form loss expressions in Chapter 3 §3.3).
    Use numpy to evaluate over a dense ŷ grid at fixed y.
  CAPTION:
    Figure 2.1 — Conceptual shape of MSE, Huber, MedSE, MADL, GMADL, and the multiplicative
    hybrid loss at a fixed realised return y = 0.05. Illustrative only; no training data
    is used. Formulas match those in Chapter 3 §3.3.
-->
**Figure 2.1 — Conceptual loss-function shape comparison (placeholder; see comment for chart spec and formulas).**

1. **Symmetric asymmetry.** The reward for a correct large-magnitude prediction equals the penalty for an incorrect large-magnitude prediction. The loss encodes no preference for avoiding losses over capturing gains, which is a mismatch with the risk-averse nature of most trading applications.
2. **Weak gradients near $\hat y \approx 0$.** When the prediction is close to zero, $y \hat y$ is close to zero, and the sigmoid or $\tanh$ derivative is close to its peak but the overall loss derivative with respect to $\hat y$ remains proportional to $|y|$ (or $|y|^b$). For small $|y|$ this can produce uninformative training signal, slowing convergence.
3. **No notion of prediction precision beyond sign.** A prediction of $+0.01$ and a prediction of $+1.00$ receive essentially the same directional reward if the realised return is positive. In a ranking context this is a feature rather than a bug — the model focuses on the ordering — but in a calibrated forecasting context it would be a liability.
4. **Scale sensitivity in the neural-network setting.** Empirically, MADL/GMADL trained MLPs produce predictions whose scale drifts far from the realised-return scale (Chapter 5 shows average R² values divergent by several orders of magnitude). The directional reward is unaffected, but any diagnostic that depends on calibration is no longer usable.

The limitations above are what motivates the *hybrid* designs evaluated in this report. Rather than choosing between a regression loss (MSE/MedSE/Huber) and a directional loss (MADL/GMADL/IMADL), a hybrid loss combines the two — either additively or multiplicatively — so that the gradient signal contains both a calibration component and a directional component.

## 2.5 Hybrid loss design and the M2-robust family

The hybrid losses studied in this report belong to two families. The additive hybrid family takes the form
$$
L_{\mathrm{add}, \alpha}(y, \hat y) = L_{\mathrm{base}}(y, \hat y) + \alpha \cdot L_{\mathrm{direction}}(y, \hat y),
$$
where $L_{\mathrm{base}}$ is a regression loss (MSE in the A-series of Chapter 5) and $L_{\mathrm{direction}}$ is a directional loss derived from MADL/GMADL. The five variants A1–A5 in the empirical tables scan $\alpha$ across five settings; A3 produces the best single-seed Sharpe at seed 42, while the extremes A1 and A5 either under-weight or over-weight the directional term.

The multiplicative hybrid family takes the form
$$
L_{\mathrm{mul}}(y, \hat y) = L_{\mathrm{base}}(y, \hat y) \cdot f_{\mathrm{direction}}(y, \hat y),
$$
where $f_{\mathrm{direction}}$ is a positive modifier that up-weights residuals whose direction is mispredicted and down-weights residuals whose direction is correctly predicted. Intuitively, the multiplicative form treats the regression loss as the "backbone" and uses the directional term as a gating factor. The Phase 1.5 variants M1–M4 are specific choices of $f_{\mathrm{direction}}$; M1 emerges as the single-seed peak in Chapter 5.

The M2-robust family, introduced in Phase 2 of the project, extends the multiplicative hybrid design with an explicit robustness parameter $\gamma$ that controls how much the magnitude of large residuals contributes to the loss. Conceptually, the modifier has the form
$$
f^{\mathrm{robust}}_{\mathrm{direction}, \gamma}(y, \hat y)
= g_{\mathrm{direction}}(y, \hat y) \cdot h_{\gamma}(|y - \hat y|),
$$
where $h_{\gamma}$ is a saturating function that approaches 1 for small residuals and levels off above a threshold controlled by $\gamma$. Small $\gamma$ flattens the loss surface (the robust component dominates); large $\gamma$ approaches the non-robust multiplicative form. The empirical sweep in Chapter 5 shows that performance is a non-monotone function of $\gamma$: Sharpe is maximised around $\gamma \approx 0.10$ but seed-stability peaks around $\gamma \approx 0.07$. The design choice of selecting $\gamma = 0.07$ rather than $\gamma = 0.10$ is therefore a deliberate trade of a small amount of mean Sharpe for a substantial reduction in seed sensitivity.

The IMADL-m2 $\alpha$ family, also evaluated in Phase 2, follows the same multiplicative-hybrid skeleton but replaces the GMADL-style directional factor with the IMADL factor and parameterises its weight through $\alpha$ rather than $\gamma$. The $\alpha$ sweep peaks at $\alpha = 0.6$; the comparable sweeps for the IMADL-GMADL $\beta$ composition and the adaptive-$\lambda$ schedule do not produce stable positive Sharpes. Chapter 5 interprets this pattern as evidence that the multiplicative-hybrid skeleton is the productive region of the loss space, and that within that skeleton the M2-robust $\gamma$ parameterisation is better behaved than the other parameterisations that were tested.

## 2.6 Validation, overfitting, and multiple testing

The standard diagnostic concerns for empirical asset-pricing studies apply here with extra force, because the research compares many loss functions on a single panel. Lopez de Prado (2014) and Bailey et al. (2014) document that the apparent performance of a backtested strategy is upward-biased whenever the researcher observes the performance of many candidate strategies on the same data and reports only the best [4, 5]. Harvey, Liu, and Zhu (2016) extend this multiple-testing argument to the broader empirical factor literature and recommend higher $t$-statistic thresholds to account for the universe of tested factors [6].

This report does not introduce new factors, but the loss-function comparison creates an analogous multiple-comparison problem. Seven baseline losses, nine Phase 1.5 variants, five Phase 2.2 $\gamma$ refinements, and additional $\alpha$/$\beta$/$\lambda$ sweeps are evaluated on the same 24-month test window. Three mitigations are applied:

1. **Reporting every tested configuration.** Chapter 5 reports the full baseline table, the full A/M sweep, and the full multi-seed Phase 2 summaries rather than only the winning row. The reader can verify the selection rule and is not left guessing which configurations were omitted.
2. **Multi-seed evaluation with explicit stability reporting.** The final recommendation is not drawn from single-seed evidence. The $\gamma$ refinement and the broader Phase 2 sweep are evaluated across three seeds per row, and stability is reported as the coefficient of variation $\mathrm{CV} = \sigma_S / |\mu_S|$. A single-seed peak does not qualify as a headline claim.
3. **Static out-of-sample protocol with no in-window tuning.** The model architecture is fixed (MLP[64,32,16]+ReLU+dropout 0.2) across every experiment, chosen via grid search on pre-test data and then frozen. Hyperparameters such as batch size and epoch count are held fixed across losses. No parameter is tuned on the 1995–1996 test window.

These mitigations do not eliminate the multiple-testing concern. With three seeds, the standard error on mean Sharpe remains wide, and the absolute CV values reported in Chapter 5 should be read as order-of-magnitude indicators rather than precise estimates. A larger seed depth and a second, independent test window would strengthen the robustness argument; both are listed in the future-work discussion of Chapter 6.

The static-window design also implies a specific position on the internal-versus-external validity trade-off. Training once on 1990–1994 and evaluating on 1995–1996 gives a clean causal attribution of performance differences to the loss function (internal validity) at the cost of limited regime coverage (external validity). A rolling-window counterpart would reverse that trade-off. The project originally planned a rolling-window extension but paused it when the static-window results began to reveal the seed-sensitivity patterns explored in Phase 2; the argument of the report is that within-phase controlled comparisons are the evidence the reader can trust most, and that generalisation to other regimes is future work.

## 2.7 Research gap and positioning

The literature reviewed above points to a clear gap. On the architecture side, the machine-learning-for-returns literature has converged on deep non-linear models as an improvement over linear factor models; the loss function, however, has remained overwhelmingly MSE or MAE. On the loss-function side, robust regression has a long history but has rarely been evaluated *against* trading-aware losses under controlled conditions; when trading-aware losses have been proposed, they have typically been demonstrated on a single dataset without systematic comparison to a comprehensive set of alternatives or to robust hybrids.

This report addresses the gap with a controlled, multi-phase comparison that fixes data, features, model architecture, portfolio construction, and evaluation protocol across every run, and varies only the loss function:

- **Phase 1 / baseline.** Seven losses covering the three families of §2.2–§2.4 (regression, absolute-loss, hybrid-multiplicative baselines), evaluated at a single seed on a static 24-month window.
- **Phase 1.5.** Nine parameterised hybrid variants (five additive A-series, four multiplicative M-series), evaluated at the same seed and window, to probe the design space of hybrid losses.
- **Phase 2.** Five γ settings of the M2-robust family plus an integrated sweep over $\alpha$, $\beta$, and $\lambda$ parameterisations, evaluated across three seeds each, to separate single-seed peaks from multi-seed expected performance.
- **Phase 2.2-fix1.** A normalisation probe on the three strongest candidates to test whether loss-component scale imbalance drives the observed ranking.
- **Phase 2.1b / Phase 2.5.** Alignment diagnostics that define the boundary within which cross-phase claims are legitimate.

Three aspects of the research positioning are worth stating explicitly.

First, the report *does not* claim to improve on the state of the art of financial machine learning. It fixes an off-the-shelf MLP architecture and a standard momentum feature set and does not re-tune them. The contribution is to the loss-function side of the design space, not to architecture or feature engineering.

Second, the report *does* claim that, within the empirical protocol studied here, a multiplicative-hybrid loss with a tuned robust component (`m2_robust_gamma07`, $\gamma = 0.07$) dominates traditional regression losses and pure directional losses on long-short Sharpe and on cross-seed stability. The claim is scoped to this protocol, this market, this frequency, and this test window. Chapter 6 revisits the scope limits in detail.

Third, the report *avoids* claims that go beyond its evidence. It does not assert that normalisation is universally unhelpful, that Phase 2 exactly replicates Phase 1.5, or that the chosen γ is optimal across regimes. The writing rules of the evidence contract (`SCHEMA.md`) make this scoping explicit; the empirical chapters in this report follow it literally.
