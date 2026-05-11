# Chapter 2: Literature Review

The design space explored in this report sits at the intersection of three strands of literature: (i) machine learning for cross-sectional stock-return prediction, (ii) robust regression under heavy-tailed outcomes, and (iii) trading-aware loss functions that target portfolio rather than point-prediction objectives. This chapter reviews each strand in turn and uses the review to motivate a controlled comparison of regression, robust, directional, and hybrid losses. The project-specific hybrid formulations are defined in Chapter 3 and evaluated in Chapter 5.

## 2.1 Machine learning for cross-sectional stock-return prediction

Empirical asset pricing has historically relied on linear factor models in which the expected cross-sectional return is assumed to be a linear function of a small set of pre-specified characteristics. The three-factor model of Fama and French (1993) — market, size, and book-to-market — remains the workhorse benchmark, later extended to include momentum, profitability, and investment factors [2]. These models succeed as parsimonious explanations but impose a linear structure on a relationship that need not be linear.

Modern machine-learning approaches relax this linearity. Gu, Kelly, and Xiu (2020) construct a comprehensive comparative study across regression trees, gradient boosting, random forests, and deep neural networks on a panel of 94 stock characteristics, and find that deeper non-linear models systematically outperform linear benchmarks out of sample [1]. Their key finding — that model capacity translates into incremental out-of-sample predictive power — has motivated a wave of MLP-, LSTM-, and transformer-based architectures applied to return prediction. Daniel and Moskowitz (2016) complement this architectural work on the feature side, isolating cumulative-return and turnover-based predictors that capture medium-term momentum while avoiding short-term reversal [3]. The feature set X1 used throughout this report (Chapter 4) follows their construction.

The pattern across this literature is informative: essentially every study iterates on architecture and feature construction while holding the training objective fixed, typically at MSE. Because the downstream decision — a long-short portfolio — depends on the cross-sectional *ranking* of predictions rather than their calibrated values, and because return distributions deviate substantially from the Gaussian noise assumption implicit in MSE, there is a clear opening to ask whether the fixed choice of MSE is the right one. That question is the starting point of this report.

## 2.2 Heavy-tailed returns and the case for robust losses

Equity returns are not Gaussian. Monthly cross-sections contain sporadic observations with returns in excess of $\pm 100\%$ (in the training-era panel used in Chapter 4, the realised monthly return distribution has a standard deviation of roughly $19.5\%$ but a maximum of $+2400\%$ for a single name-month). Four properties of this distribution challenge standard loss functions:

1. **Heavy tails.** Extreme returns occur far more often than a Gaussian model predicts, so the variance of a sample is dominated by a small number of tail observations.
2. **Skewness.** Return distributions are often negatively skewed at the individual-stock level, though the training panel's extreme positive tail ($+2400\%$) shows that right-tail outliers also occur. The asymmetry violates the symmetric-noise assumption of MSE.
3. **Time-varying volatility.** Volatility clusters, violating the homoskedasticity assumption that underpins MSE's efficiency argument.
4. **Idiosyncratic shocks.** Company-specific events (earnings surprises, management changes, regulatory shocks) produce large returns that are unrelated to predictable factors and act as noise from the perspective of a factor-style model.

Under these conditions, a loss function that is quadratic in the residual gives disproportionate weight to a small number of outliers. The trained model fits the noise rather than the predictable component. This is the classical motivation for robust regression — a line of work that goes back at least to Huber (1964) [A1] and has been applied extensively to finance whenever heavy tails matter.

Huber loss offers a principled middle ground between quadratic and linear penalisation: for residuals below a threshold $\delta$ the loss behaves like MSE, preserving efficiency under near-Gaussian conditions; for residuals above $\delta$ it behaves like MAE, limiting the influence of any single observation. The resulting M-estimator retains desirable convexity and differentiability properties while being robust to a non-trivial fraction of contamination (see Appendix A §A.1.3 for the closed-form expression).

More aggressive robustness is provided by median-based losses. Median Squared Error (MedSE), $\mathrm{MedSE} = \mathrm{median}_i[(y_i - \hat y_i)^2]$, depends only on the central ordered residual and is therefore unaffected by contamination of up to half the sample. MedSE is implemented in this project as a robust baseline (see Chapter 3 §3.3 and Appendix A); it follows the same median-of-squared-residuals principle used in robust statistics but is not a standard named loss in the deep-learning literature. The price is computational: MedSE does not decompose across observations and has subgradient rather than gradient information at the median rank. Both properties matter for neural-network training.

These classical robustness ideas underpin the baseline comparison of Chapter 5, where MSE, MedSE, and the absolute-loss family (MADL/GMADL/IMADL — discussed in §2.4) are evaluated under the same Chapter 5 baseline protocol. The comparison provides the empirical grounding for the hybrid designs defined in Chapter 3.

## 2.3 Traditional regression losses under portfolio objectives

The loss functions of §2.2 are regression losses: they target the conditional expectation $E[y_i \mid x_i]$ (MSE) or the conditional median $\text{med}[y_i \mid x_i]$ (MedSE), and their theoretical optimality is stated in terms of point-prediction error.

A cross-sectional long-short portfolio, however, depends only on the *ranks* of the predictions within each cross-section. Given the per-period prediction vector $\{\hat y_i\}_i$ for month $t$, the portfolio takes the top decile as long positions and the bottom decile as short positions. Two implications follow.

First, a prediction that is badly miscalibrated in magnitude but correctly ranked is indistinguishable, at the portfolio level, from a calibrated prediction with the same ranking. Conversely, a prediction that is correctly calibrated on average but poorly ranked in the tails produces an inferior portfolio. Optimising MSE therefore optimises a proxy of the quantity that matters for the portfolio.

Second, the R² of a model trained with a non-calibrated objective can be arbitrarily large in magnitude (as shown in Chapter 5, several variants reach average R² values below $-10^9$) while portfolio performance remains stable. R² is a scale-sensitive diagnostic and, in this setting, stops being a primary performance metric. This is a central empirical observation of the report and shapes how claims are made in Chapter 5.

The gap between point-prediction loss and portfolio objective has two standard responses in the literature. One response is to optimise a ranking loss directly (LambdaRank, ListNet, NDCG-based objectives). These methods target the ranking metric that matters for the portfolio, but they typically require pairwise or listwise comparisons within each cross-section, which is expensive when the cross-section is large. A second response is to construct *trading-aware* losses that explicitly reward correct directional predictions weighted by realised return magnitude. That second response is the family examined in §2.4, and it motivates the hybrid designs defined in Chapter 3.

## 2.4 Directional and trading-aware losses: MADL, GMADL, IMADL

The Mean Absolute Directional Loss (MADL) of Michańków, Ślepaczuk, and Bielak (2024) replaces point-prediction error with an explicit directional reward weighted by realised return magnitude [7]. The sign of $y \hat y$ encodes whether the prediction agrees with the realised return; a $\tanh$ wrapper provides a smooth, bounded translation of that sign into a training signal; and a $|y|$ factor weights the loss by how large the realised move was, so that correctly predicting a large return contributes more to training than correctly predicting a small one.

The same authors extend MADL into Generalised MADL (GMADL) [8], which replaces $\tanh$ with a sigmoid (providing a different saturation profile centred at $0.5$ rather than $0$) and uses $|y|^b$ to magnify the weighting of large returns. GMADL provides one of the directional components later reused in the project's hybrid losses.

An Inverse MADL (IMADL) variant has also been developed within the project codebase as a project-specific extension rather than a published loss. IMADL is conceptually similar to MADL but re-parameterises the directional term so that the reward curve is steeper around the zero-prediction region, at the cost of making the loss less symmetric between correct and incorrect directions. The exact IMADL formulas are documented in Chapter 3 and Appendix A.

The directional-loss family has three theoretical attractions that directly address the point-prediction–portfolio mismatch of §2.3:

1. **Directional alignment.** A model trained on MADL/GMADL is explicitly optimising for $\operatorname{sign}(\hat y) = \operatorname{sign}(y)$, which is the quantity that determines whether a name enters the long or short bucket.
2. **Magnitude weighting.** By multiplying the directional term by $|y|$ or $|y|^b$, the loss spends training signal on the moves that matter economically.
3. **Bounded directional activation.** The directional factors ($\tanh$ and sigmoid) saturate, which limits the influence of the alignment term on any single observation. However, the overall loss is weighted by $|y|$ or $|y|^b$, so the *realised-return magnitude* component is unbounded. The directional activation provides implicit robustness to prediction outliers but does not fully protect against heavy-tailed realised returns.

Several limitations of the pure directional family have also been documented:


**Figure 2.1 — Conceptual loss-function shape comparison.**

![Figure 2.1: conceptual loss-function shape comparison](figures/fig2_1_loss_shapes.png)

Generated by `paper/figures/plot_loss_shapes.py`. Illustrative shapes (no training data) of MSE vs Huber ($\delta = 0.01$), per-observation MedSE, MADL vs GMADL with the weak-gradient band around $\hat y = 0$ highlighted, and the hybrid_mul_m1 loss at $\lambda_{\mathrm{dir}} = 2$ compared against its Huber backbone. All formulas match Chapter 3 §3.3; the realised return is fixed at $y = 0.05$.

1. **Symmetric reward and penalty.** The reward for a correct large-magnitude prediction equals the penalty for an incorrect large-magnitude prediction. The loss encodes no preference for avoiding losses over capturing gains, which is a mismatch with the risk-averse nature of most trading applications.
2. **Weak gradients near $\hat y \approx 0$.** When the prediction is close to zero, $y \hat y$ is close to zero, and the sigmoid or $\tanh$ derivative is close to its peak but the overall loss derivative with respect to $\hat y$ remains proportional to $|y|$ (or $|y|^b$). For small $|y|$ this can produce uninformative training signal, slowing convergence.
3. **No notion of prediction precision beyond sign.** A prediction of $+0.01$ and a prediction of $+1.00$ receive essentially the same directional reward if the realised return is positive. In a ranking context this is a feature rather than a bug — the model focuses on the ordering — but in a calibrated forecasting context it would be a liability.
4. **Scale sensitivity in the neural-network setting.** Empirically, MADL/GMADL trained MLPs produce predictions whose scale drifts far from the realised-return scale (Chapter 5 shows average R² values divergent by several orders of magnitude). The directional reward is unaffected, but any diagnostic that depends on calibration is no longer usable.

These limitations motivate the hybrid designs evaluated later in the report. Rather than treating regression accuracy, robustness, and directional alignment as separate objectives, the project tests whether they can be combined in one training loss. Chapter 3 gives the formal definitions of the additive, multiplicative, and M2-robust variants; Chapter 5 evaluates them empirically.

## 2.5 Validation, overfitting, and multiple testing

The standard diagnostic concerns for empirical asset-pricing studies apply here with extra force, because the research compares many loss functions on a single panel. Bailey and López de Prado (2014) and Bailey et al. (2017) document that the apparent performance of a backtested strategy is upward-biased whenever the researcher observes the performance of many candidate strategies on the same data and reports only the best [4, 5]. Harvey, Liu, and Zhu (2016) extend this multiple-testing argument to the broader empirical factor literature and recommend higher $t$-statistic thresholds to account for the universe of tested factors [6].

This report does not introduce new factors, but the loss-function comparison creates an analogous multiple-comparison problem: many loss variants are evaluated on the same 24-month test window. Three mitigations are applied:

1. **Reporting every tested configuration.** Chapter 5 reports the full baseline table, the full A/M sweep, and the full multi-seed summaries rather than only the winning row. The reader can verify the selection rule and is not left guessing which configurations were omitted.
2. **Multi-seed evaluation with explicit stability reporting.** The final recommendation is not drawn from single-seed evidence. The $\gamma$ refinement and the broader multi-seed sweep are evaluated across three seeds per row, and stability is reported as the coefficient of variation $\mathrm{CV} = \sigma_S / |\mu_S|$. A single-seed peak does not qualify as a headline claim.
3. **Static out-of-sample protocol with no in-window tuning.** The model architecture is fixed (MLP[64,32,16]+ReLU+dropout 0.2) across every experiment, chosen via grid search on pre-test data and then frozen. Hyperparameters such as batch size and epoch count are held fixed across losses. No parameter is tuned on the 1995–1996 test window.

These mitigations do not eliminate the multiple-testing concern. With three seeds, the standard error on mean Sharpe remains wide, and the absolute CV values reported in Chapter 5 should be read as order-of-magnitude indicators rather than precise estimates. A larger seed depth and a second, independent test window would strengthen the robustness argument; both are listed in the future-work discussion of Chapter 6.

The static-window design also implies a specific position on the internal-versus-external validity trade-off. Training once on 1990–1994 and evaluating on 1995–1996 gives a clean causal attribution of performance differences to the loss function (internal validity) at the cost of limited regime coverage (external validity). A rolling-window counterpart would reverse that trade-off. The project originally planned a rolling-window extension but paused it when the static-window results began to reveal the seed-sensitivity patterns explored in the multi-seed phases; the argument of the report is that within-phase controlled comparisons are the evidence the reader can trust most, and that generalisation to other regimes is future work.

## 2.6 Literature synthesis

The literature reviewed above points to a clear gap. Return-prediction papers routinely vary architecture and features while keeping the loss function fixed at MSE or MAE. Robust-regression and trading-aware losses have been proposed independently but have rarely been compared against each other — or against hybrid combinations — under controlled conditions with explicit stability reporting.

This report contributes a controlled comparison of regression, robust, directional, and hybrid losses under one fixed protocol (data, features, MLP architecture, training settings, and portfolio construction). Within each comparison table the loss is the only varying factor; cross-phase comparisons serve as design motivation rather than direct causal claims. The contribution is to the loss-function side of the design space, not to architecture or feature engineering.

Chapter 3 describes the implemented protocol and the claim-boundary taxonomy that governs what can and cannot be said across experimental phases. Chapter 5 reports the empirical comparisons and synthesises the final recommendation.
