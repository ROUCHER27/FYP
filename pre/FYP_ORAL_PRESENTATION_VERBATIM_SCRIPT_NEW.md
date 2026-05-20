# FYP Oral Presentation Verbatim Script - New Draft

Target delivery: 15 to 20 minutes for the talk, plus the required 5 to 10 minutes for examiner questions. The main script deliberately spends more time on loss-function design, data, and methodology.

Source basis: actual 19-slide deck, final LaTeX chapters, `ppt_package/tables/`, and the oral-presentation requirement PDF.

---

## Slide 1 - Title

Good morning, everyone. My name is Yirong Yu, student ID 2253235, from BSc Financial Mathematics. My final year project is titled "Multiplicative Directional-Robust Loss for Cross-Sectional Stock-Return Prediction," supervised by Dr. Yi Cao.

The main idea is that stock-return models are usually trained to minimise prediction error, but the portfolio later trades the ranking of predictions. My project asks whether changing the loss function can better align model training with the long-short portfolio objective.

---

## Slide 2 - Background And Three Questions

The starting point is a mismatch. In machine-learning return prediction, MSE is often the default training loss. MSE asks the model to match realised returns in numerical magnitude.

But a long-short portfolio does not use predictions in the same way. It ranks stocks cross-sectionally, buys the top group, and shorts the bottom group. So the downstream decision depends mainly on rank and direction, not on exact calibrated return values.

The data make this mismatch more serious. Monthly stock returns are heavy-tailed. In my training sample, the standard deviation of monthly returns is 19.54 percent, and the maximum monthly return reaches plus 2400 percent. A quadratic loss can over-focus on these extreme observations, even though the portfolio later uses ranking.

This motivates three research questions. First, how does loss choice affect prediction-level and portfolio-level performance? Second, which hybrid loss design gives the best Sharpe-stability trade-off? Third, do the leading candidates remain stable under component-normalisation diagnostics, or are they just scale artefacts?

---

## Slide 3 - Literature Snapshot

The project sits between three research streams.

First, machine learning for stock prediction. Gu, Kelly, and Xiu show that nonlinear machine-learning models can improve return prediction, while Daniel and Moskowitz and Medhat and Schmeling motivate the return and turnover features used in this project.

Second, robust regression. MSE is sensitive to outliers. Huber-style losses and MedSE reduce the effect of heavy-tailed residuals, but they do not directly encode directional correctness.

Third, directional losses. MADL and GMADL reward sign alignment between realised and predicted returns. That is closer to a trading problem, but pure directional losses may lose prediction-scale control.

The gap is that previous work does not compare regression, robust, directional, and hybrid robust-directional losses under the same controlled portfolio protocol. That is what this project does.

---

## Slide 4 - Controlled Pipeline

The methodology is a controlled single-factor comparison.

The pipeline has five blocks: data, MLP model, loss function, portfolio construction, and evaluation. Within each comparison table, the data, features, training window, test window, portfolio construction, and evaluation metrics are fixed. The loss function is the treatment.

This design is important because the project is not an architecture-search study. The question is narrower: conditional on a fixed prediction pipeline, can loss-function design improve the portfolio signal?

The final recommendation is scoped to one static 24-month test window, one feature set, one MLP width configuration, and three-seed robustness rows.

---

## Slide 5 - Contributions

There are four contributions.

First, I design additive and multiplicative hybrid losses that combine directional alignment with robust magnitude control.

Second, I extend the multiplicative family into an M2-robust gamma family by adding a prediction-variance penalty.

Third, I use staged evidence: single-seed baselines, single-seed hybrid sweeps, multi-seed robustness tests, and a component-normalisation probe.

Fourth, I give a bounded recommendation: gamma07 as the primary choice, gamma10 as a high-return but seed-sensitive alternative, and alpha06 as a stable fallback.

---

## Slide 6 - Research Design And Architecture

The model is a multi-layer perceptron with 15 inputs and hidden widths 64, 32, and 16. The output is one scalar predicted one-month-ahead return for each stock-month observation.

The 15 inputs come from feature set X1. There are 10 engineered columns: cumulative return and cumulative turnover at 1, 3, 6, 9, and 12-month horizons. These are combined with 5 base panel columns: RET, VOL, SHROUT, r, and turnover.

The training window is January 1990 to December 1994. The test window is January 1995 to December 1996, giving 24 out-of-sample monthly portfolio returns. Training uses Adam with learning rate 0.001, batch size 1024, and 20 epochs.

One technical boundary matters. Phase 1 and Phase 2 use ReLU with dropout 0.2. Phase 3 keeps the same layer widths but uses tanh with dropout 0.0. Therefore, within-phase comparisons are the strongest evidence. Cross-phase comparisons are useful for design interpretation, but I do not treat them as exact single-factor improvement claims.

---

## Slide 7 - Loss Function Families

This is the key methodology slide.

The first family is regression loss. MSE is the standard squared residual loss. MedSE is the median of squared residuals, so it is more robust to extreme observations in each batch.

The second family is directional loss. MADL and GMADL use the product y times y_hat. If this product is positive, realised and predicted returns have the same sign. If it is negative, the sign is wrong. Directional losses are closer to the trading objective, but they can lose control over prediction magnitude.

The third family is additive hybrid loss:

L_add = lambda_dir D(y, y_hat) + lambda_hub H_delta(y - y_hat).

This adds a directional penalty and a Huber magnitude term.

The fourth family is my main design focus: the multiplicative hybrid:

L_mul = (1 + lambda_dir D(y, y_hat)) H_delta(y - y_hat).

Here, H_delta is the Huber backbone. It is quadratic for small residuals and linear for large residuals, so it gives magnitude control without letting outliers dominate.

D(y, y_hat) is the directional gate:

D = [1 - sigmoid(a y y_hat)] |y|^b / (batch mean of |y|^b + epsilon).

This gate has three roles. First, it is sign-sensitive: it is small when the prediction sign is correct, and larger when the sign is wrong. Second, it is magnitude-aware: wrong signs on large realised returns receive more penalty. Third, it is batch-normalised, so the directional scale is not driven only by which observations appear in a mini-batch.

The multiplicative interpretation is direct. When direction is correct, the loss is close to Huber. When direction is wrong, the Huber loss is amplified.

The M2-robust extension adds prediction-variance regularisation:

L_M2-robust = L_M2 + gamma Var(y_hat).

Gamma controls stability. Too little gamma can under-regularise predictions; too much can over-compress the ranking signal. The experiment scans gamma from 0.3 to 1.5.

---

## Slide 8 - Portfolio Construction And Evaluation

Every loss is evaluated through the same portfolio rule.

For each test month, the model produces a cross-sectional prediction vector. I rank stocks by predicted return. The top 10 percent form the long bucket, and the bottom 10 percent form the short bucket.

Within each bucket, predictions are converted to z-scores and clipped to the range from minus 3 to plus 3. Then I apply sign-consistent weights, so stronger positive signals receive more long weight and stronger negative signals receive more short weight. Finally, each stock weight is capped at 5 percent.

The long-short return is the long bucket return minus the short bucket return.

The main metric is annualised Sharpe, computed as sqrt(12) times monthly mean divided by monthly standard deviation. For multi-seed rows, I use CV, equal to Sharpe standard deviation across seeds divided by absolute mean Sharpe. R-squared is reported only as a scale diagnostic.

---

## Slide 9 - Experimental Phases

The empirical design is staged rather than one pooled leaderboard.

Phase 1 compares seven baseline losses at seed 42. Phase 2 sweeps nine hybrid variants, also at seed 42, to identify promising directions.

Phase 3a evaluates five gamma values in the M2-robust family with three seeds per row. Phase 3b compares other alpha, beta, and adaptive-lambda families. Phase 4 applies component normalisation to the leading candidates.

All headline rows use the same 24-month test window from 1995-01 to 1996-12. But single-seed and multi-seed tables support different claim strengths, so I keep their interpretations separate.

---

## Slide 10 - Data And Features

The data are a CRSP-style monthly US equity panel. The raw files cover December 1989 to December 2024, but the main experiments use a static split.

The training period is January 1990 to December 1994. In the training-era source file, there are 449,018 observations and 10,987 unique securities. The test period is January 1995 to December 1996, with 24 monthly out-of-sample portfolio returns.

The target is one-month-ahead return for the same security. It is created by shifting return forward within each PERMNO. Features at time t use information known at or before time t, while the target is realised at t plus 1. This prevents look-ahead leakage.

The final feature set is X1: cumulative returns and cumulative turnover over 1, 3, 6, 9, and 12 months, plus the base panel variables. No alternative feature set enters the final headline tables.

A key choice is no winsorisation. The training return distribution has mean 1.00 percent, standard deviation 19.54 percent, minimum minus 98.8 percent, and maximum plus 2400 percent. I keep these tails because they are exactly why robust loss design is relevant.

The data limitations are also clear: US monthly equities only, one static window, unverified delisting-return completeness, and no rolling-regime evaluation.

---

## Slide 11 - Phase 1 Baseline Loss Comparison

Phase 1 compares seven baseline losses at seed 42.

MSE performs poorly as a portfolio objective here: Sharpe is minus 0.4643 and cumulative return is minus 11.25 percent. MedSE is only slightly positive, with Sharpe 0.0932 and cumulative return plus 0.60 percent.

A more important observation is that R-squared and portfolio performance can decouple. GMADL has average R-squared around minus 7.02 times 10 to the 9, but its portfolio Sharpe is positive at 0.2025. This happens because R-squared measures calibrated point prediction, while the portfolio trades ranks.

The best baseline row is hybrid_mul_m1, with Sharpe 0.4435 and cumulative return plus 5.09 percent. This suggests that combining a directional component with robust magnitude control gives a better ranking signal than MSE or pure directional loss alone.

---

## Slide 12 - Phase 2 Hybrid A/M Sweep

Phase 2 compares additive and multiplicative hybrid variants.

The additive family peaks at A3, with Sharpe 0.5738 and cumulative return plus 8.13 percent. The multiplicative family peaks at M1, with Sharpe 0.4435 and cumulative return plus 5.09 percent.

But the table also shows sensitivity. A5 collapses to Sharpe minus 0.4110, and M3 collapses to minus 0.9691. So the conclusion is not that all hybrids work. The form and weighting of the components matter.

The reason the later phase focuses on the multiplicative side is that its prediction scale remains more controlled in this comparison. Phase 2 therefore motivates a multi-seed refinement of the multiplicative robust design through the gamma variance penalty.

---

## Slide 13 - Phase 3a Gamma Refinement

Phase 3a is the core robustness evidence.

The M2-robust gamma loss is:

L = L_M2 + gamma Var(y_hat).

Gamma controls prediction dispersion. If gamma is too small, predictions can be unstable. If gamma is too large, predictions can be over-compressed and lose ranking signal.

The results show a non-monotone trade-off. Gamma03 is unstable, with mean Sharpe 0.3234 and CV 1.0570. Gamma05 improves to Sharpe 0.7054 and CV 0.2109. Gamma07 achieves mean Sharpe 0.9156, mean cumulative return plus 27.99 percent, and the lowest CV, 0.1808.

Gamma10 has the highest mean Sharpe, 1.0043, but CV is 0.5613, about three times larger than gamma07. Gamma15 also loses stability compared with gamma07.

So gamma07 is not selected because it maximises one metric. It is selected because it gives the best joint Sharpe-stability profile.

---

## Slide 14 - Integrated Alpha, Beta, And Lambda Sweeps

The integrated sweep checks whether gamma07 is just a local success.

The IMADL-m2 alpha family provides useful corroboration. Alpha06 reaches mean Sharpe 0.6895, CV 0.2443, and cumulative return plus 30.42 percent. It is below gamma07 in Sharpe, but it is a stable positive result from a related parameterisation.

The beta family does not produce robust positive Sharpe; its CV values can become extremely large. The adaptive-lambda family also underperforms on stability.

This supports the broader conclusion that the productive region is hybrid-multiplicative plus robust control, not pure directional loss or arbitrary adaptive weighting.

---

## Slide 15 - Loss-Component Normalisation Probe

This slide tests a possible objection: maybe the leading results are caused by scale imbalance between loss components.

The normalisation probe equalises component scales and re-runs three leading candidates across three seeds.

Gamma07 is approximately stable. Its mean Sharpe changes from 0.9156 to 0.9112, which is a very small change relative to seed dispersion.

Gamma10 is scale-sensitive. It drops from 1.0043 to 0.4072 after normalisation. Alpha06 collapses from 0.6895 to minus 0.0161.

So normalisation is not a universal fix. The important result is that gamma07 survives the diagnostic, while the other two leading candidates degrade materially.

The caveat is that the scale ratios are diagnostics-estimated. A future version should implement a per-component logger to measure each loss term directly during training.

---

## Slide 16 - Headline Findings And Cumulative Paths

The cumulative return paths make the same result visible.

MSE sends the portfolio downward over the test window. Gamma10 can reach a higher best-seed path, but its seed envelope is wide. Gamma07 produces a more consistent upward path across all three seeds.

This is why the final answer is conditional but actionable: gamma07 is positive, stable, and probe-resistant within the studied protocol.

---

## Slide 17 - Answering The Three Questions

Returning to the research questions:

For RQ1, loss choice matters. R-squared and Sharpe can decouple because the portfolio trades ranks, not calibrated values.

For RQ2, the best supported hybrid design is m2_robust_gamma07: mean Sharpe 0.9156, CV 0.1808, and cumulative return plus 27.99 percent across three seeds.

For RQ3, only gamma07 is approximately stable under the component-normalisation probe. Gamma10 and alpha06 degrade materially.

So the answers are not universal claims, but they are clear within this evidence boundary.

---

## Slide 18 - Recommendation, Limitations, And Future Work

My final recommendation has three tiers.

Primary: m2_robust_gamma07. It has mean Sharpe 0.9156, CV 0.1808, and cumulative return plus 27.99 percent. It is the best-supported single choice because it combines high Sharpe, low seed sensitivity, and normalisation-probe stability.

High-return alternative: m2_robust_gamma10. It has the highest mean Sharpe, 1.0043, but CV is 0.5613 and normalised Sharpe drops to 0.4072. I would only choose it if I explicitly accept seed sensitivity.

Stable fallback: imadl_m2_alpha06. It has mean Sharpe 0.6895, CV 0.2443, and cumulative return plus 30.42 percent. It corroborates the hybrid-multiplicative region, but it is not stable under normalisation.

The main limitations are one static 24-month window, three seeds, one feature set, one architecture family, a gross-of-cost portfolio, and no per-component logger yet.

Future work should use rolling windows, at least 10 seeds, transaction-cost-aware evaluation, feature-set sensitivity tests, and direct component logging.

---

## Slide 19 - References And Closing

The references listed here cover machine-learning asset pricing, momentum features, robust regression, directional loss design, and Sharpe-ratio evaluation.

The main takeaway is that loss-function design should be treated as a first-class design variable for portfolio-oriented prediction. Under the controlled protocol in this project, a multiplicative directional-robust loss with gamma equal to 0.7 gives the best supported Sharpe-stability trade-off.

Thank you for listening. I am happy to take your questions.

---

# Q&A Preparation

## Q1. Why not optimise Sharpe ratio directly?

Direct Sharpe optimisation is difficult because the portfolio includes ranking, top-bottom selection, clipping, and capped-simplex projection. These operations are non-smooth or discontinuous with respect to individual predictions. The losses in this project are differentiable proxies that can be trained with standard backpropagation.

## Q2. Why is R-squared so negative for some losses?

R-squared measures point-prediction calibration. Directional losses can produce predictions on a poor numerical scale while still preserving useful ranks. Since the portfolio trades ranks, R-squared is a scale diagnostic rather than the main performance metric.

## Q3. Why choose gamma07 instead of gamma10?

Gamma10 has higher mean Sharpe, but it is much less stable: CV 0.5613 versus 0.1808 for gamma07. Gamma10 also drops sharply under normalisation, from 1.0043 to 0.4072. Gamma07 is better on the joint criterion.

## Q4. Does the Phase 3 architecture difference weaken the claim?

It weakens direct cross-phase improvement claims, which is why the report avoids them. The final recommendation relies on within-Phase-3 and Phase-4 comparisons, where rows share the same configuration.

## Q5. Why only three seeds?

Three seeds are enough to reveal large CV differences, but not enough for precise confidence intervals. This is why the report treats CV as an order-of-magnitude stability measure and lists 10 or more seeds as future work.

## Q6. Why no rolling window?

The static window gives stronger internal control for loss comparison. Rolling windows would test regime robustness, but they also add retraining and regime-shift effects. That is the next extension.

## Q7. Why no winsorisation?

The heavy tails are part of the problem. If they are clipped first, the experiment becomes less informative about robust loss design. This choice improves internal relevance but limits generalisation to this preprocessing pipeline.

## Q8. What about transaction costs?

The portfolio is gross of transaction costs and shorting costs. Since different losses may create different turnover, cost-aware evaluation could change the final ranking. This is a key future-work item.

## Q9. Why not recommend A3?

A3 is a strong seed-42 result, but it does not have the same multi-seed robustness and normalisation-probe support as gamma07. The final recommendation should come from the stronger evidence tier.

## Q10. What is the main practical contribution?

The project shows a controlled way to treat the loss function as a portfolio-design variable. A loss can encode directional correctness, robust magnitude control, and prediction-dispersion regularisation, and these choices materially affect portfolio performance.

---

# If Time Is Short

Keep Slides 7, 10, 13, 15, and 18 at full length.

Compress Slide 3 to the three literature streams only.

Compress Slide 5 to one sentence: "The contribution is hybrid loss design, variance regularisation, staged evidence, and a bounded recommendation."

Compress Slide 14 to one sentence: "Alpha06 corroborates the hybrid-multiplicative region, while beta and adaptive-lambda do not reach the preferred stability zone."
