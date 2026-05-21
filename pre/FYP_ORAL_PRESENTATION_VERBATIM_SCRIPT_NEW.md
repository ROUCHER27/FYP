# FYP Oral Presentation Verbatim Script - New Draft (Revised)

Target delivery: 15 to 20 minutes for the talk, plus the required 5 to 10 minutes for examiner questions. The main script deliberately spends more time on loss-function design, data, and methodology.

Source basis: actual 19-slide deck, final LaTeX chapters, `ppt_package/tables/`, and the oral-presentation requirement PDF.

---

## Slide 1 - Title

Good morning, everyone. My name is Yirong Yu, student ID 2253235, from BSc Financial Mathematics. My final year project is titled "Multiplicative Directional-Robust Loss for Cross-Sectional Stock-Return Prediction," supervised by Dr. Yi Cao.

The main idea of the topic is that stock-return models are usually trained to minimise prediction error through loss function, and the portfolio later trades the ranking of predictions. My project asks whether changing the loss function can better align model training with the long-short portfolio objective.

---

## Slide 2 - Background And Three Questions

Machine learning has been widely applied to financial return prediction and quantitative trading.
Our goal is not to forecast exact return values but to identify which stocks will outperform and which will underperform.

A long-short portfolio uses these predictions in a specific way: it ranks all stocks by predicted return, buys the top group, and shorts the bottom group. The strategy's return is the spread between winners and losers. So what matters is the **ordering** — as long as the model correctly separates better stocks from worse stocks, the exact predicted number does not matter.

Here is the mismatch. MSE — mean squared error — is the most common training loss. It asks the model to match realised returns in numerical magnitude. MSE cares about estimator numerical precision, not ranking or direction.
==**强调 MSE 的局限性**——它在乎的是预测本身的精确度，而不是方向或排名。==

So the model is trained to minimise numerical error, but the portfolio trades on rank. 
This leads to three research questions. 
First, if and how does loss function choice affect prediction-level and portfolio-level performance?
Second, which hybrid loss design gives the best Sharpe-stability trade-off? 
Third, do the leading candidates remain stable under component-normalisation diagnostics, or are they just scale artefacts?

---

## Slide 3 - Literature Snapshot

The project sits between three research streams.

First, machine learning for stock prediction. Gu, Kelly, and Xiu show that nonlinear machine-learning models can improve return prediction, while Daniel and Moskowitz and Medhat and Schmeling motivate the return and turnover features used in this project.

**Second, robust regression.** MSE is sensitive to outliers. Huber (1964) introduced the idea of using a loss that is quadratic for small errors but linear for large errors, limiting outlier influence. MedSE applies the same median-based robustness principle. These reduce the effect of heavy-tailed residuals, but they do not directly encode directional correctness.  ==y=true, **ŷ（y_pred）**==

Until 2024, we have new directional losses.** Sakowski and his team from**University of Warsaw**  proposed MADL and its generalisation GMADL, which reward sign alignment between realised and predicted returns. 
That is closer to a trading problem, however, a pure directional losse may still lose prediction-scale control.

After review all the Previous works. I found that No one has put them all under the same controlled portfolio protocol — so this is the gap and I'm trying to figure out.

---

## Slide 4 - Controlled Pipeline

The methodology is a controlled single-factor comparison.

The pipeline has five blocks: 

First, I download CRSP amarican data from  xxxxx
Then we need to find a model architecture to fix and It should satisfied the data structure -  **Our task is cross-sectional prediction: each sample is a fixed-length feature vector summarizing one stock at one point in time, with no sequential ordering across samples — MLP's simple matrix multiplication is a natural fit.**

And Loss function plays the role to help mlp model to predict the correct signs by **rewarding direction-aligned predictions and penalizing direction-mismatched ones.**

==For the portfolio, **Each month, we rank all stocks by model prediction, go long the top 10% and short the bottom 10%. Within each bucket, we assign weights via z-score of predictions, then cap any single stock at 5% maximum weight to prevent concentration risk — ensuring no single position dominates the portfolio.**==
# *later*

**For prediction, we track directional accuracy (% of correct sign predictions) and R². For portfolio performance — which is the ultimate test — we compute the annualized Sharpe ratio and cumulative return of a long-short strategy that goes long the top-decile stocks and short the bottom-decile each month.**

---

## Slide 5 - Contributions

After introducing the whole procedure. I also want to talk about the four main contributions.

First, I design additive and multiplicative hybrid losses that combine directional alignment with robust magnitude control.  

**Second, I extend the multiplicative family into an M2-robust gamma family by adding a prediction-variance penalty — the term γ · Var(ŷ) discourages the model from producing extreme or volatile predictions, which reduces monthly return variance and improves cross-seed stability of the portfolio's Sharpe ratio.**

**Third, my experiments follow a progressive design: first confirmed each loss works on a single seed, then sweep the hybrid parameters, then did a stress-test the best candidates across multiple seeds, and finally designed a normalization to check whether the loss components are numerically balanced.**

Fourth, I will deliver a recommendation with clear limits: which loss function is the primary choice, and why there are two alternative options.

---

## Slide 6 - Research Design And Architecture
The model is a multi-layer perceptron with 15 inputs and hidden widths 64, 32, and 16. The output is one scalar predicted one-month-ahead return for each stock-month observation.
The first step of the whole research is to find the hyper parameter from our history data.

For feature engineering, I construct Feature Set X1 using two raw variables: monthly stock return and monthly turnover ratio. 

**I shift each stock's history by one month to avoid look-ahead bias, then compute cumulative returns and cumulative turnover over five windows — 1, 3, 6, 9, and 12 months.

**For model configuration, I run a grid search over 64 combinations — varying layer sizes, activation functions, dropout, learning rate, and batch size. The winning configuration is a three-layer MLP with 64, 32, and 16 neurons, tanh activation, no dropout, trained with Adam optimiser for 20 epochs with a batch size of 1024. This achieved the lowest validation MSE of 0.0226.**

==*加上解释*==
Additionaly, there actually exist a mistake, 

---

## Slide 7 - Loss Function Families

This is the key methodology slide. The title says "four loss families converge on one design" — so let me walk through all four, because they share the same building blocks shown on this slide.

Start with the two core components. On the slide you can see **D** — the directional gate. D equals one minus sigmoid of a times y times y-hat, multiplied by a batch-normalised magnitude weight, absolute y to the power b divided by its batch mean. When prediction and realised return share the same sign, a times y times y-hat is large and positive, sigmoid saturates near one, so D drops to near zero. When they disagree in sign, sigmoid drops toward zero and D rises. The parameters are a equals 100 and b equals 2.

Next to it — **H delta**, the Huber backbone. Huber is quadratic when the residual is smaller than delta, and linear when larger. Delta equals 0.01, matching the scale of monthly returns.  So for small prediction errors, Huber behaves like MSE and pushes the model to fine-tune. But for extreme outliers — like a stock that returns plus 2400 percent in one month — Huber caps the penalty at a linear rate instead of letting the squared term explode. This prevents a handful of extreme observations from dominating the entire gradient.

So In the formal experiments, I actually set four sets loss functions.
**Family one: regression type.** MSE is the standard baseline — it minimises squared error. MedSE replaces the mean with the median, so it is more robust to outliers. But neither of them cares about prediction direction.

**Family two: directional.** MADL and GMADL reward the model when predicted and realised returns share the same sign. 

**Family three: additive hybrid.** As shown in the formula here — we simply add a directional penalty and a Huber magnitude term together. It works, but the two components can fight each other at different scales.

**Family four: multiplicative hybrid — my main design focus.** The formula is here. The intuition is simple: the Huber term is the backbone that controls magnitude. The directional gate — D, shown on the slide with a equals 100 and b equals 2 in the implementation — acts as a multiplier. When direction is correct, D is near zero, and the loss is just Huber. When direction is wrong, D amplifies the Huber loss. So wrong-direction predictions on large-return stocks get penalised the most.

One more extension — and I will show the results on this later in Slide 13 — we add a prediction-variance penalty controlled by a parameter gamma. The base multiplicative loss here uses lambda_dir equal to 2. Gamma controls how much the model is allowed to spread its predictions apart. Too little gamma means instability across seeds; too much gamma compresses the signal. The experiment scans gamma from 0.3 to 1.5 to find the sweet spot.

---

## Slide 8 - Portfolio Construction And Evaluation

Every loss is evaluated through the same portfolio rule.

For each test month, the model produces a cross-sectional prediction vector. I rank stocks by predicted return. The top 10 percent form the long bucket, and the bottom 10 percent form the short bucket.

Within each bucket, predictions are converted to z-scores and clipped to the range from minus 3 to plus 3. Then I apply sign-consistent weights, so stronger positive signals receive more long weight and stronger negative signals receive more short weight. Finally, each stock weight is capped at 5 percent.

The portfolio return is simply long minus short.

The main metric is annualised Sharpe, computed as sqrt(12) times monthly mean divided by monthly standard deviation. For multi-seed rows, I use CV, equal to Sharpe standard deviation across seeds divided by absolute mean Sharpe. R-squared is reported only as a scale diagnostic.

---

## Slide 9 - Experimental Phases

The empirical design is staged rather than one pooled leaderboard.

Phase 1 compares seven baseline losses at seed 42. Phase 2 sweeps nine hybrid variants, also at seed 42, to identify promising directions.

Phase 3a evaluates five gamma values in the M2-robust family with three seeds per row. Phase 3b compares other alpha, beta, and adaptive-lambda families. Phase 4 applies component normalisation to the leading candidates.

One operational note: Phase 1 and 2 use ReLU with dropout 0.2; Phase 3 uses tanh with dropout zero. This happened because the multi-seed branch was developed separately and inherited a different config. The important point is: within each phase, every row shares exactly the same model — only the loss changes. My final recommendation draws entirely from within-Phase-3 evidence, so the activation difference does not affect the conclusion.

All headline rows use the same 24-month test window from 1995-01 to 1996-12. But single-seed and multi-seed tables support different claim strengths, so I keep their interpretations separate.

---

## Slide 10 - Data And Features

The data are a CRSP-style monthly US equity panel. The raw files cover December 1989 to December 2024, but the main experiments use a static split.

The training period is January 1990 to December 1994. The training-era source file contains 449,018 rows covering 10,987 unique securities across 60 training months. The test period is January 1995 to December 1996, with 24 monthly out-of-sample portfolio returns.

The target is one-month-ahead return for the same security. It is created by shifting return forward within each PERMNO. Features at time t use information known at or before time t, while the target is realised at t plus 1. This prevents look-ahead leakage.

The final feature set is X1: cumulative returns and cumulative turnover over 1, 3, 6, 9, and 12 months, plus the base panel variables. No alternative feature set enters the final headline tables.

A key choice is no winsorisation. The training return distribution has mean 1.00 percent, standard deviation 19.54 percent, minimum minus 98.8 percent, and maximum plus 2400 percent. I keep these tails because they are exactly why robust loss design is relevant.

The data limitations are also clear: US monthly equities only, one static window, unverified delisting-return completeness, and no rolling-regime evaluation.

---

## Slide 11 - Phase 1 Baseline Loss Comparison

Phase 1 compares seven losses at seed 42.

MSE performs poorly as a portfolio objective here: Sharpe is minus 0.4643 and cumulative return is minus 11.25 percent. MedSE is only slightly positive, with Sharpe 0.0932 and cumulative return plus 0.60 percent.

A more important observation is that R-squared and portfolio performance can decouple. GMADL has average R-squared around minus 7.02 times 10 to the 9, but its portfolio Sharpe is positive at 0.2025. This happens because R-squared measures calibrated point prediction, while the portfolio trades ranks.

The best-performing row in Phase 1 is hybrid_mul_m1, with Sharpe 0.4435 and cumulative return plus 5.09 percent. This suggests that combining a directional component with robust magnitude control gives a better ranking signal than MSE or pure directional loss alone.

Notice the key decoupling: GMADL has the worst R-squared in the table — minus 7 billion — but its portfolio Sharpe is positive. This tells us the portfolio trades ranks, not calibrated values. Once you accept this, the question becomes: how do we design a loss that produces better ranks? That is exactly what the next slides answer.

---

## Slide 12 - Phase 2 Hybrid A/M Sweep

Phase 2 compares additive and multiplicative hybrid variants.

The additive family peaks at A3, with Sharpe 0.5738 and cumulative return plus 8.13 percent. A3 reaches 0.57 — already more than double what any pure regression or directional baseline achieves. The multiplicative family peaks at M1, with Sharpe 0.4435 and cumulative return plus 5.09 percent.

But the table also shows sensitivity. A5 collapses to Sharpe minus 0.4110, and M3 collapses to minus 0.9691. So the conclusion is not that all hybrids work. The form and weighting of the components matter.

Why do A5 and M3 collapse? In both cases, the directional weight is too large relative to the magnitude backbone. The model starts chasing sign-correctness so aggressively that it distorts the ranking signal. This is why the later gamma family adds explicit variance control — to prevent this kind of over-correction.

The reason the later phase focuses on the multiplicative side is that its prediction scale remains more controlled in this comparison. Phase 2 therefore motivates a multi-seed refinement of the multiplicative robust design through the gamma variance penalty.

---

## Slide 13 - Phase 3a Gamma Refinement

Phase 3a is the core robustness evidence.

The M2-robust gamma loss is: L = L_M2 + gamma times Var(y_hat). Gamma controls prediction dispersion. If gamma is too small, predictions can be unstable. If gamma is too large, predictions can be over-compressed and lose ranking signal.

Remember MSE gives Sharpe minus 0.46. Now look at the results — they show a non-monotone trade-off.

Gamma03 is unstable, with mean Sharpe 0.3234 and CV 1.0570 — meaning some seeds produce negative Sharpe. Gamma05 improves to Sharpe 0.7054 and CV 0.2109. Gamma07 achieves mean Sharpe 0.9156, mean cumulative return plus 27.99 percent, and the lowest CV, 0.1808. That is a swing of nearly 1.4 Sharpe units from MSE, with the lowest CV in the table.

Gamma10 has the highest mean Sharpe, 1.0043, but CV is 0.5613, about three times larger than gamma07. Gamma15 also loses stability compared with gamma07.

Why does gamma07 work? The intuition is: gamma 0.7 allows enough prediction spread to preserve ranking differences between stocks, but not so much that the model's output becomes unstable across seeds. Below 0.7, the model is under-regularised and seed-sensitive. Above 0.7, the signal gets compressed and you start losing return.

So gamma07 is not selected because it maximises one metric. It is selected because it balances high Sharpe and low seed sensitivity — the best of both.

---

## Slide 14 - Integrated Alpha, Beta, And Lambda Sweeps

The gamma sweep gives us one winner. But is gamma07 just a lucky point? Or is the whole hybrid-multiplicative region productive?

The answer is: the region is productive. Alpha06 — a related parameterisation from the IMADL-m2 family — reaches mean Sharpe 0.6895, CV 0.2443, and cumulative return plus 30.42 percent. It is below gamma07, but it is stably positive. This corroborates the broader design direction.

On the other hand, the beta family and the adaptive-lambda family do not reach the stability zone. Their CV values are large and their mean Sharpe is inconsistent.

So the conclusion is not just "gamma07 is good." It is: "multiplicative hybrid plus variance control is the productive design space. Gamma07 is the best point within it."

---

## Slide 15 - Loss-Component Normalisation Probe

This slide tests a possible objection: maybe the leading results are caused by scale imbalance between loss components.

The normalisation probe forces the two loss components to have equal scale, then re-runs the experiments — three leading candidates across three seeds.

Gamma07 is approximately stable. Its mean Sharpe changes from 0.9156 to 0.9112, which is a very small change relative to seed dispersion.

Gamma10 is scale-sensitive. It drops from 1.0043 to 0.4072 after normalisation. Alpha06 collapses from 0.6895 to minus 0.0161.

So normalisation is not a universal fix. The important result is that gamma07 survives the diagnostic, while the other two leading candidates degrade materially.

The caveat is that the scale ratios are diagnostics-estimated. A future version should implement a per-component logger to measure each loss term directly during training.

---

## Slide 16 - Headline Findings And Cumulative Paths

The cumulative return paths make the same result visible.

While MSE loses 11 percent over two years, gamma07 gains 28 percent. Gamma10 can reach a higher best-seed path, but its seed envelope is wide. Gamma07 produces a more consistent upward path across all three seeds.

This is why the final answer is conditional but actionable: gamma07 is positive, stable, and probe-resistant within the studied protocol.

---

## Slide 17 - Answering The Three Questions

Returning to the research questions:

For RQ1, loss choice matters. R-squared and Sharpe can decouple because the portfolio trades ranks, not calibrated values. Loss choice takes you from minus 0.46 to plus 0.92 — a swing of 1.38 Sharpe units.

For RQ2, the best supported hybrid design is m2_robust_gamma07: mean Sharpe 0.9156, CV 0.1808, and cumulative return plus 27.99 percent across three seeds. That is roughly 20 times the best regression baseline MedSE, and twice the best single-seed hybrid.

For RQ3, only gamma07 is approximately stable under the component-normalisation probe. Gamma10 and alpha06 degrade materially.

These are not universal claims — they hold within this specific protocol.

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

The takeaway is: how you define the loss function matters as much as how you build the model. Under the controlled protocol in this project, a multiplicative directional-robust loss with gamma equal to 0.7 gives the best supported Sharpe-stability trade-off.

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

The activation and dropout difference affects Sharpe absolute values, but is unlikely to change the relative ranking among gamma values. The reason is that the optimal strength of the variance penalty follows a non-monotone pattern — too small is unstable, too large kills the signal — and this pattern is a mathematical property of the penalty itself, not dependent on which activation the hidden layers use. The Phase 3 ranking conclusion — gamma07 is the best balance point — is safe under this level of perturbation.

More specifically: within Phase 3, every row shares exactly the same config (tanh, dropout zero, same seeds). The final recommendation draws entirely from within-Phase-3 evidence. I do not claim that gamma07 beats MSE by exactly 1.38 Sharpe units across configurations — I claim that among the M2-robust family under identical conditions, gamma07 has the best Sharpe-CV trade-off.

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

## Q11. Why did Phase 3 change from ReLU to tanh and dropout 0.2 to 0.0?

Phase 3's core purpose is testing the gamma variance penalty — L = L_M2 + gamma * Var(y_hat). If we keep ReLU plus dropout 0.2, the model already contains two mechanisms that affect prediction dispersion: ReLU truncates negative activations creating sparse representations, and dropout randomly shuts down hidden units during training. These mechanisms themselves change prediction variance, making it harder to isolate gamma's effect.

Tanh preserves information in both positive and negative directions and has bounded output; dropout zero avoids additional random masking during training. This cannot eliminate all confounding, but it lets the gamma sweep more directly reflect the variance penalty's influence.

The key evidence is: if the activation alone drove the results, all gamma values would perform similarly — but they do not. Gamma03 is unstable (CV 1.06), gamma07 is balanced, gamma15 over-compresses. The non-monotone pattern confirms that gamma is doing additional work beyond what tanh provides.

---

# If Time Is Short

Keep Slides 7, 10, 13, 15, and 18 at full length.

Compress Slide 3 to the three literature streams only.

Compress Slide 5 to one sentence: "The contribution is hybrid loss design, variance regularisation, staged evidence, and a qualified recommendation."

Compress Slide 14 to one sentence: "Alpha06 corroborates the hybrid-multiplicative region, while beta and adaptive-lambda do not reach the preferred stability zone."
