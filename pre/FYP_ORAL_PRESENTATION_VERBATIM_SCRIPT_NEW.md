# FYP Oral Presentation Verbatim Script - New Draft (Revised)

Target delivery: 15 to 20 minutes for the talk, plus the required 5 to 10 minutes for examiner questions. The main script deliberately spends more time on loss-function design, data, and methodology.

Source basis: actual 19-slide deck, final LaTeX chapters, `ppt_package/tables/`, and the oral-presentation requirement PDF.

---

## Slide 1 - Title

Good morning, everyone. I''m Yirong Yu, from Financial Mathematics. My final year project is titled "Multiplicative Directional-Robust Loss for Cross-Sectional Stock-Return Prediction," supervised by Dr. Yi Cao.

The main idea of the topic is that stock-return prediction models in machine learning are usually trained to minimise prediction error through loss function. My project asks whether changing the loss function can better align model training with a fixed long-short portfolio.

---

## Slide 2 - Background And Three Questions

The starting point is a mismatch between how we _train_ the model and how we _use_ it.

In this task, a nueral network takes stock features as input and outputs a predicted return for each stock. After training, those predictions are ranked cross-sectionally: the top 10 percent form the long bucket, the bottom 10 percent form the short bucket, and we have a long-short portfolio.

Now here is the problem. The loss function is what teaches the Model during training — it scores the model's predictions against realised returns and tells the model how to update its weights to improve its prediction accuracy. 

The default choice is MSE, which scores the model purely on how close the predicted value is to the realised value in magnitude. 

But the portfolio only cares about the _ranking_ of predictions, not exact values. So MSE is optimising for the wrong thing — it is teaching the model to match magnitudes, when what actually matters for portfolio performance is getting the relative order right.

This leads to the central question: can we design a better scoring rule for the Model? — one that rewards correct direction and penalises wrong direction, instead of just minimising numerical distance? 

This breaks into three research questions.:

First, if and how does loss function choice affect prediction-level and portfolio-level performance?
Second, which hybrid loss design gives the best Sharpe-stability trade-off? 
Third, do the leading candidates remain stable under component-normalisation diagnostics, or are they just scale artefacts?

---

## Slide 3 - Literature Snapshot

The project sits between three research streams.

First, machine learning for stock prediction. Gu, Kelly, and Xiu show that nonlinear machine-learning models can improve return prediction, while Daniel and Moskowitz and Medhat and Schmeling motivate the return and turnover features used in this project.

**Second, robust regression.** MSE is sensitive to outliers. Huber (1964) introduced the idea of using a loss that is quadratic for small errors but linear for large errors, limiting outlier influence. MedSE applies the same median-based robustness principle. These helps reduce the effect of heavy-tailed residuals, but they don't directly encode directional correctness.  ==y=true, **ŷ（y_pred）**==

Until 2024, we have new directional losses.** Sakowski and his team from**University of Warsaw**  proposed MADL and its generalisation GMADL, which reward sign alignment between true value and predicted returns. 

Compared to MSE, MADL makes three improvements: it rewards sign-correct predictions instead of penalising numerical distance, it weights the loss by ==absolute value== |y| so large-return stocks contribute more to training, and the tanh wrapper bounds the directional signal to limit outlier influence.
That is closer to the trading problem, however, a pure directional losse may still lose prediction-scale control.

After review all the Previous works. I found that No one has put them  all under the same controlled portfolio protocol — so this is the gap and I'm trying to figure out.

---

## Slide 4 - Controlled Pipeline

The methodology is a controlled single-factor comparison.

The pipeline has five blocks: 

First, I download CRSP amarican data from  Wharton Research Data Services website.

Then I need to find a model architecture and It should satisfied the data structure -  **Our task is cross-sectional prediction: each sample is a fixed-length feature vector summarizing one stock at one point in time, with no sequential ordering across samples — MLP's simple matrix multiplication is a natural fit.**

And Loss function plays the role to help model to predict the correct signs.

==For the portfolio, **Each month, we rank all stocks by model prediction, go long the top 10% and short the bottom 10%. Within each bucket, we assign weights via z-score of predictions, then cap any single stock at 5% maximum weight to prevent concentration risk — ensuring no single position dominates the portfolio.**==
# *later*

For prediction, we track directional accuracy (% of correct sign predictions) and R². 

For portfolio performance — which is the ultimate test — we compute the annualized Sharpe ratio and cumulative return of a long-short strategy that goes long the top-decile stocks and short the bottom-decile each month.**

As the experimental objectives changes, I will correspondingly modify elements within the matrix.

---

## Slide 5 - Contributions

After introducing the whole procedure. I also want to talk about the four main contributions.

First, I design additive and multiplicative hybrid losses that combine directional alignment with robust magnitude control.  

**Second, I extend the multiplicative family into an M2-robust gamma family by adding a prediction-variance penalty

— the term γ · Var(ŷ) discourages the model from producing extreme or volatile predictions, which reduces monthly return variance and improves cross-seed stability of the portfolio's Sharpe ratio.**

**Third, my experiments follow a progressive design step by step

Fourth, I will finally deliver a recommendation with clear limits: which loss function is the primary choice, and there are two alternative options. 

---

## Slide 6 - Research Design And Architecture
The model is a multi-layer perceptron with 15 inputs and hidden widths 64, 32, and 16. The output is one scalar predicted one-month-ahead return for each stock-month observation.
The first step of the whole research is to find the hyper parameter from our history data.

For feature engineering, I construct Feature Set X1 using two raw variables: monthly stock return and monthly turnover ratio. 

==**I compute cumulative returns and cumulative turnover over five windows — 1, 3, 6, 9, and 12 months.==

**For model configuration, I run a grid search over 64 combinations — varying layer sizes, activation functions, dropout, learning rate, and batch size to find The winning configuration .

==*加上解释*==
Additionaly, there actually exist a mistake in activator, but it will not effect the result significantly but slight numerical difference.

---

## Slide 7 - Loss Function Families

This is the key methodology slide. The title says "four loss families converge on one design" — so let me walk through all four, because they share the same building blocks shown on this slide.

Start with the two core components. On the slide you can see **D** — the directional gate.
D has two jobs. The first job is to **judge whether the predicted direction is correct**. We look at the product of y and y-hat — if they share the same sign, the product is positive; if not, it's negative. We pass this product through a sigmoid scaled by a parameter **a**.(because of two percentage multiply) The sigmoid will turn it value between 0,1.

D's second job is to **weight each observation by how much it matters**.

D's second job is to **weight each observation by how much it economically matters**. Larger-return stocks are where the portfolio actually makes or loses money, so we weight each observation by absolute y raised to a power **b**. The value **b equals 2** is taken from the GMADL paper — squaring sharpens the focus on extreme returns without letting a few outliers dominate the loss.

But raw |y|² weighting introduces a new problem. Different batches sample different mixes of large and small movers, so the loss scale would swing between batches just from random sampling. To fix this, we **divide by the batch-mean of the same weight**, keeping the average weight within every batch equal to one. This preserves the relative emphasis on large movers while stabilising the loss scale across batches.


==Parameter **b** controls how aggressively large-return stocks are emphasised.==

Next to it — **H delta**, the Huber backbone. Huber is quadratic when the residual is smaller than delta, and linear when larger. Delta equals 0.01, matching the scale of monthly returns.  So for small prediction errors, Huber behaves like MSE and pushes the model to fine-tune. But for extreme outliers — like a stock that returns plus 2400 percent in one month — Huber caps the penalty at a linear rate instead of letting the squared term explode. This prevents a handful of extreme observations from dominating the entire gradient.

So In the formal experiments, I actually set four sets loss functions.
**Family one: regression type.** MSE . MedSE 

**Family two: directional.** MADL and GMADL reward the model when predicted and realised returns share the same sign. 

**Family three: additive hybrid.** As shown in the formula here — we simply add a directional penalty and a Huber magnitude term together. It works, but the two components can fight each other at different scales.

**Family four: multiplicative hybrid — my main design focus.** The formula is here. The intuition is simple: the Huber term is the backbone that controls magnitude. The directional gate — D, shown on the slide with a equals 100 and b equals 2 in the implementation — acts as a multiplier. When direction is correct, D is near zero, and the loss is just Huber. When direction is wrong, D amplifies the Huber loss. So wrong-direction predictions on large-return stocks get penalised the most.

One more extension — and I will show the results on this later in Slide 13 — we add a prediction-variance penalty controlled by a parameter gamma. The base multiplicative loss here uses lambda_dir equal to 2. 

Var(ŷ) 是模型预测值在 batch 内的**方差**——所有股票的预测值有多分散。
- 如果模型把所有股票都预测成 0.05，方差 = 0（预测高度集中）
- 如果模型把不同股票预测成 -0.3, +0.5, +0.8, -0.2... 方差很大（预测高度分散）
**γ 是这个方差的"惩罚力度"**：
- γ 越大 → loss 越不允许预测分散 → 模型倾向于"压缩"预测，让所有股票的预测值靠拢
- γ 越小 → 几乎不惩罚分散 → 模型可以自由让预测值散开
---

## Slide 8 - Portfolio Construction And Evaluation

Every loss is evaluated through the same portfolio rule.

For each test month, the model produces a cross-sectional prediction vector. I rank stocks by predicted return. The top 10 percent form the long bucket, and the bottom 10 percent form the short bucket.

Within each bucket, predictions are converted to z-scores and clipped to the range from minus 3 to plus 3. Then I apply sign-consistent weights, so stronger positive signals receive more long weight and stronger negative signals receive more short weight. Finally, each stock weight is capped at 5 percent.

The portfolio return is simply long minus short.

The main metric is annualised Sharpe, computed as sqrt(12) times monthly mean divided by monthly standard deviation. 

For multi-seed rows, I use CV, equal to Sharpe standard deviation across seeds divided by absolute mean Sharpe. 
R-squared is reported only as a scale diagnostic.

---

## Slide 9 - Experimental Phases

The empirical design is staged rather than one pooled leaderboard.

Phase 1 compares seven baseline losses at seed 42. Phase 2 sweeps nine hybrid variants, also at seed 42, to identify promising directions.

Phase 3a evaluates five gamma values in the M2-robust family with three seeds per row. Phase 3b compares other alpha, beta, and adaptive-lambda families. Phase 4 applies component normalisation to the leading candidates.

---

## Slide 10 - Data And Features

The data are a CRSP-style monthly US equity panel. The raw files cover December 1989 to December 2024, but the main experiments use a static split.

The training period is January 1990 to December 1994. The training-era source file contains 449,018 rows covering 10,987 unique securities across 60 training months. The test period is January 1995 to December 1996, with 24 monthly out-of-sample portfolio returns.

The target is one-month-ahead return for the same security. It is created by shifting return forward within each PERMNO. Features at time t use information known at or before time t, while the target is realised at t plus 1. This prevents look-ahead leakage.

The final feature set is X1: cumulative returns and cumulative turnover over 1, 3, 6, 9, and 12 months, plus the base panel variables. No alternative feature set enters the final headline tables.

A key choice is no winsorisation. The training return distribution has mean 1.00 percent, standard deviation 19.54 percent, minimum minus 98.8 percent, and maximum plus 2400 percent. I keep these tails because they are exactly why robust loss design is relevant.

---

## Slide 11 - Phase 1 Baseline Loss Comparison

Phase 1 is the first controlled comparison test. There are Seven loss functions — two regression, three directional, and two multiplicative hybrids — are each trained once on the same five-year training window at seed 42, then evaluated on the same 24-month out-of-sample period. Everything except the loss function is frozen

MSE performs poorly as a portfolio objective here: Sharpe is minus 0.4643 and cumulative return is minus 11.25 percent. MedSE is only slightly positive, with Sharpe 0.0932 and cumulative return plus 0.60 percent.

A more important observation is that R-squared and portfolio performance can decouple. GMADL has average R-squared around minus 7.02 times 10 to the 9, but its portfolio Sharpe is positive at 0.2025. This happens because R-squared measures calibrated point prediction, while the portfolio trades ranks. This decoupling tells us: once we accept that the portfolio uses ranking, the question becomes how to design a loss that produces better ranks.

The best-performing row in Phase 1 is hybrid_mul_m1, with Sharpe 0.4435 and cumulative return plus 5.09 percent. Combining a directional component with robust magnitude control gives a better ranking signal than MSE or pure directional loss alone. This motivates the hybrid sweep in Phase 2.


---

## Slide 12 - Phase 2 Hybrid A/M Sweep

So Phase 1 showed us that, under the same protocol, MSE actually destroys value — it gives Sharpe minus 0.46. Among the seven candidate losses we tested, hybrid_mul_m1 came out on top with Sharpe 0.4435. Now the natural question is: can we do better by tuning the hybrid parameters? And does the additive form compete with the multiplicative form?

Phase 2 answers this. Same setup as Phase 1 — same seed, same window, same portfolio. The only thing that changes is which hybrid variant we plug in.

Quick note on the labels. A1 through A5 are different combinations of lambda-dir and lambda-hub — the directional weight and the Huber weight in the additive form. M1 through M4 only vary lambda-dir, since the multiplicative form has just one parameter to tune.

Looking at the results: the additive family peaks at A3 with Sharpe 0.5738. The multiplicative family peaks at M1 with Sharpe 0.4435. A3 looks numerically better, but several variants — A5 and M3 — collapse hard, showing that the parameter choice matters a lot.

I think there are two observations decide what we carry into Phase 3. 
First, the multiplicative form keeps its R-squared in single digits — M1 is minus 4.79 — while the additive form's R-squared explodes into the thousands. So the multiplicative version keeps its predictions on a sensible numerical scale. 
Second, the collapses we just saw — A5 and M3 — point to a specific failure mode: when the directional weight is too strong, predictions become overly dispersed and the ranking signal breaks down. This suggests we need explicit control over how spread out the predictions are.

This is exactly what Phase 3 adds. We take M1's setting — lambda-dir equals 2 — as the fixed base, then add a variance penalty: gamma times the within-batch variance of y-hat. The penalty discourages predictions from spreading too far apart, directly addressing the collapse mechanism. We then evaluate this across three random seeds — not just one — to see whether the design is genuinely robust or just lucky.

---

## Slide 13 - Phase 3a Gamma Refinement

So now we are in Phase 3. As we just saw, the loss is L equals L-M2 plus gamma times the within-batch variance of y-hat — the multiplicative hybrid plus an explicit dispersion penalty. The question this slide answers is: what value of gamma actually works?

Every row is now evaluated across three seeds including previous 42. 
This lets us measure seed sensitivity through CV, the coefficient of variation: standard deviation of Sharpe across seeds divided by the mean. 
Lower CV means the model behaves consistently regardless of random initialisation. 
Within Phase 3, every row uses exactly the same config — only gamma changes.

Now, lets check the results. 
Gamma 0.3 is under-regularised. Mean Sharpe 0.3234, but CV is 1.05 — meaning some seeds actually produce negative Sharpe. The variance penalty is too weak.

Gamma 0.5 improves: Sharpe 0.7054, CV drops to 0.21.

Gamma 0.7 is the sweet spot. Mean Sharpe 0.9156, cumulative return plus 27.99 percent, and the lowest CV in the table at 0.1808. Crucially, no seed produces a negative Sharpe — every seed is positive.

Gamma 1.0 actually has the highest mean Sharpe at 1.0043 — looks tempting. But its CV is 0.56, about three times worse than gamma 0.7. Its best seed is excellent, its worst seed is mediocre. That is not stability.

Gamma 1.5 over-compresses. Both Sharpe and stability degrade.

So combine three charts we can find that
Below 0.7 — the variance penalty is too weak, the model spreads too much, and seed sensitivity kicks in.
Above 0.7 — the penalty is too strong, predictions get squeezed together, and the ranking signal we need for the portfolio gets destroyed. 

---

## Slide 14 - Integrated Alpha, Beta, And Lambda Sweeps

OK, so gamma 0.7 wins in the gamma sweep. But a fair question is: are we just lucky? Maybe we happened to find one good point by coincidence. Or maybe the whole idea — multiplicative hybrid plus some form of regularisation — is genuinely productive, and other entry points into this space would also work.

Phase 3b tests this. I ALSO CHOOSE Same protocol as Phase 3a same three-seed evaluation The only thing that changes is which parameterised family we sweep. I three alternatives.

**IMADL-m2 alpha** is a linear blend of IMADL and M2: L equals alpha times IMADL plus one minus alpha times M2. We scan alpha from 0.2 to 0.8.

**IMADL-GMADL beta** blends two pure directional losses: L equals beta times IMADL plus one minus beta times GMADL. No Huber backbone — pure direction stacking.

**Adaptive Lambda** uses a dynamic weight that depends on realised return magnitude: when y is small, the blend leans toward IMADL; when y is large, it leans toward M2. Lambda controls the transition speed.

The IMADL-m2 alpha family peaks at alpha 0.6 with mean Sharpe 0.6895, CV 0.2443, and cumulative return plus 30.42 percent. It is the best of the three alternatives, and it is the only one that enters the stability zone — CV under 0.35. So we keep this as our stable fallback in the final recommendation. But its Sharpe is still 25 percent below gamma 0.7.

The Adaptive Lambda family does not work. Its best row, lambda 1.0, has mean Sharpe 0.4938 — about half of gamma 0.7 — and CV 1.5426, eight times worse. The dynamic weighting does not produce a stable signal across seeds.

And The IMADL-GMADL beta family obviously cannot compete gamma07


---

## Slide 15 - Loss-Component Normalisation Probe

We have shown that gamma 0.7 wins on the joint Sharpe-stability criterion within Phase 3. But there is one more objection a careful examiner could raise — and it is worth taking seriously.

Recall the multiplicative hybrid combines a directional gate D with a Huber term, and these two pieces operate on very different numerical scales. According to our diagnostics, in the gamma family the directional component is roughly 113 times larger than the magnitude component. In the alpha family it is around 34 times larger. The point is the same in both cases: the two components are not contributing equally to the loss.

Phase 4 tests exactly this. We take the three leading candidates — gamma 0.7, gamma 1.0, and alpha 0.6 — and re-run them with a normalised version of the loss. 
Another time applying the same architecture and portcol.
We re-run three seeds per candidate.
The results show that scale-sensitivity varies dramatically across the three candidates.

Gamma07 is approximately stable. Its mean Sharpe changes from 0.9156 to 0.9112, which is a very small change relative to seed dispersion.

Gamma10 is scale-sensitive. It drops from 1.0043 to 0.4072 after normalisation. Alpha06 collapses from 0.6895 to minus 0.0161.

So normalisation is not a universal fix. It is a diagnostic — and gamma 0.7 is the only candidate that passes.

---

## Slide 16 - Headline Findings And Cumulative Paths

SKIP IT

---

## Slide 17 - Answering The Three Questions
So now I can give you direct answers to the three questions I posed at the start.

**RQ1 asked: how does loss choice affect prediction-level and portfolio-level performance?**

Loss choice has a major effect on both, but the two effects are not aligned. From Phase 1, MSE produces Sharpe negative 0.46 and a negative cumulative return — the conventional baseline actually destroys value. Switching to GMADL gets you a positive Sharpe of 0.20, even though its R-squared is negative seven billion. Switching to the multiplicative hybrid M1 takes you to Sharpe 0.44. So three different losses, on the same data and same architecture, produce three completely different portfolio outcomes. And R-squared, which is the standard prediction-level metric, does not predict any of this — GMADL has the worst R-squared in the table but still beats MSE on Sharpe. The portfolio trades ranks, not calibrated values, so R-squared and Sharpe are essentially independent metrics in this setting.

**RQ2 asked: which hybrid design gives the best supported Sharpe-stability trade-off?**

The answer is m2-robust gamma 0.7. As you saw on the slide, it has the highest mean Sharpe within the low-CV stability zone, no seed produces a negative Sharpe, and its cumulative return over the test window is the strongest of any low-CV variant. Gamma 1.0 has a slightly higher mean Sharpe, but its seed dispersion is three times worse, so it is not the best joint choice. Alpha 0.6 sits in the same productive design space and confirms the direction is real, but it is below gamma 0.7 on Sharpe. So gamma 0.7 is the best-supported single answer.

**RQ3 asked: are the leading candidates stable under the component-normalisation diagnostic?**

Only one of them is. Gamma 0.7 survives the probe almost untouched — its Sharpe shifts by less than the per-seed noise. Gamma 1.0 drops by roughly half, and alpha 0.6 collapses to essentially zero. So among the three leading candidates, gamma 0.7 is the only one whose signal does not depend on accidental scale imbalance between loss components. This is the strongest evidence that gamma 0.7 is a transferable design choice rather than a dataset-specific artefact.不，你理解错我的意思了。这三个问题的回答应该是基于我这项研究的结果来直接回答。你前面说的很不对应，而且比较啰嗦。

针对这几个问题，建议如下：

1. 第一个问题（Prediction Level）：
   你应该明确表明不同的 Loss Function 确实对结果有显著影响，然后列举一下我们实验中的例子。
2. 第二个问题：
   直接这样处理就行。
3. 第三个问题：
   （此处建议继续补充第三个问题的具体回答逻辑）

---

## Slide 18 - Recommendation, Limitations, And Future Work

My final recommendation has three tiers.

Primary: m2_robust_gamma07. It has mean Sharpe 0.9156, CV 0.1808, and cumulative return plus 27.99 percent. It is the best-supported single choice because it combines high Sharpe, low seed sensitivity, and normalisation-probe stability.

High-return alternative: m2_robust_gamma10. It has the highest mean Sharpe, 1.0043, but CV is 0.5613 and normalised Sharpe drops to 0.4072. I would only choose it if I explicitly accept seed sensitivity.

Stable fallback: imadl_m2_alpha06. It has mean Sharpe 0.6895, CV 0.2443, and cumulative return plus 30.42 percent. It corroborates the hybrid-multiplicative region, but it is not stable under normalisation.

The main limitations are one static 24-month window, three seeds, one feature set, one architecture family.





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
