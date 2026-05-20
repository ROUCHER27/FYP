# FYP Defense 逐字稿 (Rubric-Aligned Final Version)

**评分标准对照**:
- Delivery 45%: 幻灯片准备(5%), 视觉辅助(5%), 语音清晰(10%), 信息充分(10%), 组织结构(5%), 节奏时间(5%), 互动眼神(5%)
- Contents 55%: 项目概述(5%), 背景理论文献(5%), 清晰正确简洁(10%), 证明与连贯性(10%), 数学内容与洞察(10%), 结果解读(5%), 回答问题(10%)

**总时长**: 15–16 分钟 | **语速**: ~130 words/min (清晰不赶)

---

## Slide 1: Title (~20s)

Good morning, everyone. My name is Yirong Yu, student ID 2253235. My project is titled "Multiplicative Directional-Robust Loss for Cross-Sectional Stock-Return Prediction," supervised by Dr. Yi Cao.

---

## Slide 2: Outline (~15s)

My presentation covers six parts: motivation and research questions, literature background, methodology — including our novel loss function design, data description, empirical results across four experimental phases, and finally my recommendation with limitations.

---

## Slide 3: Motivation — The Loss-Portfolio Mismatch (~50s)

【评分重点: Background/relevance + Mathematical insight】

Let me start with the core problem. In machine learning for stock prediction, most studies vary the model architecture or feature set, but keep the loss function fixed at Mean Squared Error.

However, the downstream task — constructing a long-short portfolio — depends only on the cross-sectional *ranking* of predictions, not on their calibrated magnitudes. Mathematically, if we denote predictions as ŷ, the portfolio selects the top and bottom deciles by rank. Whether ŷ equals 0.05 or 0.50 is irrelevant — only the ordering matters.

This creates a fundamental mismatch. Worse, monthly stock returns are heavy-tailed — in our training data, the standard deviation is 19.5% but the maximum is plus 2400%. MSE assigns quadratic weight to these outliers, wasting model capacity on observations the portfolio ignores entirely.

This mismatch motivates our central question: can we design a loss function that directly serves the portfolio objective?

---

## Slide 4: Research Questions (~30s)

【评分重点: Outline of the project】

We pose three research questions.

RQ1: Does loss function choice affect portfolio-level performance when all other factors are controlled?

RQ2: Can we design a hybrid loss combining directional accuracy with robust magnitude control, and which parameterisation achieves the best Sharpe-to-stability trade-off?

RQ3: Is the winning candidate approximately stable under diagnostic component normalisation — ruling out scale artifacts?

---

## Slide 5: Literature — Three Strands (~45s)

【评分重点: Background, relevant theory, literature reference】

Our work sits at the intersection of three literature strands.

First, machine learning for stock prediction. Gu, Kelly, and Xiu (2020) showed that deep neural networks systematically outperform linear factor models on a panel of 94 characteristics. Daniel and Moskowitz (2016) and Medhat and Schmeling (2021) provide the momentum and turnover features we adopt.

Second, robust regression. Huber (1964) introduced the Huber loss that transitions from quadratic to linear penalisation beyond a threshold δ, limiting outlier influence. Median-based estimators provide even stronger robustness.

Third, directional losses. Michankow, Slepaczuk, and Bielak (2024) proposed MADL and GMADL, which reward correct sign predictions weighted by return magnitude.

The gap: no prior work has compared regression, robust, directional, and hybrid losses under the same controlled protocol. That is precisely what we do.

---

## Slide 6: Model & Experimental Protocol (~40s)

【评分重点: Clarity, correctness, conciseness】

Our experimental design is deliberately simple to ensure clean causal attribution.

The model is a multi-layer perceptron: input dimension 15, hidden layers [64, 32, 16], ReLU activation, dropout 0.2. Training uses Adam with learning rate 0.001, batch size 1024, for 20 epochs.

The protocol is a static split: train on 1990-01 through 1994-12 — five years — and test out-of-sample on 1995-01 through 1996-12 — twenty-four months. No retraining occurs during the test period.

The critical design principle: within each comparison table, data, features, architecture, optimizer, and portfolio construction are all frozen. The loss function is the *only* varying factor. This single-factor control gives direct causal attribution of performance differences to the loss.

---

## Slide 7: Loss Function Families (~55s)

【评分重点: Mathematical content and insightfulness】

We evaluate three families plus our proposed hybrids.

Regression: MSE with mean reduction, and MedSE — the median of squared residuals — which is robust to up to 50% contamination.

Directional: MADL uses a tanh gate weighted by |y|; GMADL uses sigmoid with |y|^b weighting, amplifying large-return sensitivity.

Our contribution is the hybrid family. The additive form sums a directional penalty D and a Huber term H:

L_add = λ_dir · D(y, ŷ) + λ_hub · H(y − ŷ)

The multiplicative form — our key innovation — uses D as a *gating factor*:

L_mul = (1 + λ · D(y, ŷ)) · H_δ(y − ŷ)

where D is a batch-normalised directional penalty: D = [1 − σ(a·y·ŷ)] · |y|^b / E[|y|^b].

When the prediction sign is correct, D approaches zero and the loss reduces to plain Huber. When the sign is wrong, the loss is amplified by the factor (1 + λD). This asymmetric gating is the core mechanism.

---

## Slide 8: Why Multiplicative Works — Three Properties (~35s)

【评分重点: Major proofs and coherence】

This figure demonstrates three properties that distinguish our multiplicative hybrid.

Panel A — directional asymmetry: for a fixed realised return y = +5%, the loss is substantially higher when ŷ is negative than when ŷ is positive. MSE, by contrast, is symmetric.

Panel B — magnitude awareness: at a fixed wrong-sign prediction ŷ = −3%, the hybrid penalty grows faster than Huber as |y| increases, because the batch-normalised |y|^b weight amplifies the gate.

Panel C — implicit variance penalty: as prediction dispersion increases, more predictions fall into sign-wrong regions, causing expected batch loss to grow super-linearly. This motivates adding an *explicit* variance regulariser: L = L_M2 + γ · Var(ŷ), which defines our M2-robust gamma family.

---

## Slide 9: Portfolio Construction (~25s)

【评分重点: Coherence between sections】

The portfolio construction is identical for every loss function — only the prediction vector changes.

Six steps: given predictions ŷ_t, select the top and bottom 10% as long and short buckets. Within each bucket, compute z-scores clipped to [−3, +3]. Apply sign-consistent weighting — long bucket uses max(z, 0), short uses max(−z, 0). Project onto the capped simplex with a 5% per-name maximum. Compute the long-short return as r_long minus r_short.

This ensures any performance difference is attributable solely to the loss function's effect on the prediction vector.

---

## Slide 10: Data (~30s)

【评分重点: Sufficient information provided】

We use a CRSP-style US equity monthly panel. The training era contains 449,018 observations across 10,987 unique securities over 60 months.

The feature set X1 comprises 15 columns: cumulative returns and cumulative turnover at horizons 1, 3, 6, 9, and 12 months, plus five base panel columns — RET, VOL, SHROUT, r, and turnover.

Importantly, we apply no winsorisation. The training-era return distribution has mean 1.0%, standard deviation 19.5%, and a maximum of +2400%. These heavy tails are precisely what motivates robust loss design — removing them would obscure the effect we aim to measure.

---

## Slide 11: Phase 1 — Baseline Loss Comparison (~50s)

【评分重点: Interpretation of results + Mathematical insight】

Now to results. Phase 1 compares seven baseline losses at seed 42 over the 24-month test window.

Key findings from the table: MSE produces an annualised Sharpe of −0.4643 and cumulative return of −11.25%. It actually destroys value. MedSE is barely positive — Sharpe 0.0932, cumulative return +0.60%.

Among directional losses, GMADL achieves Sharpe +0.2025, but its average R² is −7.02 × 10⁹. This is a crucial finding: it proves that R² completely decouples from portfolio performance. The portfolio uses cross-sectional ranks, not calibrated point predictions. R² measures calibration — it is a diagnostic, not a performance metric in this setting.

The clear winner is hybrid_mul_m1 — Sharpe 0.4435, cumulative return +5.09%, and the lowest monthly long-short standard deviation at 0.0173. This demonstrates that combining directional and magnitude terms produces a more stable signal than either alone, motivating the hybrid sweep in Phase 2.

---

## Slide 12: Phase 2 — Hybrid A/M Variant Sweep (~35s)

【评分重点: Coherence between sections/results】

Phase 2 extends the hybrid family into nine parameterised variants — five additive (A1–A5) and four multiplicative (M1–M4).

The additive family peaks at A3: Sharpe 0.5738, cumulative return +8.13%. The multiplicative family peaks at M1: Sharpe 0.4435. However, the M-family maintains R² in single digits — M1 has R² = −4.79 versus A3's −1383.64. This means M-family predictions remain on a physically interpretable scale.

This interpretability advantage, combined with the gating mechanism, motivates extending the multiplicative family — not the additive — into multi-seed evaluation. Importantly, these are single-seed results at seed 42. They motivate the design direction but cannot support robustness claims. For that, we need Phase 3.

---

## Slide 13: Phase 3a — Multi-Seed γ Refinement (~50s)

【评分重点: Mathematical content + Major proofs】

Phase 3 introduces the M2-robust gamma family. The loss is:

L = L_M2(y, ŷ) + γ · Var(ŷ)

where L_M2 is the multiplicative hybrid with λ_dir = 2.0, and Var(ŷ) is the within-batch prediction variance. We scan γ ∈ {0.3, 0.5, 0.7, 1.0, 1.5}, each evaluated across three random seeds.

Results: gamma 0.7 achieves mean Sharpe 0.9156 with coefficient of variation 0.1808 and mean cumulative return +27.99%. Gamma 1.0 has slightly higher mean Sharpe at 1.0043, but its CV is 0.5613 — three times worse.

Critically, no seed of gamma 0.7 produces a negative Sharpe. Its minimum per-seed Sharpe is 0.7532. This stability is the key differentiator.

The relationship between γ and stability is non-monotone: γ = 0.3 under-regularises (CV = 1.057), γ = 1.5 over-compresses the signal (CV = 0.456). The stability peak at γ = 0.7 is an internal optimum — not an endpoint of a monotone trend.

---

## Slide 14: γ Tuning Curve (~30s)

【评分重点: Clarity + Visual aids】

This three-panel figure makes the trade-off explicit.

Panel A: mean Sharpe versus γ — peaks in the 0.7–1.0 range. Panel B: coefficient of variation versus γ — clear minimum at 0.7. Panel C: mean monthly long-short return standard deviation versus γ — also minimised at 0.7.

The key insight: gamma 0.7 sits at the stability peak on *all three dimensions simultaneously*. It does not sacrifice Sharpe for stability, nor stability for low volatility. This joint optimality is what makes it the primary recommendation.

---

## Slide 15: Sharpe-Stability Frontier (~35s)

【评分重点: Interpretation of results】

This figure places all multi-seed variants from every family on a single Sharpe-versus-CV plane. The blue region marks our preferred zone: CV ≤ 0.35.

Gamma 0.7 — the red star — achieves the highest Sharpe within this preferred region. The IMADL-GMADL beta family produces CVs from 4.0 to 139.5 — completely unstable. The adaptive-lambda family also fails to reach the preferred region.

Alpha 0.6 — the green diamond — provides independent corroboration from the IMADL-m2 family: mean Sharpe 0.6895, CV 0.2443. It confirms that the multiplicative-hybrid region is productive across different parameterisations.

---

## Slide 16: Normalisation Probe (~40s)

【评分重点: Rigor/correctness + Mathematical insight】

A natural concern: the directional and magnitude components operate on different scales — the estimated ratio is approximately 113:1. Could the observed Sharpe differences simply reflect scale imbalance rather than genuine loss-family effects?

The normalisation probe equalises component scales and re-runs three seeds. Results:

Gamma 0.7: mean Sharpe moves from 0.9156 to 0.9112 — a change of only 0.004, smaller than the per-seed dispersion. The signal is genuine.

Gamma 1.0: drops from 1.0043 to 0.4072 — halved. Its apparent advantage was partially a scale artifact.

Alpha 0.6: collapses from 0.6895 to −0.0161 — essentially zero.

This confirms that gamma 0.7 is the only candidate whose performance is approximately scale-invariant. Its signal is robust to the diagnostic normalisation.

---

## Slide 17: Cumulative Return Paths (~20s)

【评分重点: Visual aids + Interpretation】

This figure provides visual evidence of portfolio performance over time.

Left panel: Phase 1 baselines — MSE and IMADL produce negative or flat cumulative paths throughout the 24-month window. Hybrid M1 accumulates steadily.

Right panel: the gamma family across three seeds — gamma 0.7 in red shows the most consistent upward path with a narrow seed envelope. Gamma 1.0 in orange reaches higher peaks in its best seed but shows substantially wider dispersion between seeds.

---

## Slide 18: Final Recommendation (~40s)

【评分重点: Interpretation of results and implications】

My final recommendation has three tiers, each with explicit scope.

Primary: m2_robust_gamma07. Mean Sharpe 0.9156, CV 0.1808, mean cumulative return +27.99% over 24 months. Approximately stable under normalisation. No seed produces a negative Sharpe. This is the best-supported single choice.

High-return alternative: m2_robust_gamma10. Mean Sharpe 1.0043 — the highest in the study — but CV 0.5613 and normalisation-sensitive. Appropriate only if one knowingly accepts seed sensitivity for higher best-case performance.

Stable fallback: imadl_m2_alpha06. Mean Sharpe 0.6895, CV 0.2443, highest cumulative return at +30.42%. This provides independent corroboration from a different parameterisation family that the multiplicative-hybrid region is productive.

---

## Slide 19: Limitations & Future Work (~30s)

【评分重点: Rigor — honest scope acknowledgment】

I want to be transparent about the boundaries of these claims.

First, a single 24-month evaluation window — we do not sample across market regimes. Second, three seeds per row — sufficient for order-of-magnitude CV differentiation but not second-decimal precision. Third, one feature set and one frozen architecture — feature and architecture sensitivity are untested. Fourth, gross-of-cost portfolio — no transaction or borrowing costs.

These are deliberate scope choices for clean internal validity. Priority future work: rolling-window evaluation across regimes, at least 10 seeds for tighter confidence intervals, a per-component loss logger to replace estimated scale ratios, and feature-set sensitivity tests.

---

## Slide 20: Thank You (~15s)

To summarise: loss function design is a first-class design variable for portfolio-oriented prediction. A multiplicative hybrid with variance regularisation — specifically gamma 0.7 — outperforms both traditional regression losses and pure directional losses under controlled conditions, with verified multi-seed stability and normalisation robustness.

Thank you. I'm happy to take your questions.

---

*Total estimated time: ~15 minutes*

---

## 附录：Q&A 准备要点 (Backup for 回答问题 10%)

**Q: Why only 3 seeds?**
A: Three seeds are sufficient for order-of-magnitude CV differentiation (0.18 vs 0.56) but not for second-decimal precision. This is explicitly stated as a claim boundary. Future work targets ≥10 seeds.

**Q: Why a static window instead of rolling?**
A: Static window gives clean single-factor control — we can attribute differences purely to the loss. Rolling window introduces confounds from regime changes and retraining dynamics. It's the natural next step but outside our current claim boundary.

**Q: Why not test other architectures?**
A: Our research question is about loss function design, not architecture search. Fixing the architecture is the controlled-experiment design choice. A parallel study varying architecture while holding loss fixed would answer a different question.

**Q: What about transaction costs?**
A: The portfolio is gross-of-cost. This is a common simplification in loss-function comparison studies because costs affect all losses equally under the same turnover profile. However, different losses may induce different turnover — this is acknowledged as a limitation.

**Q: Why does R² decouple from Sharpe?**
A: R² measures point-prediction calibration. The portfolio uses only cross-sectional ranks. A model can produce predictions on a completely wrong scale (hence negative R²) while still ranking stocks correctly (hence positive Sharpe). This is empirically demonstrated by GMADL: R² = −7×10⁹ but Sharpe = +0.20.

**Q: How do you know gamma 0.7 isn't overfitting to this specific window?**
A: We cannot fully rule this out with a single window — this is our primary limitation. However, three pieces of evidence support genuine signal: (1) multi-seed stability (CV 0.18), (2) normalisation robustness, and (3) independent corroboration from the alpha family. A rolling-window test would provide stronger evidence.
