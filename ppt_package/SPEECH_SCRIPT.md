# Defense Speech Script (逐字稿)
## ~15 minutes | Yirong Yu | FYP Thesis Defense

---

## Slide 1: Title (~20s)

Good morning, everyone. My name is Yirong Yu, student ID 2253235. My project is titled "Multiplicative Directional-Robust Loss for Cross-Sectional Stock-Return Prediction," supervised by Dr. Yi Cao. Thank you for being here today.

---

## Slide 2: Outline (~15s)

I'll walk you through six parts: first the motivation and research questions, then the methodology including our loss function design, followed by data description, empirical results across four experimental phases, and finally my recommendation and conclusions.

---

## Slide 3: Motivation (~50s)

Let me start with the core problem. In machine learning for stock prediction, most studies vary the model architecture or feature set, but they almost always keep the loss function fixed at Mean Squared Error. However, the downstream task — constructing a long-short portfolio — depends only on the cross-sectional ranking of predictions, not on their calibrated magnitudes.

This creates a fundamental mismatch. MSE treats all prediction errors equally, but a portfolio doesn't care whether you predict a return of 5% or 50% — it only cares whether that stock ranks in the top or bottom bucket. Worse, monthly stock returns have extremely heavy tails — in our training data, the maximum single-month return is plus 2400 percent. MSE wastes model capacity fitting these outliers that the portfolio construction completely ignores.

This mismatch is the motivation for our study: can we design a better loss function?

---

## Slide 4: Research Questions (~30s)

We pose three research questions. First: does loss function choice actually affect portfolio-level performance when everything else is held fixed? Second: can we design a hybrid loss that combines directional accuracy with robust magnitude control, and which parameterisation gives the best Sharpe-to-stability trade-off? Third: is the winning loss function approximately stable under a diagnostic component normalisation, ruling out scale artifacts?

---

## Slide 5: Literature (~45s)

Our work sits at the intersection of three literature strands. First, machine learning for stock prediction — Gu, Kelly, and Xiu in 2020 showed that deep neural networks systematically outperform linear factor models. Second, robust regression — dating back to Huber in 1964 — which limits the influence of heavy-tailed outliers. Third, directional losses like MADL and GMADL from Michankow and colleagues in 2024, which explicitly reward correct sign predictions weighted by return magnitude.

The gap we address is simple: no prior work has compared regression, robust, directional, and hybrid losses under the same controlled experimental protocol. That's exactly what we do.

---

## Slide 6: Model & Protocol (~40s)

Our experimental design is deliberately simple. We use a multi-layer perceptron with three hidden layers of width 64, 32, and 16, consuming 15 input features. The training window is static: five years from 1990 to 1994, with a 24-month out-of-sample test from 1995 to 1996.

The critical design choice is single-factor control. Within each comparison table, the data, features, architecture, training protocol, and portfolio construction are all frozen. The loss function is the only variable that changes. This gives us clean causal attribution of any performance difference directly to the loss function.

---

## Slide 7: Loss Function Families (~50s)

We evaluate three families of loss functions plus our proposed hybrids. The regression family includes MSE and Median Squared Error. The directional family includes MADL, GMADL, and IMADL, which reward correct sign predictions.

Our key contribution is the hybrid family, which comes in two forms. The additive hybrid simply sums a directional penalty and a Huber magnitude term. The multiplicative hybrid — which is our main innovation — uses the directional penalty as a gating factor on the Huber backbone. The formula is: loss equals one plus lambda times the directional gate, all multiplied by the Huber term.

As you can see in this figure, when the prediction direction is wrong, the gate amplifies the loss. When the direction is correct, the gate relaxes to zero and the loss reduces to plain Huber. This asymmetry is the key mechanism.

---

## Slide 8: Why Multiplicative Works (~35s)

The multiplicative hybrid has three desirable properties shown in this figure. Panel A shows directional asymmetry — wrong-direction predictions are penalised much more heavily than correct ones. Panel B shows magnitude awareness — the penalty grows with the size of the realised return, focusing training signal on economically important moves. Panel C shows an implicit variance penalty — when predictions are dispersed, more of them fall into the sign-wrong region, causing batch loss to grow super-linearly. This last property motivates adding an explicit variance regulariser, which leads to our M2-robust gamma family.

---

## Slide 9: Portfolio Construction (~25s)

The portfolio construction is standard and identical for every loss function. We take the prediction vector, select the top and bottom 10 percent as long and short buckets, apply within-bucket z-score weighting clipped to plus-minus three, enforce a 5 percent per-name cap through simplex projection, and compute the long-short return. Only the predictions change across experiments; the construction pipeline is fixed.

---

## Slide 10: Data (~30s)

We use a CRSP-style US equity monthly panel covering about 11,000 unique securities in the training era. The feature set X1 consists of cumulative returns and cumulative turnover at five horizons — 1, 3, 6, 9, and 12 months — plus five base panel columns, giving 15 input dimensions total. Importantly, we apply no winsorisation. The heavy tails are the entire point of the study — they're what motivates robust loss design.

---

## Slide 11: Phase 1 — Baselines (~45s)

Now to results. Phase 1 compares seven baseline losses at seed 42. The key findings: MSE produces a negative Sharpe of minus 0.46 — it actually loses money. MedSE is barely positive at 0.09. Among directional losses, GMADL achieves a positive Sharpe of 0.20, but its average R-squared is negative 7 billion. This is a crucial finding — it proves that R-squared completely decouples from portfolio performance. The portfolio uses ranks, not calibrated values.

The clear winner among baselines is hybrid multiplicative M1, with Sharpe 0.44 and the lowest monthly volatility. This motivates extending the multiplicative family in the next phase.

---

## Slide 12: Phase 2 — Hybrid Sweep (~35s)

Phase 2 sweeps nine parameterised hybrid variants — five additive and four multiplicative. The additive family peaks at A3 with Sharpe 0.57, while the multiplicative peaks at M1 with 0.44. However, the M-family maintains R-squared in single digits, meaning its predictions stay on a reasonable scale. This interpretability advantage, combined with the gating mechanism I described earlier, motivates extending the M-family into multi-seed evaluation rather than the additive family.

Importantly, these are single-seed results. They motivate the design direction but cannot support robustness claims.

---

## Slide 13: Phase 3a — Gamma Refinement (~50s)

Phase 3 is where we get serious about robustness. We add an explicit prediction-variance penalty to the M2 loss: L equals L-M2 plus gamma times the variance of predictions. We scan five gamma values, each evaluated across three random seeds.

The results are striking. Gamma 0.7 achieves mean Sharpe 0.92 with a coefficient of variation of only 0.18 — meaning its cross-seed variability is less than one-fifth of its mean. Gamma 1.0 has a slightly higher mean Sharpe at 1.00, but its CV is 0.56 — three times worse. And critically, no seed of gamma 0.7 produces a negative Sharpe. This is the stability we're looking for.

---

## Slide 14: Gamma Tuning Curve (~30s)

This three-panel figure makes the trade-off explicit. Panel A shows mean Sharpe versus gamma — it peaks around 0.7 to 1.0. Panel B shows CV versus gamma — it has a clear minimum at 0.7. Panel C shows portfolio volatility versus gamma — again minimised at 0.7. The key point is that gamma 0.7 sits at the stability peak on all three dimensions simultaneously. It's not optimising one metric at the expense of others.

---

## Slide 15: Sharpe-Stability Frontier (~35s)

This figure places all multi-seed variants from every family on a single Sharpe versus CV plane. The blue shaded region marks our preferred zone: CV less than or equal to 0.35. Gamma 0.7, marked with the red star, achieves the highest Sharpe within this preferred region. The IMADL-GMADL beta family and adaptive lambda family completely fail to reach this region — their CVs are too high. Alpha 0.6, the green diamond, provides independent corroboration from a different parameterisation family.

---

## Slide 16: Normalisation Probe (~40s)

A natural concern is whether our results are driven by scale imbalance between the directional and magnitude components. The normalisation probe tests this by equalising component scales and re-running.

The result: gamma 0.7 barely moves — from 0.9156 to 0.9112, a change smaller than the per-seed dispersion. But gamma 1.0 halves to 0.41, and alpha 0.6 collapses to essentially zero. This confirms that gamma 0.7's signal is genuine and not a scale artifact. The other two candidates' apparent performance was partially driven by component imbalance.

---

## Slide 17: Cumulative Returns (~20s)

This figure provides visual evidence. The left panel shows Phase 1 baselines over time — MSE and IMADL produce negative or flat paths. The right panel shows the gamma family across seeds — gamma 0.7 in red produces the most consistent upward path with a narrow seed envelope, while gamma 1.0 in orange reaches higher peaks but with much wider dispersion.

---

## Slide 18: Final Recommendation (~40s)

My final recommendation has three tiers. Primary: m2-robust gamma 0.7 — mean Sharpe 0.92, CV 0.18, cumulative return plus 28 percent, stable under normalisation. This is the best-supported single choice.

High-return alternative: gamma 1.0 — higher mean Sharpe at 1.00, but with explicit caveat that its CV is three times worse and it's sensitive to normalisation. Only appropriate if you knowingly accept seed sensitivity.

Stable fallback: IMADL-m2 alpha 0.6 — mean Sharpe 0.69, CV 0.24, highest cumulative return at plus 30 percent. This provides independent corroboration that the multiplicative-hybrid region is the productive one.

---

## Slide 19: Limitations (~30s)

I want to be transparent about scope. We use a single 24-month evaluation window — no rolling retraining. Three seeds per row — sufficient for order-of-magnitude differentiation but not second-decimal precision. One feature set, one architecture, and gross-of-cost portfolio. These are deliberate choices for clean internal validity. Future work should extend to rolling windows across market regimes, at least 10 seeds, a per-component loss logger, and feature-set sensitivity tests.

---

## Slide 20: Thank You (~15s)

To summarise in one sentence: loss function design is a first-class design variable for portfolio-oriented prediction, and a multiplicative hybrid with variance regularisation outperforms both traditional regression losses and pure directional losses under controlled conditions. Thank you. I'm happy to take questions.

---

*Total estimated time: ~14 minutes*
