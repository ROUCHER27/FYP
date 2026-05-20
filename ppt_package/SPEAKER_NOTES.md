# Speaker Notes — Per-Slide Talking Points

## Slide 1: Title
- "Good morning/afternoon. My project investigates loss function design for neural-network-based stock return prediction."

## Slide 2: Outline
- "I'll cover motivation, methodology, data, results across four experimental phases, and my final recommendation."

## Slide 3: Motivation
- "Most ML stock prediction papers vary architectures and features but keep MSE fixed. But a long-short portfolio only cares about RANKING — which stocks are top and bottom. MSE wastes capacity fitting heavy-tailed outliers that the portfolio ignores."
- "Monthly returns have extreme tails — max +2400% in our training data. This is the core mismatch."

## Slide 4: Research Questions
- "Three questions: First, does loss choice actually matter for portfolio performance? Second, can we design a hybrid loss that combines directional accuracy with robust magnitude control? Third, is the winner robust to scale normalisation?"

## Slide 5: Literature
- "Three strands converge: ML for stock prediction (Gu et al. showed deep models beat linear), robust regression (Huber, MedSE for heavy tails), and directional losses (MADL/GMADL that reward correct sign predictions). The gap: no one has compared all three under the same controlled protocol."

## Slide 6: Model & Protocol
- "Simple MLP, 15 features, static 5-year train / 2-year test. The key design: ONLY the loss function varies. Everything else is frozen. This gives clean causal attribution."

## Slide 7: Loss Families
- "We test regression, directional, and two hybrid families. The multiplicative hybrid is the key innovation: it uses the directional penalty as a GATING FACTOR on the Huber backbone. When direction is wrong, loss is amplified. When correct, it reduces to plain Huber."

## Slide 8: Why Multiplicative Works
- "Three properties: (a) asymmetric — penalises wrong direction more; (b) magnitude-aware — larger returns get stronger penalty; (c) implicit variance penalty — dispersed predictions hit more sign-wrong cases, so batch loss grows super-linearly."

## Slide 9: Portfolio Construction
- "Standard long-short: top/bottom 10%, z-score weighted, 5% per-name cap. Same for every loss — only the predictions change."

## Slide 10: Data
- "CRSP US equities, ~11,000 securities. X1 features: cumulative returns and turnover at 5 horizons. No winsorisation — heavy tails are the point of the study."

## Slide 11: Phase 1 Baselines
- "MSE gives negative Sharpe. MedSE barely positive. GMADL has R² of negative 7 billion but positive Sharpe — this proves R² decouples from portfolio performance. hybrid_mul_m1 is the clear winner at Sharpe 0.44."

## Slide 12: Phase 2 Hybrid Sweep
- "Additive peaks at A3 (Sharpe 0.57), multiplicative at M1 (0.44). But M-family has controlled R² — stays in single digits. This interpretability advantage motivates extending M-family into Phase 3."

## Slide 13: Phase 3a γ Refinement
- "Now multi-seed: 3 seeds per variant. gamma07 achieves mean Sharpe 0.92 with CV only 0.18. gamma10 has higher mean Sharpe (1.00) but 3× the CV. No seed of gamma07 produces negative Sharpe."

## Slide 14: γ Tuning Curve
- "The three-panel view shows gamma07 sits at the stability peak on ALL dimensions simultaneously — Sharpe, CV, and portfolio volatility. It's not just minimising one metric at the cost of others."

## Slide 15: Sharpe-Stability Frontier
- "Plotting all variants on one plane: gamma07 is the best point in the preferred region (CV ≤ 0.35). The β-family and adaptive-λ completely fail to reach this region."

## Slide 16: Normalisation Probe
- "Does scale imbalance explain the results? gamma07 barely moves (0.9156 → 0.9112). gamma10 halves. alpha06 collapses to zero. This confirms gamma07's signal is genuine."

## Slide 17: Cumulative Returns
- "Visual evidence: gamma07 shows consistent upward path with narrow seed envelope. MSE and IMADL are flat or negative throughout."

## Slide 18: Final Recommendation
- "Three tiers: Primary is gamma07 (best stability), high-return alternative is gamma10 (with explicit seed-sensitivity caveat), fallback is alpha06 (independent corroboration from a different family)."

## Slide 19: Limitations
- "Single window, 3 seeds, one feature set, gross-of-cost. These are deliberate scope choices for clean internal validity. Future work: rolling window, more seeds, per-component logger."

## Slide 20: Thank You
- "The key takeaway: loss function design is a first-class design variable for portfolio-oriented prediction. A multiplicative hybrid with variance regularisation outperforms both traditional and pure directional losses."
