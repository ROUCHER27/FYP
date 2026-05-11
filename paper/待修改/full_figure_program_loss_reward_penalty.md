# Full Figure Program - Loss Reward/Penalty Focus

Source reviewed: `paper/已修改/chapter2_literature_review.md`, `paper/已修改/chapter3_methodology.md`, `paper/已修改/chapter4_data.md`, `paper/已修改/chapter5_empirical_results_discussion.md`, `paper/已修改/results_source_of_truth.md`, existing figure scripts under `paper/figures/`, and `.kiro/skills/paper-plot-skills/`.

Purpose: reorganise all figures around the main intellectual contribution: the new hybrid loss design and how different loss components reward or penalise prediction signs. Existing figures can be deleted, moved, or redrawn if they do not serve this story.

## Core figure narrative

The report should not look like a collection of phase charts. The figure sequence should tell one story:

```text
1. Why sign correctness matters for a long-short portfolio.
2. How directional losses reward/penalise prediction signs.
3. How hybrid losses combine sign-aware reward/penalty with robust magnitude control.
4. Which hybrid designs survive empirical testing.
5. Why gamma07 is the recommended trade-off rather than just the highest-Sharpe point.
```

This means mechanism figures should move from Chapter 2 into Chapter 3. Chapter 2 should review literature and motivate the gap; Chapter 3 should show the new loss design. Chapter 5 should show empirical evidence only.

## Recommended final figure set

Target: **6 main-text figures**, plus 1 optional appendix figure.

| Final figure | Chapter | Action | Purpose |
|---|---|---|---|
| Figure 3.1 | Chapter 3 | New | Reward/penalty response of directional and hybrid loss components |
| Figure 3.2 | Chapter 3 | New or replace old Figure 2.1 | Loss-surface comparison of directional, additive, and multiplicative hybrids |
| Figure 4.1 | Chapter 4 | Keep, light caption revision | Data coverage and heavy-tail motivation for robust magnitude control |
| Figure 5.1 | Chapter 5 | Replace current Figures 5.1 and 5.2 with one condensed figure | Single-seed evidence: pure losses vs initial hybrid designs |
| Figure 5.2 | Chapter 5 | Redraw current Figure 5.3 | Gamma tuning curve: mean Sharpe vs stability |
| Figure 5.3 | Chapter 5 | New, replaces current Figure 5.4 as the main integrated view | Sharpe-CV frontier across all multi-seed hybrid families |
| Figure 5.4 | Chapter 5 | Replace current Figure 5.5 | Normalisation retention slope plot |
| Appendix Figure B.1 | Appendix B | Optional | Multi-seed cumulative return paths |

Do **not** keep a separate Chapter 2 mechanism figure. That makes the literature review carry methods content and repeats Chapter 3.

## Existing figure decisions

### Current Figure 2.1 - Conceptual loss-function shape comparison

Decision: **delete from Chapter 2**.

Reason: it is not literature review; it explains implemented loss behaviour. Move the useful idea into new Chapter 3 mechanism figures.

Replacement: new Figure 3.1 and Figure 3.2.

### Current Figure 3.1 - Portfolio construction pipeline

Decision: **move to Appendix or keep only if page space allows**.

Reason: it is correct, but it does not highlight the new loss. The portfolio rule is fixed and can be described in prose/table. If the final report is visually crowded, this should be Appendix B rather than a main-text figure.

If kept in main text, renumber it after the new loss-mechanism figures or make it Figure 3.3.

### Current Figure 4.1 - Data coverage and heavy-tail distribution

Decision: **keep**.

Reason: it supports the robust component of the new loss. The heavy-tail panel directly motivates replacing plain MSE with Huber/robust magnitude control.

Caption revision: make the link explicit:

```markdown
The heavy right tail motivates the robust magnitude backbone used in the hybrid losses of Chapter 3.
```

### Current Figure 5.1 - Baseline loss comparison

Decision: **delete or merge into new Figure 5.1**.

Reason: the current version is useful but mostly repeats Table 5.1. It does not strongly show the transition from pure losses to hybrid reward/penalty design.

Replacement: combine baseline losses and A/M hybrid variants into one "pure-to-hybrid performance map" described below.

### Current Figure 5.2 - Phase 2 hybrid variants

Decision: **delete or merge into new Figure 5.1**.

Reason: it is a grouped bar chart of a table. The key message is not every bar; it is that hybrid designs create the productive region and that additive seed-42 peaks are not final robustness evidence.

Replacement: new Figure 5.1.

### Current Figure 5.3 - Gamma refinement

Decision: **keep the idea, redraw the visual**.

Reason: gamma tuning is central to the new loss recommendation. However, a bar chart is less intuitive than a tuning curve. Redraw as a two-panel line plot:

- mean Sharpe vs gamma;
- CV vs gamma;
- highlight gamma07 as the recommended trade-off and gamma10 as high-return but unstable.

### Current Figure 5.4 - IMADL-m2 alpha sweep

Decision: **replace with Sharpe-CV frontier**.

Reason: the alpha sweep alone is too narrow. The integrated section needs one visual that compares all multi-seed hybrid families on the same Sharpe-stability plane.

Replacement: new Figure 5.3.

### Current Figure 5.5 - Normalisation probe

Decision: **replace with retention slope plot**.

Reason: the current grouped bar + dots is acceptable but not visually decisive. A slope plot makes the key story immediate: gamma07 stays flat; gamma10 and alpha06 fall.

Replacement: new Figure 5.4.

## Final figure specifications

## Figure 3.1 - Reward/Penalty Logic of Hybrid Loss Components

**Main purpose:** show how prediction sign is rewarded or penalised.

**Suggested files:**

```text
paper/figures/plot_loss_reward_penalty_response.py
paper/figures/fig3_1_loss_reward_penalty_response.png
```

**Placement:** Chapter 3 §3.3, immediately after the definition of the normalised directional term:

```math
D(y, \hat y) = [1 - \sigma(a y \hat y)] \cdot |y|^b / (\mathbb{E}_{batch}[|y|^b] + \epsilon)
```

**Panel design:**

Use a 2x2 layout:

```text
(a) y = -10%
(b) y = -2%
(c) y = +2%
(d) y = +10%
```

Prediction grid:

```text
yhat in [-20%, +20%]
```

Each panel shows:

- `Directional gate D(y, yhat)`;
- `Multiplicative gate 1 + lambda_dir D(y, yhat)`;
- optional dashed line: `GMADL signed score`.

Shading:

- Sign-correct region: light blue.
- Sign-wrong region: light orange.
- Vertical line: `yhat = 0`, labelled `sign boundary`.

Do not use Chinese words such as `奖励区`, `惩罚区`, or `方向切换点`.

**Visual message:** if the prediction sign is wrong, the directional gate rises and the multiplicative hybrid amplifies the magnitude loss. If the sign is correct, the gate relaxes.

**Caption draft:**

```markdown
**Figure 3.1 - Reward and penalty logic of the hybrid loss components.** The panels show how the normalised directional gate and the multiplicative gate respond as the prediction changes sign for four realised-return values. Blue shading marks sign-correct predictions and orange shading marks sign-wrong predictions. The figure is illustrative and uses the closed-form components in Chapter 3; no training data are used.
```

## Figure 3.2 - Hybrid Loss Surface: Directional Reward Meets Magnitude Control

**Main purpose:** compare different loss constructions on the same true-return / predicted-return plane.

**Suggested files:**

```text
paper/figures/plot_hybrid_loss_surfaces.py
paper/figures/fig3_2_hybrid_loss_surfaces.png
```

**Placement:** Chapter 3 §3.3, after additive and multiplicative formulas.

**Panel design:**

Use a 2x2 heatmap:

```text
(a) GMADL signed directional score
(b) Huber magnitude backbone
(c) Additive hybrid A3
(d) Multiplicative hybrid M1 or M2
```

If exact M2-robust gamma formulas are available from the evidence branch, a fifth/sixth panel can compare `M2-robust gamma07`; otherwise do not fake an exact M2-robust surface. Use the formulas that are present in `Model_Train/losses.py` for exact main-branch surfaces.

Axes:

```text
x-axis: Realised return y
y-axis: Prediction yhat
range: [-15%, +15%]
```

Overlay:

- horizontal line at `yhat = 0`;
- vertical line at `y = 0`;
- dashed diagonal `yhat = y`, labelled `calibration line`;
- small labels: `Sign correct` in quadrants I/III and `Sign wrong` in quadrants II/IV.

**Visual message:** GMADL encodes sign-based reward/penalty; Huber encodes magnitude control; additive hybrids superimpose both; multiplicative hybrids amplify magnitude penalties mainly when the prediction sign is wrong.

**Caption draft:**

```markdown
**Figure 3.2 - Loss-surface comparison of directional and hybrid objectives.** The panels separate the signed directional score, the robust magnitude backbone, and two hybrid combinations. The horizontal and vertical zero lines define sign correctness; the dashed diagonal is the calibration line. The multiplicative hybrid preserves the Huber magnitude backbone while increasing loss in sign-wrong regions.
```

## Figure 4.1 - Data Coverage and Return Tails

**Main purpose:** justify robust magnitude control.

Keep existing figure:

```text
paper/figures/fig4_1_data_coverage.png
```

Light revision only: make the caption explicitly connect heavy tails to robust loss design.

**Caption change:**

```markdown
The heavy-tailed training-era return distribution motivates the robust magnitude component used by the hybrid losses in Chapter 3.
```

## Figure 5.1 - From Pure Losses to Initial Hybrid Designs

**Main purpose:** combine the current baseline and A/M sweep figures into one empirical transition figure.

**Suggested files:**

```text
paper/figures/plot_pure_to_hybrid_summary.py
paper/figures/fig5_1_pure_to_hybrid_summary.png
```

**Replacement:** current `fig5_1_baseline_comparison.png` and `fig5_2_phase15_variants.png`.

**Data source:**

- `paper/已修改/results_source_of_truth.md` sections 1 and 2.
- Same source JSONs currently used for Tables 5.1 and 5.2.

**Panel design:**

Use 2 panels:

```text
(a) Seed-42 Sharpe by loss family
(b) Calibration diagnostic: Avg R2 magnitude vs Sharpe
```

Panel (a):

- x-axis grouped by `Regression`, `Directional`, `Initial hybrid`;
- y-axis annualised Sharpe;
- highlight `hybrid_mul_m1` and `A3`;
- use hatch for hybrid rows.

Panel (b):

- x-axis: `log10(abs(avg R2))`;
- y-axis: Sharpe;
- mark MADL/GMADL as high R2-magnitude but not necessarily high Sharpe;
- mark hybrid M-family as more interpretable calibration region.

**Visual message:** pure regression and pure directional losses are not enough; the productive region starts when sign-aware loss is combined with magnitude control.

**Caption draft:**

```markdown
**Figure 5.1 - From pure losses to initial hybrid designs.** Panel (a) compares seed-42 annualised Sharpe across regression, directional, and initial hybrid losses. Panel (b) shows the decoupling between point-prediction calibration diagnostics and portfolio Sharpe: pure directional losses can produce extreme R2 magnitudes, while the initial hybrid family provides a more interpretable bridge between ranking performance and magnitude control. These rows are single-seed evidence and are used only for design motivation.
```

## Figure 5.2 - Gamma Tuning of the M2-Robust Hybrid Loss

**Main purpose:** show how the new loss is tuned.

**Suggested files:**

```text
paper/figures/plot_gamma_tuning_curve.py
paper/figures/fig5_2_gamma_tuning_curve.png
```

**Replacement:** current `fig5_3_gamma_refinement.png`.

**Data source:**

```text
doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv
```

**Panel design:**

Use a 1x2 line plot:

```text
(a) Mean Sharpe vs gamma
(b) CV vs gamma
```

Highlight:

- `gamma07`: red star, labelled `Recommended trade-off`;
- `gamma10`: orange triangle, labelled `Higher Sharpe, higher CV`;
- optional shaded preferred band around `gamma = 0.05` to `0.10`.

**Visual message:** gamma controls the robustness/magnitude behaviour; gamma07 is selected because it balances high mean Sharpe and low seed sensitivity.

**Caption draft:**

```markdown
**Figure 5.2 - Tuning the M2-robust hybrid loss.** Mean Sharpe and cross-seed CV are plotted across the gamma sweep. `gamma10` has the highest mean Sharpe but much higher seed sensitivity, while `gamma07` gives the best reported Sharpe-stability trade-off.
```

## Figure 5.3 - Sharpe-Stability Frontier Across Hybrid Families

**Main purpose:** integrate all multi-seed hybrid evidence.

**Suggested files:**

```text
paper/figures/plot_sharpe_cv_frontier.py
paper/figures/fig5_3_sharpe_cv_frontier.png
```

**Replacement:** current `fig5_4_imadl_alpha_sweep.png`.

**Data source:**

- gamma rows: `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`;
- integrated rows: values listed in `paper/已修改/results_source_of_truth.md`, or read from:

```bash
git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv
```

**Panel design:**

Scatter plot:

```text
x-axis: Cross-seed CV (lower is better)
y-axis: Mean annualised Sharpe
marker size: Mean cumulative return
color: hybrid family
```

Use a broken x-axis only if including extreme beta rows with CV above 10. A cleaner main-text version should exclude or grey out extreme beta rows and state in the caption that unstable near-zero-mean rows are omitted for readability.

Highlight:

- red star: `gamma07` (`Recommended`);
- orange triangle: `gamma10` (`High-return but unstable`);
- green diamond: `alpha06` (`Stable fallback`).

**Visual message:** the final recommendation is a frontier decision, not a single highest number.

**Caption draft:**

```markdown
**Figure 5.3 - Sharpe-stability frontier across multi-seed hybrid families.** Each point is a three-seed grouped summary. The preferred region is high mean Sharpe with low CV. `gamma07` is highlighted as the recommended trade-off, `gamma10` as a higher-return but less stable alternative, and `alpha06` as a stable fallback from the IMADL-m2 family.
```

## Figure 5.4 - Normalisation Retention of Leading Candidates

**Main purpose:** make the component-scaling probe visually decisive.

**Suggested files:**

```text
paper/figures/plot_normalisation_retention_slope.py
paper/figures/fig5_4_normalisation_retention.png
```

**Replacement:** current `fig5_5_normalisation_probe.png`.

**Data source:** `paper/已修改/results_source_of_truth.md` section 5.

**Panel design:**

Slope plot:

```text
x-axis: Original -> Normalised
y-axis: Mean annualised Sharpe
series: gamma07, gamma10, alpha06
annotation: retention percentage
```

Retention values:

```text
gamma07: about 99.5%
gamma10: about 40.5%
alpha06: about -2.3%
```

**Visual message:** gamma07's performance is not explained away by component scale; gamma10 and alpha06 are scale-sensitive.

**Caption draft:**

```markdown
**Figure 5.4 - Sharpe retention under diagnostic component normalisation.** The slope lines compare original and normalised mean Sharpe for the three leading candidates. `gamma07` retains approximately all of its mean Sharpe, whereas `gamma10` and `alpha06` degrade materially under the same diagnostic probe.
```

## Appendix Figure B.1 - Multi-seed Cumulative Return Paths

**Status:** optional.

**Purpose:** provide time-path evidence without overloading Chapter 5.

**Suggested files:**

```text
paper/figures/plot_seed_cumulative_paths.py
paper/figures/figB_1_seed_cumulative_paths.png
```

**Data source:**

```text
doc/phase2-fix/phase2_2/gamma_refinement/results/m2_robust_gamma07_seed*_cap05/sanity_metrics_m2_robust_gamma07.csv
doc/phase2-fix/phase2_2/gamma_refinement/results/m2_robust_gamma10_seed*_cap05/sanity_metrics_m2_robust_gamma10.csv
```

Use `line_confidence_band`: mean cumulative return line plus seed envelope.

**Visual message:** gamma07's recommendation is supported by a steadier path, not only by a summary table.

## Figure numbering after restructure

If the plan above is adopted, update figure numbering as:

```text
Chapter 2: no figure
Chapter 3: Figure 3.1, Figure 3.2, optional Figure 3.3 if portfolio flow remains
Chapter 4: Figure 4.1
Chapter 5: Figure 5.1, Figure 5.2, Figure 5.3, Figure 5.4
Appendix B: Figure B.1 optional
```

Remove references to old files from main chapter prose:

```text
fig2_1_loss_shapes.png
fig5_1_baseline_comparison.png
fig5_2_phase15_variants.png
fig5_3_gamma_refinement.png
fig5_4_imadl_alpha_sweep.png
fig5_5_normalisation_probe.png
```

These files do not need to be deleted from disk immediately, but they should not remain linked from the final manuscript if replaced.

## Implementation order

Implement figures in this order:

1. `plot_loss_reward_penalty_response.py`
2. `plot_hybrid_loss_surfaces.py`
3. `plot_gamma_tuning_curve.py`
4. `plot_sharpe_cv_frontier.py`
5. `plot_normalisation_retention_slope.py`
6. optional `plot_pure_to_hybrid_summary.py`
7. optional `plot_seed_cumulative_paths.py`

Rationale: mechanism figures first, then the two figures that support the final recommendation. The pure-to-hybrid summary is useful, but Tables 5.1 and 5.2 already carry those numbers, so it can be implemented after the core mechanism/frontier figures.

## Quality checks

Before inserting any new figure:

```bash
python paper/figures/<script>.py
file paper/figures/<output>.png
rg -n "[\\u4e00-\\u9fff]" paper/figures/<script>.py
```

The Chinese-character check should return no figure-visible strings. Bilingual source comments are acceptable only if they do not appear in labels, titles, legends, annotations, or colorbars.

For every empirical figure, the caption must include:

```text
train 1990-01..1994-12
test 1995-01..1996-12
seed set or 3 seeds per row
cap05 if applicable
source artifact path or Appendix B reference
```

For every synthetic mechanism figure, the caption must include:

```text
Illustrative; no training data.
Closed-form components match Chapter 3 / Appendix A.
```

## Final recommendation

Adopt this figure program:

1. Remove method-mechanism graphics from Chapter 2.
2. Make Chapter 3 the visual centre of loss reward/penalty design.
3. Keep Chapter 4's data figure as the motivation for robust magnitude control.
4. Replace table-like Chapter 5 bar charts with figures that show tuning, frontiers, and retention.

This will make the report look less like a phase-result log and more like a coherent final report centred on a designed hybrid loss.
