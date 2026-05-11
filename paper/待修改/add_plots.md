# Add Plots Plan

Source reviewed: `paper/已修改/chapter2_literature_review.md`, `paper/已修改/chapter3_methodology.md`, `paper/已修改/chapter5_empirical_results_discussion.md`, `paper/已修改/results_source_of_truth.md`, existing scripts in `paper/figures/`, and `.kiro/skills/paper-plot-skills/`.

Purpose: design additional publication-quality figures that show how the hybrid losses behave and why the final recommendation is credible. This plan references the interim-report MADL/GMADL scoring figures, but all new plot text must be English only.

## Current figure gap

The current revised draft already has:

- `Figure 2.1`: conceptual loss-function shape comparison.
- `Figure 5.1`: baseline cumulative return and Sharpe.
- `Figure 5.2`: additive/multiplicative hybrid single-seed sweep.
- `Figure 5.3`: M2-robust gamma multi-seed Sharpe and CV.
- `Figure 5.4`: IMADL-m2 alpha sweep vs gamma references.
- `Figure 5.5`: normalisation probe.

The missing visual layer is not another table-like bar chart. The report would benefit from:

1. a clearer mechanism figure showing how hybrid losses judge sign-correct and sign-wrong predictions;
2. a 2D loss-surface figure inspired by the interim-report heatmap, but with correct English annotations and cleaner boundaries;
3. one integrated results figure showing the full Sharpe-stability frontier across hybrid families.

Recommended main-text additions: **Plot 1** and **Plot 3** below. Plot 2 is useful if Chapter 3 needs stronger method explanation; otherwise it can replace or expand the current Figure 2.1. Plot 4 is optional if page space allows.

## Global plotting rules

Use the existing project style in `paper/figures/_style.py`:

- `dpi=300`, white background, no page header inside the image.
- Serif font via STIX / DejaVu Serif unless a chosen paper-plot template explicitly uses sans-serif.
- No Chinese text inside figure title, axis labels, legend, annotations, or colorbar.
- Do not use old report labels such as "Progress in Semester 1".
- Do not use branch names in visible plot labels. Captions may cite source paths.
- Do not introduce new numbers unless they exist in `paper/已修改/results_source_of_truth.md` or are computed directly from verified CSV/JSON artifacts.
- Prefer color plus marker/line style, not color alone. Green/red reward/penalty colors from the interim report are acceptable only if accompanied by English labels and not used as the sole distinction.

Suggested English annotation vocabulary:

```text
Sign-correct region
Sign-wrong region
Directional gate
Magnitude backbone
Hybrid loss
Lower is better
Recommended
High-return but unstable
Stable fallback
```

## Plot 1 - Hybrid Directional Response Panels

**Status:** recommended main-text mechanism figure.

**Purpose:** reproduce the useful idea from the interim-report 2x2 MADL/GMADL line plot, but adapt it to the final hybrid-loss story. The figure should show how the directional term and multiplicative gate respond when the realised return is negative/positive and small/large.

**Suggested filename:**

```text
paper/figures/plot_hybrid_direction_response.py
paper/figures/fig3_2_hybrid_direction_response.png
```

**Placement:** Chapter 3 §3.3, after the definition of `D(y, yhat)` and before the additive/multiplicative formulas. If Chapter 3 is already too figure-heavy, place it in Appendix A and refer to it once from Chapter 3.

**Paper-plot style:** `line_training_curve` adapted to a 2x2 scientific response plot. Four-sided axes, outward ticks, no Chinese labels.

**Panel layout:**

```text
(a) Realised return y = -10%
(b) Realised return y = -2%
(c) Realised return y = +2%
(d) Realised return y = +10%
```

Use realistic monthly-return examples instead of the interim report's `-0.8` and `+0.8`, which look too extreme for the main text. Suggested prediction grid:

```text
yhat in [-0.20, 0.20]
```

**Curves to draw in each panel:**

1. `Directional gate D(y, yhat)` from Chapter 3.
2. `Multiplicative gate 1 + lambda_dir * D`, using `lambda_dir = 2` for `hybrid_mul_m1` or `lambda_dir = 5` for `hybrid_mul_m2`; choose one and state it clearly.
3. Optional thin dashed reference: `GMADL signed score`, labelled as `GMADL signed score` and described as "negative values are rewards".

**Region shading:**

- Light blue: `Sign-correct region`.
- Light orange: `Sign-wrong region`.
- Vertical line at `yhat = 0`, labelled `sign boundary`.

Do not copy the interim figure's Chinese labels `方向切换点`, `奖励区`, or `惩罚区`.

**Axis labels and title text:**

```text
x-axis: Prediction yhat
y-axis: Relative response
legend: Directional gate; Multiplicative gate; GMADL signed score
```

**Draft caption:**

```markdown
**Figure 3.2 - Directional response of hybrid loss components.** The panels show how the normalised directional gate and the multiplicative gate respond as the prediction changes sign for four realised-return values. The sign-correct region is shaded in blue and the sign-wrong region in orange. The figure is illustrative and uses the closed-form components in Chapter 3; it does not use training data.
```

**Why it helps:** this figure explains the "scoring logic" behind the hybrid functions without turning Chapter 2 into a methods section.

## Plot 2 - Hybrid Loss Surface Map

**Status:** recommended if replacing or expanding current Figure 2.1; otherwise optional.

**Purpose:** modernise the interim-report heatmap. The old heatmap is conceptually useful because it shows the true-return / predicted-return plane, but it uses Chinese labels and diagonal boundaries that can confuse sign correctness. The new version should use English labels and the correct sign regions.

**Suggested filename:**

```text
paper/figures/plot_hybrid_loss_surface.py
paper/figures/fig3_3_hybrid_loss_surface.png
```

**Placement:** Chapter 3 §3.3 or Appendix A. Prefer Chapter 3 if deleting Chapter 2 §2.5, because the figure then supports methodology rather than literature review.

**Paper-plot style:** custom heatmap following `plot-from-image` proportions, with `paper/figures/_style.py` font settings. If using a template analogy, it is closest to an adapted loss-surface version of `line_loss_with_inset`, not a bar/line chart.

**Panel layout:**

```text
(a) GMADL signed directional score
(b) Huber magnitude backbone
(c) Directional gate D(y, yhat)
(d) Multiplicative hybrid loss
```

**Axes:**

```text
x-axis: Realised return y
y-axis: Prediction yhat
range: [-0.15, 0.15] for both axes
```

**Boundaries to overlay:**

- Solid vertical line at `y = 0`.
- Solid horizontal line at `yhat = 0`.
- Thin dashed diagonal `yhat = y`, labelled `calibration line`.

The sign-correct regions are quadrants I and III, where `y * yhat > 0`. The sign-wrong regions are quadrants II and IV. Do not use only diagonal lines as the sign boundary.

**Color design:**

- For signed GMADL score: diverging colormap centred at zero; colorbar label `GMADL signed score`.
- For positive loss/gate panels: sequential colormap (`viridis` or `magma`); colorbar labels `Magnitude loss`, `Directional gate`, and `Hybrid loss`.
- Avoid red/green-only reward/penalty encoding unless paired with text labels.

**English annotations:**

```text
Sign correct
Sign wrong
Calibration line
```

**Draft caption:**

```markdown
**Figure 3.3 - Loss-surface view of the hybrid construction.** The four panels separate the signed directional score, the Huber magnitude backbone, the normalised directional gate, and their multiplicative combination. The horizontal and vertical zero lines define sign correctness; the dashed diagonal is the calibration line. The figure is illustrative and is evaluated on a synthetic return grid.
```

**Why it helps:** this gives a clean English replacement for the interim heatmap and makes the hybrid design visually intuitive.

## Plot 3 - Sharpe-CV Frontier Across Hybrid Families

**Status:** strongest recommended new Chapter 5 figure.

**Purpose:** show the main empirical conclusion in one figure: `gamma07` is not the highest-Sharpe point, but it is the best Sharpe-stability trade-off; `gamma10` has higher Sharpe but larger seed sensitivity; `alpha06` is a stable fallback.

**Suggested filename:**

```text
paper/figures/plot_sharpe_cv_frontier.py
paper/figures/fig5_6_sharpe_cv_frontier.png
```

**Placement:** Chapter 5 after Table 5.4 / Figure 5.4, before the normalisation probe. It can replace some explanatory prose in §5.5.

**Paper-plot style:** `scatter_broken_axis` if including the beta family, because CV ranges from about `0.18` to above `100`. If excluding extreme beta rows, a normal scatter plot with `xlim=(0, 2)` is enough.

**Data source:**

- `paper/已修改/results_source_of_truth.md` sections 3 and 4.
- Gamma rows from `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`.
- Integrated rows from `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`, or the values already listed in the source-of-truth file.

**Visual mapping:**

```text
x-axis: Cross-seed CV (lower is better)
y-axis: Mean annualised Sharpe
marker size: Mean cumulative return
color: loss family
star marker: m2_robust_gamma07
triangle marker: m2_robust_gamma10
diamond marker: imadl_m2_alpha06
```

**Family colors:**

```text
M2-robust gamma: deep navy
IMADL-m2 alpha: green
Adaptive lambda: grey
IMADL-GMADL beta: muted orange
Recommended point: deep red star with black edge
```

**Annotations:**

```text
Recommended: gamma07
High-return but unstable: gamma10
Stable fallback: alpha06
Preferred region
```

**Draft caption:**

```markdown
**Figure 5.6 - Sharpe-stability frontier across multi-seed hybrid variants.** Each point is a three-seed grouped summary; the x-axis reports the coefficient of variation of Sharpe and the y-axis reports mean annualised Sharpe. Marker size represents mean cumulative return. The preferred region is high Sharpe with low CV. `m2_robust_gamma07` is highlighted as the recommended trade-off, while `m2_robust_gamma10` is shown as a higher-Sharpe but less stable alternative and `imadl_m2_alpha06` as the stable fallback.
```

**Why it helps:** the current Figure 5.3 and Figure 5.4 show separate slices. This frontier shows the whole recommendation logic in one view.

## Plot 4 - Multi-seed Cumulative Return Paths

**Status:** optional but valuable if enough page space remains.

**Purpose:** show the time path behind the summary Sharpe numbers. This addresses a reader question that tables cannot answer: whether `gamma07` is consistently growing through the 24-month test window or whether its Sharpe comes from a few months.

**Suggested filename:**

```text
paper/figures/plot_seed_cumulative_paths.py
paper/figures/fig5_7_seed_cumulative_paths.png
```

**Placement:** Chapter 5 §5.8 or Appendix B.

**Paper-plot style:** `line_confidence_band`.

**Data source:**

- Gamma07 and gamma10 per-seed monthly CSVs:
  - `doc/phase2-fix/phase2_2/gamma_refinement/results/m2_robust_gamma07_seed*_cap05/sanity_metrics_m2_robust_gamma07.csv`
  - `doc/phase2-fix/phase2_2/gamma_refinement/results/m2_robust_gamma10_seed*_cap05/sanity_metrics_m2_robust_gamma10.csv`
- Alpha06 per-seed monthly CSVs only if available from the integrated branch. If not easily accessible, draw only gamma07 vs gamma10 in the main figure and leave alpha06 to Table 5.4.

**Visual design:**

```text
x-axis: Test month
y-axis: Cumulative long-short return
main line: mean cumulative return across seeds
band: min-max seed envelope or ±1 std
series: gamma07, gamma10, optional alpha06
```

Use a mean line plus light band rather than plotting every seed as a separate legend entry. If individual seed paths are shown, use thin translucent lines with no separate legend entry.

**Draft caption:**

```markdown
**Figure 5.7 - Multi-seed cumulative long-short return paths.** Lines show the mean cumulative long-short return across seeds and shaded bands show the seed envelope over the 1995-01..1996-12 test window. `gamma07` is intended to show the recommended stability profile; `gamma10` is included to show the higher-return but wider-seed-dispersion alternative.
```

**Why it helps:** it turns the final recommendation from a table of summary statistics into a portfolio path story.

## Plot 5 - Normalisation Retention Slope Plot

**Status:** optional replacement for current Figure 5.5, not an additional main-text figure unless space allows.

**Purpose:** make the normalisation probe visually sharper. The current Figure 5.5 is already valid. A slope plot would emphasise that `gamma07` is almost unchanged while `gamma10` and `alpha06` drop.

**Suggested filename:**

```text
paper/figures/plot_normalisation_retention_slope.py
paper/figures/fig5_5_normalisation_probe.png
```

This would replace the existing output path rather than creating a new figure number.

**Paper-plot style:** `bar_paired_delta` logic adapted into a two-column slope chart.

**Data source:** `paper/已修改/results_source_of_truth.md` section 5.

**Visual design:**

```text
x positions: Original, Normalised
y-axis: Mean annualised Sharpe
series: gamma07, gamma10, alpha06
annotation: retention percentage
```

Retention values:

```text
gamma07: 0.9112 / 0.9156 ≈ 99.5%
gamma10: 0.4072 / 1.0043 ≈ 40.5%
alpha06: -0.0161 / 0.6895 ≈ -2.3%
```

**Draft caption:**

```markdown
**Figure 5.5 - Sharpe retention under diagnostic component normalisation.** The slope lines compare original and normalised mean Sharpe for the three leading candidates. `gamma07` retains approximately all of its mean Sharpe, whereas `gamma10` and `alpha06` degrade materially under the same diagnostic probe.
```

**Why it helps:** it makes the normalisation story immediately visible. Do not add this as a separate sixth/seventh figure if the current Figure 5.5 remains.

## Plot 6 - Top-candidate Radar Summary

**Status:** optional for presentation slides or appendix only; not recommended as a main-text figure unless the report becomes too table-heavy.

**Purpose:** visually compare `gamma07`, `gamma10`, and `alpha06` across several normalised dimensions.

**Paper-plot style:** `radar_dual_series` only supports two-series comparison most cleanly. If used, make one radar for `gamma07` vs `gamma10`, and another small companion radar for `gamma07` vs `alpha06`; do not force three methods into a crowded radar.

**Axes:**

```text
Mean Sharpe
Low CV
Minimum Sharpe
Cumulative return
Normalisation retention
```

All axes must be normalised to `[0, 1]` and the caption must state that the radar is a visual summary, not a new metric.

**Why it is optional:** radar charts are attractive but can look decorative and may be less rigorous than the Sharpe-CV frontier. Use only if visual variety is needed.

## Recommended final add set

If only two figures can be added:

1. Add **Plot 1 - Hybrid Directional Response Panels** in Chapter 3.
2. Add **Plot 3 - Sharpe-CV Frontier Across Hybrid Families** in Chapter 5.

If one more figure is acceptable:

3. Add **Plot 4 - Multi-seed Cumulative Return Paths** in Chapter 5 or Appendix B.

Avoid adding both Plot 1 and Plot 2 unless one replaces the current Figure 2.1. Otherwise the report will become visually repetitive.

## Implementation notes

Use the existing `paper/figures/_style.py` rather than creating a separate style file. For all new scripts:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import apply_paper_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/<figure_name>.png"
```

For synthetic mechanism figures, state in the caption:

```text
Illustrative; no training data.
```

For empirical figures, state:

```text
Evaluation window 1995-01..1996-12; cap05; three seeds per row where applicable.
```

Before adding any plot to the report, verify:

```bash
python paper/figures/<script>.py
file paper/figures/<output>.png
rg -n "[\\u4e00-\\u9fff]" paper/figures/<script>.py
```

The `rg` check should return no Chinese text in labels, titles, legends, or annotations. Source comments may be bilingual, but figure-visible strings must be English only.
