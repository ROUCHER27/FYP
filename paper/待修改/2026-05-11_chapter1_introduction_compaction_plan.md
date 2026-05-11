# Chapter 1 Introduction Compaction Plan - 2026-05-11

Source reviewed: `paper/已修改/chapter1_introduction.md`.

Purpose: respond to the current concern that the Introduction is still too long and is doing part of Chapter 2's job. This document gives a concrete replacement plan for Chapter 1 only. It does not touch the highlighted marks in the source files.

## Overall judgement

The current Introduction is structurally correct but over-expanded. Its main problem is not missing content; it is that Chapter 1 explains too much. It already performs several jobs that should belong to later chapters:

- Chapter 2's literature job: detailed comparison of MSE/MAE, MADL/GMADL/IMADL, robust losses, and hybrid motivation.
- Chapter 3's methodology job: full claim-strength taxonomy, exact model input description, portfolio construction details, and non-claim list.
- Chapter 5's empirical job: table-specific phase mapping and result-specific contribution wording.

Recommended direction: Chapter 1 should be a short entry chapter. It should introduce the problem, state the narrowed research gap, define the research questions, list the contributions, and tell the reader where the evidence appears. It should not explain every loss family or reproduce the evidence gate.

Target length: reduce Chapter 1 from about 2,050 words to about 1,250-1,400 words.

## Structural change

Recommended final section layout:

```text
1.1 Background and motivation
1.2 Research gap
1.3 Research objectives
1.4 Research questions
1.5 Contributions and report structure
```

Delete current `1.5 Scope and claim boundaries` as an independent section.

Reason: the scope boundaries are useful, but their current form duplicates Chapter 3 §3.8 and Chapter 4. Keep only a one-paragraph boundary statement inside `1.3 Research objectives` or `1.5 Contributions and report structure`.

## Section-by-section edits

### 1.1 Background and motivation

Keep the current logic, but compress it to three paragraphs:

1. Quantitative return prediction uses machine learning, but portfolio performance depends on ranking rather than calibrated point forecasts.
2. Generic regression losses do not directly encode this downstream portfolio objective and are sensitive to heavy-tailed monthly returns.
3. Directional losses motivate the project, but their limitations are only introduced briefly; detailed formulas and loss-shape discussion stay in Chapter 2 and Chapter 3.

Delete or move from Chapter 1:

- The detailed four-part GMADL limitation list.
- The reference to Figure 2.1 as if Chapter 1 must interpret the figure.
- Any lengthy explanation of sigmoid, magnitude weighting, or gradient behaviour.

Suggested compressed replacement for the final paragraph of `1.1`:

```markdown
Directional losses such as MADL and GMADL address part of this mismatch by rewarding correct return signs and weighting those signs by realised-return magnitude. They therefore provide a natural starting point for portfolio-oriented training. However, pure directional losses can still be unstable in heavy-tailed monthly equity panels and can drift away from useful prediction-scale diagnostics. This report therefore studies hybrid loss functions that combine directional alignment with robust magnitude control.
```

### 1.2 Research gap

Keep this section, but shorten it. Chapter 2 already reviews the literature in detail, so Chapter 1 should only state the gap at a high level.

Recommended shape:

- One paragraph: architecture/features are studied more often than loss functions.
- One paragraph: robust losses and trading-aware losses are usually studied separately.
- One paragraph: this report's controlled comparison addresses that gap, within explicitly limited evidence boundaries.

Avoid:

- Detailed historical progression from Fama-French to deep networks.
- Listing all three gaps in long form.
- Repeating the full same-conditions protocol.

Suggested replacement:

```markdown
The literature on machine learning for stock-return prediction has paid far more attention to architectures and feature sets than to the training objective itself. MSE and MAE remain common defaults even when the downstream task is ranking stocks into long-short portfolios rather than producing calibrated forecasts.

Two relevant loss-function traditions have developed largely separately. Robust regression losses reduce the effect of heavy-tailed residuals, while directional and trading-aware losses target the sign and ranking information that matters for portfolio construction. What is less well established is how these families compare under the same data, feature set, model, and portfolio rule, or whether hybrid losses can combine their advantages.

This report addresses that gap through a controlled empirical comparison of regression, robust, directional, and hybrid losses. Within each comparison table, the protocol fixes the data, X1 feature input, MLP architecture, training settings, and portfolio construction; cross-phase comparisons are treated as design evidence rather than direct causal proof.
```

### 1.3 Research objectives

Current issue: the objectives are accurate but too phase-heavy. They should read as objectives, not as a Chapter 5 table guide.

Recommended replacement:

```markdown
## 1.3 Research objectives

This report has four objectives.

First, it benchmarks standard regression, robust, directional, and initial hybrid losses under a common 24-month out-of-sample protocol. This establishes the baseline setting in which the loss function is the main design variable.

Second, it develops and evaluates hybrid loss functions that combine a directional component with a robust magnitude component. The purpose is to test whether ranking alignment and residual robustness can be brought into the same training objective.

Third, it evaluates the most relevant hybrid families under multi-seed conditions, using both mean Sharpe and cross-seed stability to avoid relying on a single favourable run.

Fourth, it checks whether the recommended loss remains approximately stable under a diagnostic loss-component normalisation probe. This probe does not prove general robustness, but it helps separate a substantive loss-family effect from a simple component-scaling artefact.

The study is deliberately narrow. It fixes the equity universe, X1 feature input, MLP architecture, static 1990-1994 training window, 1995-1996 test window, and long-short portfolio construction. Architecture search, feature-set sensitivity, rolling-window validation, transaction costs, and non-US or non-monthly settings are outside the main claim boundary.
```

Notes:

- This absorbs the useful part of the deleted `1.5 Scope and claim boundaries`.
- It avoids listing exact feature columns and portfolio weighting details; those belong to Chapter 4 and Chapter 3.
- It still protects the paper from overclaiming.

### 1.4 Research questions

Current issue: RQ2 is long and reads like a methods/results summary. The research questions should be short enough that Chapter 6 can answer them cleanly.

Recommended replacement:

```markdown
## 1.4 Research questions

The objectives above are organised around three research questions.

**RQ1: How does loss choice affect prediction-level and portfolio-level performance under a fixed evaluation protocol?** This question compares standard regression losses, robust losses, directional losses, and initial hybrid losses while holding the data, model, training window, test window, and portfolio rule fixed within each comparison table.

**RQ2: Which hybrid-loss design gives the best supported Sharpe-stability trade-off?** This question first uses a single-seed sweep to map the additive and multiplicative hybrid design space, then uses multi-seed evidence to evaluate the most relevant hybrid families. Single-seed rows are treated as design evidence, not robustness evidence.

**RQ3: Are the leading hybrid-loss candidates approximately stable under diagnostic component normalisation?** This question tests whether the observed ordering is mainly driven by the relative scale of the loss components. The result is interpreted as a diagnostic boundary check rather than a universal normalisation claim.

Chapter 5 reports the empirical evidence for these questions, and Chapter 6 synthesises the answers.
```

Notes:

- This keeps the three-question structure.
- It removes unnecessary references to A-series, M-series, gamma, alpha, beta, and adaptive-lambda from the Introduction.
- Those details should appear in Chapter 5, where the tables can support them.

### Delete current 1.5 Scope and claim boundaries

Delete the section as a standalone section:

```text
## 1.5 Scope and claim boundaries
```

Move only the essential boundary content into the last paragraph of the proposed `1.3 Research objectives`.

Do not keep:

- the detailed "inside scope / outside scope" bullet lists;
- the three-level claim-strength taxonomy;
- the "Non-claims" list.

Where the deleted material belongs:

- Exact data, windows, features: Chapter 4.
- Training protocol, architecture, loss definitions, portfolio construction: Chapter 3.
- Claim-strength taxonomy: Chapter 3 §3.8.
- Evidence strength labels: Chapter 5 table notes.
- Limitations and non-claims: Chapter 6.

### 1.6 Contributions and report structure -> new 1.5

Current issue: the contribution list has seven items and reads like a results inventory. It should be shortened to four contributions plus one report-structure paragraph.

Recommended replacement:

```markdown
## 1.5 Contributions and report structure

The report makes four contributions.

1. **A controlled loss-function benchmark.** It compares standard regression, robust, directional, and initial hybrid losses under a common 24-month evaluation protocol.
2. **A hybrid-loss design study.** It develops additive and multiplicative hybrids that combine directional alignment with robust magnitude control, then maps their behaviour under the fixed protocol.
3. **A multi-seed robustness comparison of leading hybrid candidates.** It evaluates the strongest supported hybrid candidates using mean Sharpe, cumulative return, and cross-seed stability, rather than relying only on seed-42 performance.
4. **A bounded final recommendation.** It identifies `m2_robust_gamma07` as the best-supported primary choice under the reported evidence, with `m2_robust_gamma10` as a higher-return but less stable alternative and `imadl_m2_alpha06` as a stable fallback.

The remainder of the report is organised as follows. Chapter 2 reviews return-prediction, robust-regression, directional-loss, and hybrid-loss literature. Chapter 3 describes the methodology, including model architecture, loss definitions, training protocol, portfolio construction, evaluation metrics, phase design, and claim boundaries. Chapter 4 documents the data source, sample construction, X1 feature input, preprocessing, and data limitations. Chapter 5 presents the empirical results and discusses the final recommendation. Chapter 6 answers the research questions, states the limitations, and outlines future work.
```

Notes:

- This keeps the contribution list short enough for an Introduction.
- It avoids detailed numeric claims in Chapter 1, leaving exact Sharpe/CV values to Chapter 5 and Chapter 6.
- It no longer foregrounds internal branch or phase labels as if the report were a multi-round progress report.

## Concrete deletion list

Delete from current Chapter 1:

- `1.3` final sentence: "Each objective maps to one or two chapters..." because the proposed structure already routes evidence through Chapter 5.
- Current `1.5 Scope and claim boundaries` in full.
- The seven-item contribution list in current `1.6`.

Shorten or rewrite:

- Current `1.1` paragraph 4: reduce the GMADL limitations to one short motivation paragraph.
- Current `1.2` paragraph 2: reduce the three-gap list to one concise paragraph.
- Current `1.4` RQ2: remove detailed variant names and clarify that single-seed rows are not robustness evidence.

## Expected improvement

This change should make Chapter 1 do the right job:

- It introduces the project without becoming a literature review.
- It keeps scope boundaries without duplicating Chapter 3.
- It avoids putting Chapter 5's detailed experiment map before the reader has seen the methodology and data.
- It removes the standalone `Scope and claim boundaries` section, which currently makes the Introduction feel like a defensive methods appendix.
- It keeps the final report aligned with the school structure: Chapter 1 motivates and frames; Chapter 2 reviews literature; Chapter 3 defines methodology; Chapter 4 defines data; Chapter 5 carries experimental detail and results.
