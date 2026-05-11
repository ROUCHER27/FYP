# Review: Chapter 4 Data

Source reviewed: `paper/chapter4_data.md`.

## Summary

Chapter 4 is necessary and generally well placed, but its feature-section inconsistency affects the whole report. It also needs more careful wording around file-level month counts, forward-filling returns, and survivorship/delisting assumptions.

## Content and evidence issues

1. `chapter4_data.md:88` says X1 has 15 dimensions, then defines only 10 features: five horizons times cumulative return and cumulative turnover. This is the highest-priority fix in the chapter. Either X1 is 10 engineered features plus 5 retained raw/base variables, or the model input description is wrong. Add a table of the exact columns passed into the MLP.

2. `chapter4_data.md:21` says `89.12-94.csv` contributes 61 distinct year-months. That is file coverage including `1989-12`, not the training window. Later `chapter4_data.md:155` says the "training window contains ... 61 month-sections," which conflicts with the 60-month training window. Change this to "the source file contains 61 month-sections; the training partition uses 60."

3. `chapter4_data.md:63` says the target definition prevents look-ahead leakage. This is directionally correct, but feature construction also depends on lagging and split timing. Add "provided that feature rows are built only from lagged variables as in §4.4."

4. `chapter4_data.md:136` calls forward-filling `{RET,VOL,SHROUT}` a "standard conservative treatment." Forward-filling returns can fabricate a return observation and may not be conservative. Either cite this choice, justify it as an implementation fact rather than a standard, or flag it as a data limitation.

5. `chapter4_data.md:145` says no winsorisation is applied, but if the exact 15 model inputs include raw `RET`, `VOL`, `SHROUT`, `r`, or `to`, this point becomes even more important. The feature-dimension fix should state whether raw/unscaled variables enter the model.

6. `chapter4_data.md:156` says CRSP monthly files typically include delisting rows via `RET`, but this project's data snapshot has not verified delisting return handling. Good limitation, but it should be elevated: long-short performance can be biased materially by missing delisting returns.

7. `chapter4_data.md:23-42` still contains a figure-placeholder HTML comment even though `fig4_1_data_coverage.png` exists. Delete the comment block.

## SciWrite issues

1. `chapter4_data.md:7` says "roughly five years" but also "one-month overlap." This is fine, but a small file-coverage table would be clearer than prose.

2. The chapter uses "CRSP-style" throughout. If the data is actually CRSP, say "CRSP monthly stock file" and cite `[A4]`; if not, explain why "style" is used.

3. `chapter4_data.md:51-65` is clear, but the target formula uses `y_{i,t}=r_{i,t+1}` while later chapters use `y_{i,t+1}`. Standardise notation.

## Top priority revisions

1. Fix X1 dimensions and list exact model inputs.
2. Correct 60 vs 61 month wording.
3. Reframe forward-fill as an implementation choice with a limitation.
4. Remove figure-placeholder comments.
