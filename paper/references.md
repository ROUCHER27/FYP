# References

Bracket citations used in Chapters 1 and 2 map to the entries below. Entries follow a concise author–year format suitable for Markdown; a LaTeX conversion pass can re-style them without changing content.

[1] Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* The Review of Financial Studies, 33(5), 2223–2273.

[2] Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.

[3] Daniel, K., & Moskowitz, T. J. (2016). *Momentum crashes.* Journal of Financial Economics, 122(2), 221–247.

[4] Lopez de Prado, M. (2014). *Deflating the Sharpe ratio.* Working paper / Journal of Portfolio Management (reflecting the Bailey and Lopez de Prado treatment of inflated Sharpe statistics under multiple testing).

[5] Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). *The probability of backtest overfitting.* Journal of Computational Finance (forthcoming / working paper version).

[6] Harvey, C. R., Liu, Y., & Zhu, H. (2016). *… and the cross-section of expected returns.* The Review of Financial Studies, 29(1), 5–68.

[7] Michańków, J., Ślepaczuk, R., & Bielak, P. (2024). *Mean Absolute Directional Loss as a new loss function for machine-learning-based trading strategies.* Working paper / conference preprint. Introduces the MADL formulation used in Chapters 2 and 3.

[8] Michańków, J., Ślepaczuk, R., & Bielak, P. (2024). *Generalized Mean Absolute Directional Loss (GMADL).* Companion / extended formulation introducing the sigmoid-based directional loss weighted by $|y|^b$.

## Notes on the bibliography

- The references above are cited by bracket `[N]` in the chapter prose. A LaTeX pass can replace the Markdown bracket numbers with `\cite{}` calls while preserving the same numbering.
- Robust-regression references (Huber 1964 and subsequent M-estimator work) are named in Chapter 2 §2.2 but without bracket citation; a full bibliography preparation pass should add them as auxiliary entries once the thesis template is selected.
- Internal project artefacts (Phase 1.5, Phase 2.2 γ refinement, `doc/phase2.5/*`, `doc/final_report_all_24m_evidence/`) are not cited as external works; they are referenced inline by path and form part of the report's source-of-truth evidence (`paper/results_source_of_truth.md`).
- Source-code references (`Model_Train/models.py`, `Model_Train/losses.py`, `Model_Train/features.py`, `Model_Train/data_preprocess.py`, `sanity_check_signal_tilted.py`, `best_hyperparameters.txt`) are cited by file path; they are part of the reproducible artefact bundle and do not appear in this bibliography.
