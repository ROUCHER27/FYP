# References

Bracket citations used in Chapters 1 and 2 map to the entries below. Entries follow a concise author–year format suitable for Markdown; a LaTeX conversion pass can re-style them without changing content.

[1] Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* The Review of Financial Studies, 33(5), 2223–2273.

[2] Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.

[3] Daniel, K., & Moskowitz, T. J. (2016). *Momentum crashes.* Journal of Financial Economics, 122(2), 221–247.

[4] Bailey, D. H., & López de Prado, M. (2014). *The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality.* The Journal of Portfolio Management, 40(5), 94–107.

[5] Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). *The probability of backtest overfitting.* The Journal of Computational Finance, 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322 (first published online 19 September 2016.)

[6] Harvey, C. R., Liu, Y., & Zhu, H. (2016). *… and the cross-section of expected returns.* The Review of Financial Studies, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059

[7] Michańków, J., Ślepaczuk, R., & Bielak, P. (2024). *Mean Absolute Directional Loss as a new loss function for machine-learning-based trading strategies.* Working paper, University of Warsaw. Introduces the MADL formulation used in Chapters 2 and 3.

[8] Michańków, J., Ślepaczuk, R., & Bielak, P. (2024). *Generalized Mean Absolute Directional Loss (GMADL).* Working paper, University of Warsaw. Extended formulation introducing the sigmoid-based directional loss weighted by $|y|^b$.

[9] Gu, S., Kelly, B., & Xiu, D. (2021). *Autoencoder asset pricing models.* Journal of Econometrics, 222(1), 429–450. Extends [1] with latent-factor extraction; supports the broader ML asset-pricing context of Chapter 2.

[10] Han, C. (2021). *Bimodal characteristic returns and predictability enhancement via machine learning.* Management Science. Demonstrates that ML models can exploit non-linear return predictability from characteristic-based features.

[11] Medhat, M., & Schmeling, M. (2021). *Short-term momentum.* The Review of Financial Studies, 35(3), 1480–1526. Motivates the cumulative-return and cumulative-turnover feature construction used in X1 (Chapter 4 §4.4).

## Auxiliary references

These works are cited by bracket `[A1]..[A8]` in the chapter body where they are substantively used. A LaTeX pass that adopts a full author–year bibliography style should merge them into the main list with their own keys.

[A1] Huber, P. J. (1964). *Robust estimation of a location parameter.* The Annals of Mathematical Statistics, 35(1), 73–101. Canonical reference for the quadratic-linear Huber loss and for the M-estimator view of robust regression used in Chapters 2 and 3.

[A2] Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv:1412.6980 / ICLR 2015. Provides the Adam optimiser invoked in every run (`torch.optim.Adam` with PyTorch defaults; see Chapter 3 §3.4).

[A3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* Journal of Machine Learning Research, 15, 1929–1958. Reference for the dropout regularisation with $p = 0.2$ applied after each hidden-layer ReLU (Chapter 3 §3.2).

[A4] Center for Research in Security Prices (CRSP), The University of Chicago Booth School of Business. *CRSP Monthly Stock File* (data as of the snapshot used in this project). The data source for `PERMNO`, `RET`, `VOL`, and `SHROUT` described in Chapter 4 §4.1.

[A5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS 2019. Implementation framework for the MLP, loss functions, and training loop under `Model_Train/` and `sanity_check_signal_tilted.py`.

[A6] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362. Used for all numeric processing in the runners and figure scripts.

[A7] McKinney, W. (2010). *Data structures for statistical computing in Python.* Proceedings of the 9th Python in Science Conference, 56–61. Reference for pandas, used in the data-loader and figure scripts.

[A8] Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90–95. Reference for matplotlib, used to generate Figures 5.3, 5.4, and 5.5.

## Notes on the bibliography

- The main references above are cited by bracket `[N]` in the chapter prose. Auxiliary references are labelled `[A1]..[A8]` to avoid renumbering the main list; a LaTeX pass can replace the Markdown bracket numbers with `\cite{}` calls while preserving the same numbering.
- Internal project artefacts (Phase 1.5, Phase 2.2 γ refinement, `doc/phase2.5/*`, `doc/final_report_all_24m_evidence/`) are not cited as external works; they are referenced inline by path and form part of the report's source-of-truth evidence (`paper/results_source_of_truth.md`).
- Source-code references (`Model_Train/models.py`, `Model_Train/losses.py`, `Model_Train/features.py`, `Model_Train/data_preprocess.py`, `sanity_check_signal_tilted.py`, `best_hyperparameters.txt`) are cited by file path; they are part of the reproducible artefact bundle and do not appear in this bibliography.
- Figure regeneration scripts under `paper/figures/` (`plot_gamma_refinement.py`, `plot_integrated_sweep.py`, `plot_normalisation_probe.py`) are referenced in figure captions together with their source CSVs / JSONs; they are likewise part of the reproducible bundle rather than the external bibliography.
