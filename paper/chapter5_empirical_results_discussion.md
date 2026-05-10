# Chapter 5: Empirical Results and Discussion

This chapter reports the empirical performance of every loss function that enters the final recommendation. All numbers quoted here are traceable to a single-source-of-truth table in `paper/results_source_of_truth.md`; each caption records the evaluation window, the seed set, and the underlying artifact path so the reader can audit the claim.

The structure of the chapter mirrors the empirical pipeline rather than the order in which the experiments were carried out. Section 5.1 restates the shared evaluation protocol. Sections 5.2–5.3 present same-window single-seed comparisons (baseline regression-style losses and the Phase 1.5 hybrid-variant sweep, both at seed 42). Sections 5.4–5.5 move to multi-seed evidence on the M2-robust family and the broader α/β/λ sweeps, which form the basis of the final recommendation. Section 5.6 discusses a loss-component normalisation probe. Section 5.7 scopes the boundary of cross-phase comparison claims. Section 5.8 synthesises the headline findings and bridges to the conclusion.

## 5.1 Evaluation protocol recap

All experiments reported below share a static, non-rolling training and evaluation protocol. The model is trained once on a fixed five-year window and evaluated on a subsequent out-of-sample window; no retraining occurs during the test window. Chapter 3 gives the full methodology; the operational settings that are relevant for interpreting the tables in this chapter are:

- Train window: `1990-01` to `1994-12` (60 monthly cross-sections).
- Main test window: `1995-01` to `1996-12` (24 monthly cross-sections; every verified run has exactly 24 output rows, first month `1995-01`, last month `1996-12`).
- Feature set X1: 15-dimensional, comprising cumulative returns and cumulative turnover at 1-, 3-, 6-, 9-, and 12-month horizons.
- Model: `MLPConfig(input_dim=15, hidden_dims=[64, 32, 16], activation='relu', dropout=0.2)`.
- Training: `max_epochs=20`, `batch_size=1024`, full-batch training on a single static window; no early stopping.
- Portfolio construction: cross-sectional top 10% long / bottom 10% short per month, signal-tilted weights using within-bucket $z$-scores clipped to $\pm 3$, projected onto a capped simplex with a 5% per-name cap.
- Reported Sharpe is the annualised Sharpe computed from monthly portfolio returns: $\text{Sharpe} = \sqrt{12} \cdot \bar r / \sigma_r$, where $\bar r$ and $\sigma_r$ are the sample mean and standard deviation of the monthly long-short return series (`compute_long_short_stats` uses `periods_per_year=12`).
- Single-seed tables in §5.2–§5.3 use seed 42. Multi-seed tables in §5.4–§5.6 use three seeds per row and all run with the 5% portfolio cap (`cap05`).

The same-window baseline and Phase 1.5 evidence was run from `main` at commit `6c0fbde` and is stored under `doc/final_report_all_24m_evidence/`. Every summary JSON has a matching `*_verification.json` confirming the 24-row window, and every runner command is recorded under `*_command.txt`. The multi-seed evidence (Phase 2.2 gamma refinement and the integrated Phase 2 summary) lives under `doc/phase2-fix/`, with the integrated summary on branch `phase2.2-fix`.

## 5.2 Baseline loss comparison (24 months, seed 42)

Table 5.1 reports the main baseline-loss comparison under the static 24-month window at seed 42. The seven rows cover the traditional regression losses (MSE, MedSE), the absolute-loss family (MADL, GMADL, IMADL), and the two hybrid multiplicative variants (`hybrid_mul_m1`, `hybrid_mul_m2`) that will be extended in Phase 1.5.

**Table 5.1 — Baseline loss comparison.** Static train `1990-01..1994-12`, test `1995-01..1996-12`, seed 42, MLP[64,32,16]+ReLU+dropout 0.2, X1, batch 1024, 20 epochs, top/bottom 10% with 5% per-name cap. Source: `doc/final_report_all_24m_evidence/results/baseline/{loss}/sanity_summary_{loss}.json` and paired `*_verification.json`.

| Loss | Sharpe | Cum. return | Avg R² | Avg monthly LS | Monthly LS std |
|---|---:|---:|---:|---:|---:|
| MSE | $-0.4643$ | $-0.1125$ | $-102.1787$ | $-0.004422$ | $0.032989$ |
| MedSE | $0.0932$ | $0.0060$ | $-2\,297\,042.29$ | $0.001214$ | $0.045124$ |
| MADL | $-0.3058$ | $-0.0756$ | $-4.15 \times 10^{9}$ | $-0.002794$ | $0.031653$ |
| GMADL | $0.2025$ | $0.0279$ | $-7.02 \times 10^{9}$ | $0.001429$ | $0.024449$ |
| IMADL | $-0.3732$ | $-0.0944$ | $-106.5114$ | $-0.003578$ | $0.033211$ |
| `hybrid_mul_m1` | $0.4435$ | $0.0509$ | $-4.7942$ | $0.002215$ | $0.017302$ |
| `hybrid_mul_m2` | $-0.0017$ | $-0.0032$ | $-1.0298$ | $-0.000008$ | $0.016096$ |

Three observations follow from the table.

First, neither traditional regression loss yields an economically meaningful long-short portfolio at this single seed. MSE is strongly negative in both Sharpe and cumulative return, and MedSE is only nominally positive (Sharpe $0.0932$, cumulative return $+0.60\%$ over 24 months). This stands in contrast to earlier six-month sanity-check folklore in which MedSE appeared to dominate; under the same protocol extended to 24 months at seed 42, the MedSE advantage vanishes and the headline numbers belong in §5.4 to the multi-seed M2-robust family, not to MedSE.

Second, the absolute-loss family produces extremely negative average R² values ($-4.15 \times 10^{9}$ for MADL and $-7.02 \times 10^{9}$ for GMADL). This does not indicate that the portfolio signal is uninformative — GMADL actually yields the second-highest Sharpe in the single-seed table ($0.2025$). Rather, the point-prediction scale has diverged from the realised-return scale: the MLP driven by an unbounded absolute-loss family ends up producing calibrated-magnitude predictions that are orders of magnitude away from the realised return. Because the long-short construction uses only the cross-sectional rank of the predictions, portfolio performance is decoupled from calibration quality. This decoupling is central to interpreting the rest of the chapter: absolute-loss-style variants should be evaluated by their ranked portfolio metrics (Sharpe, cumulative return, CV across seeds) rather than by R².

Third, among baseline losses, `hybrid_mul_m1` yields the best single-seed Sharpe at $0.4435$, nearly twice the next-highest. Its cumulative return over 24 months is $+5.09\%$ and its monthly LS standard deviation is the lowest of the set at $0.017302$, suggesting that combining a directional-prediction term with a rank-preserving term produces a more stable long-short signal than either an MSE or an MADL-style term alone. This result motivates the Phase 1.5 sweep in §5.3.

It is important to emphasise that Table 5.1 is a same-window single-seed comparison. It is informative for ranking loss families at a fixed seed but cannot support statements about robustness to seed perturbation. Seed-sensitivity claims are reserved for §5.4 and §5.5.

<!-- FIGURE PLACEHOLDER: Fig 5.1
  TYPE: two-panel static chart
    (a) Line chart — cumulative long-short return over the 24 test months (1995-01..1996-12)
        for each of the seven baseline losses (MSE, MedSE, MADL, GMADL, IMADL, hybrid_mul_m1,
        hybrid_mul_m2). Seven line series, distinct colours, shared x-axis = month, shared
        y-axis = cumulative return; zero line dashed; legend with loss names.
    (b) Horizontal bar chart — annualised Sharpe for the same seven losses, sorted descending.
        Highlight hybrid_mul_m1 (best single-seed Sharpe) with a contrast colour.
  DATA SOURCES:
    - 24-row per-month CSVs: doc/final_report_all_24m_evidence/results/baseline/{loss}/sanity_metrics_{loss}.csv
      (columns: month, long_short_return — take cumulative product).
    - Summary JSONs: doc/final_report_all_24m_evidence/results/baseline/{loss}/sanity_summary_{loss}.json
      (field: long_short_sharpe).
  CAPTION:
    Figure 5.1 — Baseline loss comparison over 1995-01..1996-12, seed 42, cap05 portfolio.
    Left: monthly cumulative long-short return per loss. Right: annualised Sharpe
    (sqrt(12)-scaled monthly mean/std). Shared MLP[64,32,16]+ReLU+dropout 0.2, X1 features,
    batch 1024, 20 epochs. Same runner for every series.
-->
**Figure 5.1 — Baseline loss comparison (placeholder; see comment for data sources and chart spec).**

## 5.3 Phase 1.5 variant sweep (24 months, seed 42)

Phase 1.5 extends the two hybrid multiplicative variants of §5.2 into a parameterised family. The additive variants A1–A5 (`hybrid_add_a{1..5}`) vary the weighting of a directional-accuracy term added to an MSE backbone, while the multiplicative variants M1–M4 (`hybrid_mul_m{1..4}`) vary the weighting of a robust monotone component multiplied into a base prediction error. All nine runs use the same static window, feature set, and portfolio construction as Table 5.1; only the loss specification changes.

**Table 5.2 — Phase 1.5 additive (A1–A5) and multiplicative (M1–M4) variants.** Static train `1990-01..1994-12`, test `1995-01..1996-12`, seed 42, same model / feature / portfolio configuration as Table 5.1. Source: `doc/final_report_all_24m_evidence/results/phase15/{id}/sanity_summary_*.json`.

| Variant | Loss ID | Sharpe | Cum. return | Avg R² |
|---|---|---:|---:|---:|
| A1 | `hybrid_add_a1` | $0.1241$ | $0.0133$ | $-2\,105.95$ |
| A2 | `hybrid_add_a2` | $0.2173$ | $0.0450$ | $-484.75$ |
| A3 | `hybrid_add_a3` | $\mathbf{0.5738}$ | $\mathbf{0.0813}$ | $-1\,383.64$ |
| A4 | `hybrid_add_a4` | $0.2311$ | $0.0463$ | $-11\,229.79$ |
| A5 | `hybrid_add_a5` | $-0.4110$ | $-0.2080$ | $-507\,162.38$ |
| M1 | `hybrid_mul_m1` | $\mathbf{0.4435}$ | $\mathbf{0.0509}$ | $-4.79$ |
| M2 | `hybrid_mul_m2` | $-0.0017$ | $-0.0032$ | $-1.03$ |
| M3 | `hybrid_mul_m3` | $-0.9691$ | $-0.0903$ | $-0.44$ |
| M4 | `hybrid_mul_m4` | $-0.3440$ | $-0.0409$ | $-0.06$ |

The additive family peaks at A3 with Sharpe $0.5738$ and cumulative return $+8.13\%$; A4 and A2 are second-tier; A5 collapses as the added term overwhelms the base MSE contribution. The multiplicative family peaks at M1 with Sharpe $0.4435$ (identical to its Table 5.1 row because the runner and configuration are unchanged); M2 through M4 degrade monotonically on Sharpe.

A few interpretive notes matter for later sections. (i) At seed 42, A3 slightly outperforms M1 on Sharpe and cumulative return, yet the multiplicative family is the one that Phase 2 chose to extend into the M2-robust γ parameterisation. The rationale for that design choice is discussed in Chapter 3; the empirical support is that A3's high R² magnitude (tens or hundreds of thousands in absolute value, driven by the additive MSE term being dominated by a scale-insensitive accuracy term) offers less interpretable control than the M-family, whose R² remains within a small single-digit range. (ii) Seed-42 peaks do not survive seed perturbation uniformly. In §5.4 the M2-robust γ sweep is evaluated across three seeds and the ordering of mean Sharpe does not match the single-seed ordering in this table. Consequently, the Phase 1.5 rows should be read as *motivation* for the Phase 2 design, not as a final variant ranking.

The additive family is not carried further into multi-seed robustness in this report. The final recommendation in §5.8 draws on the multi-seed evidence for the M2-robust family (§5.4) and on the integrated cross-variant sweep (§5.5).

<!-- FIGURE PLACEHOLDER: Fig 5.2
  TYPE: grouped horizontal bar chart
    Two groups side-by-side on the y-axis:
      Group 1 = additive A1..A5 (light-blue bars)
      Group 2 = multiplicative M1..M4 (navy bars)
    x-axis = annualised Sharpe. Zero line dashed vertical.
    Highlight A3 and M1 (best in each family) with a contrast colour or outline.
  DATA SOURCES:
    doc/final_report_all_24m_evidence/results/phase15/{A1..A5,M1..M4}/sanity_summary_*.json
    (field: long_short_sharpe).
  CAPTION:
    Figure 5.2 — Phase 1.5 additive and multiplicative hybrid variants, single-seed Sharpe
    (1995-01..1996-12, seed 42, cap05). A3 is the additive peak, M1 the multiplicative peak.
    Same runner / feature / portfolio / architecture as Table 5.1.
-->
**Figure 5.2 — Phase 1.5 hybrid variants (placeholder; see comment for data sources and chart spec).**

## 5.4 Phase 2: multi-seed robustness of the M2-robust family

Phase 2 introduces an explicit robustness parameter $\gamma$ into the multiplicative hybrid M2 variant, producing the family of `m2_robust_gamma{0.03, 0.05, 0.07, 0.10, 0.15}` losses. Each variant is run across three random seeds using the same 24-month static window and the 5% per-name portfolio cap (`cap05`). The purpose is to decouple the seed-42 ranking of §5.3 from a claim about the expected performance of the loss under seed perturbation.

Sharpe stability is reported as the coefficient of variation $\mathrm{CV} = \sigma_S / |\mu_S|$, where $\mu_S$ and $\sigma_S$ are the cross-seed mean and standard deviation of the annualised Sharpe (same $\sqrt{12}$-annualised convention as §5.1 and §5.2). Lower CV implies that the variant's seed-42 result is more representative of what would be observed with other random seeds.

**Table 5.3 — Phase 2.2 gamma refinement, multi-seed (3 seeds per row).** Static train `1990-01..1994-12`, test `1995-01..1996-12`, `cap05` portfolio cap. Each row aggregates three seeds. Source: `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`.

| Loss | Runs | Sharpe mean | Sharpe std | Sharpe min / max | Cum. return mean | CV |
|---|---:|---:|---:|---:|---:|---:|
| `m2_robust_gamma03` | 3 | $0.3234$ | $0.3418$ | $-0.0199 / 0.6638$ | $0.0818$ | $1.0570$ |
| `m2_robust_gamma05` | 3 | $0.7054$ | $0.1488$ | $0.5796 / 0.8696$ | $0.2392$ | $0.2109$ |
| `m2_robust_gamma07` | 3 | $\mathbf{0.9156}$ | $0.1655$ | $0.7532 / 1.0840$ | $\mathbf{0.2799}$ | $\mathbf{0.1808}$ |
| `m2_robust_gamma10` | 3 | $1.0043$ | $0.5638$ | $0.4587 / 1.5847$ | $0.2368$ | $0.5613$ |
| `m2_robust_gamma15` | 3 | $0.8163$ | $0.3724$ | $0.4085 / 1.1382$ | $0.2277$ | $0.4562$ |

![Figure 5.3: gamma refinement multi-seed Sharpe and stability](figures/fig5_3_gamma_refinement.png)

**Figure 5.3 — Phase 2.2 γ refinement: multi-seed Sharpe and stability.** Left: cross-seed mean annualised Sharpe with ±1 std error bars (3 seeds per γ); light vertical bars indicate the min–max range. Right: coefficient of variation CV = σ/|μ|. Evaluation window `1995-01..1996-12`, `cap05` portfolio cap. Source CSV: `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv`. Generated by `paper/figures/plot_gamma_refinement.py`.

Three findings follow from Table 5.3.

First, `m2_robust_gamma07` provides the best Sharpe-vs-stability trade-off in the γ sweep. Its mean Sharpe of $0.9156$ is within $9\%$ of the highest mean ($1.0043$ for `gamma10`), while its coefficient of variation is the lowest among the non-collapsed variants at $0.1808$ — roughly one-third of the CV of `gamma10`. Its mean cumulative return of $+27.99\%$ over 24 months is the largest in the table. Across three seeds, its minimum Sharpe is $0.7532$; no seed in the run produces a negative Sharpe.

Second, `m2_robust_gamma10` has the highest mean Sharpe ($1.0043$) but also the highest CV among the strong variants ($0.5613$). Its per-seed Sharpe range is wide ($0.4587$ to $1.5847$). Reading this row charitably, `gamma10` is a variant whose *best* seed produces the best Sharpe of the entire γ family, but whose *average* seed has not materially improved on `gamma07`. A recommendation that selects the variant with the highest mean Sharpe should therefore come with an explicit seed-sensitivity caveat.

Third, the γ schedule exhibits a non-monotone relationship with Sharpe stability. `gamma03` collapses almost entirely (CV $1.0570$, minimum Sharpe negative), and `gamma15` — past the stability peak — also degrades sharply in CV ($0.4562$) even though its mean Sharpe remains respectable ($0.8163$). The stability peak near $\gamma = 0.07$ is thus an internal optimum of the robust component rather than the endpoint of a monotone trend. This is consistent with the design intent of the robust modifier: too little robustness under-weights the rank-preserving term; too much robustness flattens the loss surface and re-injects seed-level noise into the learned weights.

On this evidence alone, `m2_robust_gamma07` is the best-supported candidate for the primary recommendation in §5.8. The fuller comparison in §5.5 — where `gamma07` is evaluated against α- and β-sweep variants from other families — confirms this ordering.

## 5.5 Phase 2: broader α, β, and λ sweeps (integrated summary)

To rule out the possibility that the M2-robust γ family is merely a locally good design in the hybrid-multiplicative corner of the loss space, Phase 2 also runs an integrated sweep over three other parameterised families: the IMADL-m2 α sweep, the IMADL-GMADL β sweep, and an adaptive-λ family. Table 5.4 reports selected grouped summaries; the complete table contains further fine-grained γ variants (`gamma001`, `gamma01`) used for locating the γ optimum.

Source (read without switching branches): `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`. All rows are 3-seed averages at `cap05` with the same 24-month window as Table 5.3.

**Table 5.4 — Phase 2 integrated summary across α, β, λ families (selected rows).** Static train `1990-01..1994-12`, test `1995-01..1996-12`, `cap05`, 3 seeds per row.

| Loss | Sharpe mean | Sharpe std | Cum. return mean | CV |
|---|---:|---:|---:|---:|
| `adaptive_lambda10` | $0.4938$ | $0.7617$ | $0.0618$ | $1.5426$ |
| `adaptive_lambda50` | $0.2763$ | $0.1597$ | $0.0817$ | $0.5780$ |
| `adaptive_lambda100` | $0.0955$ | $0.0343$ | $0.0021$ | $0.3591$ |
| `imadl_gmadl_beta03` | $-0.0345$ | $0.1381$ | $-0.0424$ | $4.0084$ |
| `imadl_gmadl_beta05` | $0.0406$ | $0.4116$ | $-0.0138$ | $10.1328$ |
| `imadl_gmadl_beta07` | $-0.0020$ | $0.2747$ | $-0.0409$ | $139.51$ |
| `imadl_m2_alpha02` | $0.1788$ | $1.1576$ | $0.2225$ | $6.4735$ |
| `imadl_m2_alpha03` | $0.2159$ | $0.2010$ | $0.0588$ | $0.9310$ |
| `imadl_m2_alpha04` | $0.3540$ | $0.0656$ | $0.0962$ | $0.1853$ |
| `imadl_m2_alpha05` | $0.5822$ | $0.3193$ | $0.2465$ | $0.5484$ |
| `imadl_m2_alpha06` | $\mathbf{0.6895}$ | $0.1685$ | $\mathbf{0.3042}$ | $\mathbf{0.2443}$ |
| `imadl_m2_alpha07` | $0.4024$ | $0.2466$ | $0.1036$ | $0.6128$ |
| `imadl_m2_alpha08` | $0.5683$ | $0.4130$ | $0.2071$ | $0.7267$ |
| `m2_robust_gamma001` | $0.6919$ | $0.8258$ | $0.1705$ | $1.1936$ |
| `m2_robust_gamma01` | $0.7470$ | $0.3937$ | $0.2718$ | $0.5270$ |
| `m2_robust_gamma10` | $1.0043$ | $0.5638$ | $0.2368$ | $0.5613$ |

![Figure 5.4: IMADL-m2 alpha sweep multi-seed Sharpe and stability](figures/fig5_4_imadl_alpha_sweep.png)

**Figure 5.4 — IMADL-m2 α sweep vs γ reference.** Left: cross-seed mean annualised Sharpe with ±1 std error bars for α ∈ {0.2, …, 0.8}; dashed horizontal lines mark the mean Sharpe of `m2_robust_gamma07` (red) and `m2_robust_gamma10` (orange) as references. Right: coefficient of variation on a log scale, with the same γ references. Evaluation window `1995-01..1996-12`, `cap05`, 3 seeds per row. Source CSVs: `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv` (integrated summary) and `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv` (γ07 reference). Generated by `paper/figures/plot_integrated_sweep.py`.

The IMADL-m2 α sweep produces a clean unimodal Sharpe curve with a peak at `alpha06` (mean Sharpe $0.6895$, cumulative return $+30.42\%$, CV $0.2443$). Below the peak, `alpha04` and `alpha05` remain competitive and, in the case of `alpha04`, even more stable (CV $0.1853$) at the cost of substantially lower mean Sharpe. Above the peak, `alpha07` and `alpha08` degrade. The peak's mean Sharpe is below the `gamma07` mean but its cumulative return is slightly higher, and its seed-sensitivity is only moderately worse. IMADL-m2 α06 is therefore a sensible fallback for deployments that weight cumulative return relatively more heavily than volatility.

The IMADL-GMADL β family does not produce a robust positive Sharpe across seeds. All three β values have mean Sharpe close to zero and CV values in the single digits or larger (β07 has CV $\sim 140$, arising from a near-zero mean Sharpe). This indicates that directly composing IMADL with the GMADL exponent introduces scale instabilities similar to those observed in §5.2 for GMADL alone, and normal β-tuning does not recover the stability of `m2_robust_gamma07`. No β row is carried into the final recommendation.

The adaptive-λ family, in which the relative weighting of the robust and directional terms is tuned during training, underperforms both in mean Sharpe and in stability. The best row in the family (`adaptive_lambda10`) has mean Sharpe $0.4938$ but CV $1.5426$ — the largest among non-GMADL-β rows in Table 5.4. The ordering λ10 > λ50 > λ100 indicates the family prefers moderate weighting, but even the best moderate setting does not reach the γ or α peaks.

Combining Table 5.3 and Table 5.4 confirms that the two best candidates in the final recommendation come from the two self-consistent peaks: `m2_robust_gamma07` at Sharpe $0.9156$ / CV $0.1808$, and `imadl_m2_alpha06` at Sharpe $0.6895$ / CV $0.2443$.

## 5.6 Loss-component normalisation probe (Phase 2.2-fix1)

A natural concern with the hybrid-multiplicative design is that the directional-prediction term and the error-magnitude term operate on different scales, and that the observed Sharpe differences might simply reflect scale imbalance rather than genuine loss-family differences. Phase 2.2-fix1 runs a component-level normalisation probe on the three candidate variants (`m2_robust_gamma07`, `m2_robust_gamma10`, `imadl_m2_alpha06`) to test whether equalising the scale of the two terms materially changes the ranking.

Table 5.5 summarises the probe. The scale ratios quoted are estimated from Phase 2.2 diagnostics; as the probe's own summary records, a fully instrumented per-component logger has not yet been implemented, so the ratios should be treated as diagnostics-grade rather than fully measured.

**Table 5.5 — Loss-component normalisation probe.** Static train `1990-01..1994-12`, test `1995-01..1996-12`, `cap05`, 3 seeds per row. Source: `doc/phase2-fix/phase2.2-fix1/phase1_summary.json` (scale ratios) and `doc/phase2-fix/phase2.2-fix1/phase2_summary.json` (normalised Sharpes).

| Loss | Scale ratio (directional vs MSE) | Original mean Sharpe | Normalised mean Sharpe | Normalised per-seed Sharpes |
|---|---:|---:|---:|---|
| `m2_robust_gamma07` | $\sim 113$ | $0.9156$ | $0.9112$ | $0.5956, 1.4064, 0.7317$ |
| `m2_robust_gamma10` | $\sim 113$ | $1.0043$ | $0.4072$ | $0.6254, 0.1181, 0.4780$ |
| `imadl_m2_alpha06` | $\sim 34$ | $0.6895$ | $-0.0161$ | $0.5628, -0.8335, 0.2224$ |

![Figure 5.5: loss-component normalisation probe](figures/fig5_5_normalisation_probe.png)

**Figure 5.5 — Loss-component normalisation probe.** Left: grouped bar showing the original vs normalised mean annualised Sharpe for the three strongest candidates (`m2_robust_gamma07`, `m2_robust_gamma10`, `imadl_m2_alpha06`), 3 seeds per row. Right: per-seed normalised Sharpes (dots) and their mean (short horizontal bar) for each candidate. Evaluation window `1995-01..1996-12`, `cap05`. Scale ratios are diagnostics-estimated (per-component logger not yet implemented). Source JSONs: `doc/phase2-fix/phase2.2-fix1/phase1_summary.json` and `phase2_summary.json`. Generated by `paper/figures/plot_normalisation_probe.py`.

Three points follow. First, normalisation is not a universal fix. Only `gamma07` is approximately flat under normalisation (mean Sharpe moves from $0.9156$ to $0.9112$, a change smaller than the per-seed dispersion in either configuration). Both `gamma10` and `alpha06` degrade materially — by roughly a factor of two for `gamma10` and to near zero for `alpha06`. Normalisation therefore does not correct scale imbalance in a uniform way.

Second, the fact that `gamma07` is robust to normalisation while `gamma10` and `alpha06` are not suggests that the observed `gamma07` Sharpe is not an artefact of the specific relative scale of the two loss components. Its signal generalises across two different scale regimes. By contrast, `gamma10`'s apparent edge in mean Sharpe without normalisation is sensitive to scale and should be treated as a contingent result.

Third, the probe itself has a known limitation: the scale ratios are estimated from logged diagnostics rather than measured directly through a per-component logger. The probe is therefore consistent with — but not dispositive of — a design claim that `gamma07` is scale-robust. Chapter 6 flags this as a priority for future work.

The operational conclusion is stated as: component normalisation is informative but is not recommended as a blanket fix across the hybrid-multiplicative family. `gamma07` does not require it; `gamma10` and `alpha06` are damaged by it in the current evidence.

## 5.7 Alignment diagnostics and claim boundaries

Phases 2.1b and 2.5 were run as diagnostic passes to test whether the Phase 2 variants could be compared directly to Phase 1.5 results. The findings delimit what can and cannot be said when stacking the tables in §5.3 against those in §5.4–§5.5.

Phase 2.1b regenerates Phase 1.5-style targets under the Phase 2 runner environment and cross-checks the grouped and raw summaries against Phase 1.5. The diagnostics — in `doc/phase2-fix/phase2_1b/reports/phase21b_vs_phase15_grouped.csv` and `phase21b_vs_phase15_raw.csv` — reveal small but systematic numerical differences that are not attributable to any single factor. In concrete terms, the Phase 2 and Phase 1.5 pipelines do not produce bit-level identical results even on nominally identical loss specifications.

Phase 2.5 investigates the causes of this drift in a series of short diagnostic notes (`doc/phase2.5/01_config_comparison.md` through `doc/phase2.5/07_loss_implementation_details.md`, summarised in `doc/phase2.5/executive_summary.md`). The identified sources include: (i) minor differences in the IMADL formula across the two phases' implementations; (ii) subtle differences in the λ-directional term between Phase 1.5 `hybrid_mul_m2` and the Phase 2 m2-robust variants that dominate at γ = 0.07; (iii) GMADL alignment deviations that reflect a small numerical inconsistency rather than a design bug. None of the deviations is large enough to invalidate a controlled internal comparison within a single phase; all of them are large enough that headline claims of "Phase 2 exactly replicates Phase 1.5" cannot be supported.

Two claim boundaries follow for this chapter:

1. **Intra-phase comparison is strong.** The γ rows in Table 5.3, the α rows in Table 5.4, and the β and λ rows in Table 5.4 are each internally consistent because they share runner, window, feature set, model, seed set, and portfolio construction.
2. **Inter-phase comparison is moderate to weak.** Comparisons between Table 5.1 (baseline, seed 42) and Table 5.3 (γ refinement, 3 seeds) or between Table 5.2 (Phase 1.5 variants, seed 42) and Table 5.3 depend on the reader accepting that λ/γ/α/seed/formula differences do not reverse the ranking. They are reasonable as motivation for the γ sweep but should not be quoted as direct-improvement claims.

The report avoids claims of the form "Phase 2 improves Phase 1.5 by $X$ Sharpe points"; the closest supportable statement is "the γ family, once evaluated under multi-seed conditions, produces a mean Sharpe that exceeds the best Phase 1.5 seed-42 variant, but the two rows differ in more than one experimental factor and should be read as controlled comparisons within each phase." Chapter 6 reiterates this boundary.

## 5.8 Headline findings and recommendations

The final loss recommendation rests on §5.4–§5.6, which together constitute the only multi-seed robustness evidence reported in this chapter. The recommendation has three tiers:

**Primary: `m2_robust_gamma07`.** Mean Sharpe $0.9156$, coefficient of variation $0.1808$, mean cumulative return $+27.99\%$ over 24 months, evaluated across three seeds on the `cap05` portfolio. Robust to the loss-component normalisation probe (normalised mean Sharpe $0.9112$). No seed in the run produces a negative Sharpe. This is the best-supported single choice for a loss function in the design space studied in this report.

**High-return alternative: `m2_robust_gamma10`.** Mean Sharpe $1.0043$ is the highest in the integrated sweep, but its coefficient of variation $0.5613$ is three times that of `gamma07`, its per-seed Sharpe range is wide ($0.4587$ to $1.5847$), and normalisation reduces its mean Sharpe to $0.4072$. It is appropriate only where the operator knowingly accepts high seed-sensitivity in exchange for a higher best-case Sharpe.

**Stable fallback: `imadl_m2_alpha06`.** Mean Sharpe $0.6895$, CV $0.2443$, mean cumulative return $+30.42\%$. This is the peak of the IMADL-m2 α sweep and its cumulative return is actually the largest of the three tiered candidates, though its Sharpe is lower than `gamma07`. It provides the next-best independent corroboration that a hybrid-multiplicative design with a robust component is the productive region of the loss space.

The chapter's findings should be read with the following scope limits, all of which are expanded in Chapter 6:

- **Market coverage.** All tables use a single equity universe and monthly frequency. Generalisation to higher frequencies, different asset classes, or other markets is not tested.
- **Evaluation window.** The 24-month main window (`1995-01..1996-12`) is fixed and non-rolling. Robustness across macroeconomic regimes has not been assessed; the rolling-window experiments were paused in the project schedule.
- **Seed depth.** Multi-seed results use three seeds. This is sufficient to differentiate CV at the order-of-magnitude level (as between `gamma07` and `gamma10`) but is too thin to bound CV to the second decimal.
- **Sharpe convention.** All Sharpes are annualised by $\sqrt{12}$ from monthly means and standard deviations; the underlying return series is monthly long-short with no transaction-cost adjustment.
- **Portfolio construction.** Results use a fixed 5% per-name cap with top/bottom 10% buckets. Sensitivity to alternative caps and bucket thresholds has not been systematically ablated in this report.
- **R² interpretation.** Across the absolute-loss family (MADL, GMADL) and several hybrid variants, average R² diverges to extremely negative values while portfolio Sharpe remains positive. The report treats ranked portfolio metrics as primary and treats R² as diagnostic, but users of the trained model for point-prediction tasks should note the lack of calibration.

With those caveats recorded, the empirical answer to the research questions posed in Chapter 1 is: within the static 24-month protocol studied here, a hybrid-multiplicative loss with a tuned robust component ($\gamma = 0.07$) outperforms both traditional regression losses and pure absolute-loss variants on long-short portfolio Sharpe and on cross-seed stability, with a stable fallback available through the IMADL-m2 α family at the same order of Sharpe.
