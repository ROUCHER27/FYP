# 2026-05-11 Chapter 3/4 preprocessing and reproducibility boundary review

Scope: `paper/已修改/chapter4_data.md`, `paper/已修改/chapter3_methodology.md`, Appendix A/B, and the referenced Python files.

## 1. Chapter 4 highlighters and Python-reference audit

### 4.1 All highlighted items in Chapter 4

| Location | Highlighted text | Issue type | Decision |
|---|---|---|---|
| `chapter4_data.md:3` | `documents the data` | Wording highlight only | Not necessary to highlight. The phrase is fine, but can be simplified to "describes the data" if aiming for plainer prose. |
| `chapter4_data.md:3` | `Model_Train/data_preprocess.py`, `Model_Train/features.py` | Code-source reference | Useful for internal traceability, but too code-facing for an opening paragraph. Prefer moving the file mapping to Appendix B or a code-reference note. |
| `chapter4_data.md:23` | `F==igure ... ==` | Broken highlight/typo | Formatting issue only. Remove `==`; no content decision needed. |
| `chapter4_data.md:109` | `Model_Train.data_preprocess.prepare_panel_data` | Function-level implementation reference | The reference is accurate and relevant to §4.5, but the function is not currently listed in Appendix A/B. Either keep it and add an Appendix B code map, or rewrite as "The implemented preprocessing pipeline..." without a function path. |
| `chapter4_data.md:125` | `4.6 Data limitations` | Heading highlight only | Remove `==`; the section is necessary and should remain. |

### 4.2 Python references in Chapter 4 and Appendix coverage

| Chapter 4 reference | Exists in repo? | Exists in Appendix A/B? | Necessary in main text? | Recommendation |
|---|---:|---:|---|---|
| `Model_Train/data_preprocess.py` (`chapter4_data.md:3`) | Yes | No | Optional | Move to Appendix B code map; do not foreground in opening paragraph. |
| `Model_Train/features.py` (`chapter4_data.md:3`, `64`) | Yes | No | Yes for X1 provenance | Keep one reference in §4.4 or add Appendix B code map; opening reference can be removed. |
| `Model_Train.data_preprocess.load_raw_csvs` (`chapter4_data.md:19`) | Yes, function in `Model_Train/data_preprocess.py` | No | Optional | Prose description is enough unless the appendix lists the preprocessing code path. |
| `Model_Train/data_preprocess.add_basic_variables` and `add_target_return` (`chapter4_data.md:31`) | Yes | No | Optional | Since formulas immediately define `r`, `to`, and `target_ret`, function names can be omitted or moved to Appendix B. |
| `paper/figures/plot_data_coverage.py` (`chapter4_data.md:27`) | Yes | No | Acceptable in figure caption | Keep if figure regeneration provenance is required; otherwise move script provenance to list of figures/appendix. |
| `Model_Train/features.py` (`chapter4_data.md:64`) | Yes | No | Yes | Keep, because §4.4 distinguishes X1 from unused feature constructors. Add Appendix B map if consistency is desired. |
| `run_sanity_check_*.py` (`chapter4_data.md:103`) | Yes, multiple root runners | Partly: Appendix B lists exact `run_sanity_check_{loss}.py` and `run_sanity_check_hybrid_mul_m2.py` commands | Yes | Keep. Appendix B already partially covers runner scripts, but a compact code map would make coverage explicit. |
| `Model_Train.data_preprocess.prepare_panel_data` (`chapter4_data.md:109`) | Yes, function in `Model_Train/data_preprocess.py` | No | Yes for §4.5 if code-level reproducibility is desired | Keep only if Appendix B adds a preprocessing code-path entry. Otherwise rewrite without the dotted function reference. |
| `sanity_check_signal_tilted.py` (`chapter4_data.md:123`) | Yes | No direct Appendix A/B entry | Borderline | This belongs more naturally in Chapter 3 §3.5 Portfolio construction. In Chapter 4, say only that feature preprocessing does not clip input variables. |

Current appendix status:

- Appendix A covers `Model_Train/losses.py` only.
- Appendix B covers per-seed results and reproduction commands, including runner scripts, but does not list `data_preprocess.py`, `features.py`, `sanity_check_signal_tilted.py`, or figure scripts.
- Therefore, if the final text keeps code-level references in Chapter 4, add a short Appendix B subsection such as "B.7 Code path map" listing data preprocessing, feature construction, portfolio construction, loss definitions, runner scripts, and figure scripts.

Recommended minimum revision for Chapter 4:

1. Remove all `==...==` markers.
2. Keep only the code references that affect reproducibility: `features.py` for X1 and `prepare_panel_data` or `data_preprocess.py` for §4.5.
3. Move broad "verified against code" language from the opening paragraph into Appendix B.
4. In §4.5, do not cite every helper function. The formulas and ordered steps already explain the preprocessing.

## 2. Chapter 3 §3.1 phase overview

Applied edit: the four detailed phase bullets were compressed into one sentence in `paper/已修改/chapter3_methodology.md`:

> The study proceeds through four stages: a seed-42 baseline comparison, a seed-42 hybrid-variant sweep, a multi-seed robust-hybrid refinement, and diagnostic checks that define the evidence gate; §3.7 lists the runner, seed, window, and artifact configuration for each stage.

Reason: §3.7 already gives detailed runner/evidence configuration, so §3.1 should only orient the reader.

## 3. Chapter 3 §3.8 necessity

Recommendation: do not delete the content outright, but shorten it or move details to Appendix B.

Why it is still useful:

1. It protects the report from overclaiming across phases with different seed sets, branches, formulas, and runner configurations.
2. It explains why seed-42 tables are not robustness evidence.
3. It keeps the normalisation probe and Phase 2.5 diagnostics in the correct evidence tier.
4. It is directly aligned with `SCHEMA.md` and `paper/results_source_of_truth.md`.

Why it feels deletable:

1. It partially duplicates Appendix B and Chapter 5 §5.7.
2. The heading sounds defensive in a methodology chapter.
3. The current version is longer than needed if Appendix B remains.

Best compromise:

- Keep a short §3.8 with one reproducibility paragraph and three claim-boundary bullets.
- Move the detailed evidence gate, forbidden inference patterns, and code-path map to Appendix B.
- If deleting the section heading, preserve the key boundaries at the end of §3.7 and in Chapter 5 §5.7.

Important formatting issue: `chapter3_methodology.md:243` has an opening `==` before "Phase 2.5 alignment diagnostics" without a closing marker. If §3.8 is retained, remove that marker during the final cleanup pass.
