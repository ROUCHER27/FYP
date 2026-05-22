# Final Report All-Gaps 24M Evidence Colab Workflow

Use `doc/final_report_all_24m_evidence_colab.ipynb` in Google Colab to run every final-report evidence gap after the corresponding root runners exist.

Output root:

```text
/content/drive/MyDrive/FYP/final_report_all_24m_evidence
```

The workflow writes:

- `results/<group>/<run_id>/`: per-run metrics, summaries, plots, and checkpoints emitted by each runner.
- `logs/<group>_<run_id>.log`: branch, commit, command, timestamps, stdout/stderr, and verification output.
- `manifests/`: run manifest, per-run command files, status JSON, and verification JSON.
- `reports/final_report_all_24m_evidence_status.csv`: machine-readable status summary.
- `reports/final_report_all_24m_evidence_status.txt`: human-readable status summary.

Fixed protocol:

```text
train-start 1990-01
train-end   1994-12
test-start  1995-01
test-months 24
max-epochs  20
batch-size  1024
seed        42
```

Run matrix:

- Baseline losses: `mse`, `medse`, `madl`, `gmadl`, `imadl`, `hybrid_mul_m1`, `hybrid_mul_m2`.
- Phase 1.5 lambda sweep variants: `A1`, `A2`, `A3`, `A4`, `A5`, `M1`, `M2`, `M3`, `M4`.

The script invokes root runner files named `run_sanity_check_<loss>.py`. Phase 1.5 labels are reported as `A1`-`A5` and `M1`-`M4`, while the concrete runner stems are assumed to be `hybrid_add_a1`-`hybrid_add_a5` and `hybrid_mul_m1`-`hybrid_mul_m4`.

Common commands from repo root:

```bash
bash scripts/run_final_report_all_24m_evidence_colab.sh --dry-run
bash scripts/run_final_report_all_24m_evidence_colab.sh
bash scripts/run_final_report_all_24m_evidence_colab.sh --force
```

Normal runs skip outputs that already verify. Verification requires one metrics CSV with 24 rows from `1995-01` through `1996-12` and one summary JSON under the per-run result directory. Use `--force` only when intentionally replacing verified evidence.
