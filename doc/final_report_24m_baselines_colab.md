# Final Report 24-Month Baseline Rerun

Use this in Google Colab after mounting Drive. The script writes all artifacts to a persistent Drive-backed directory and skips already completed metrics when their CSV row count and month range verify successfully.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone Or Update Main

```bash
%%bash
set -euo pipefail
REPO_DIR="/content/FYP"
REPO_URL="https://github.com/ROUCHER27/FYP.git"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin main
git checkout main
git pull --ff-only origin main
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

If the repo is already checked out manually, just `cd /content/FYP` and confirm the branch/commit with the last two commands.

## 3. Install Runtime Dependencies

```bash
%%bash
set -euo pipefail
cd /content/FYP
python -m pip install -q numpy pandas matplotlib seaborn torch
```

## 4. Self-Check The Rerun Script

```bash
%%bash
set -euo pipefail
cd /content/FYP
bash -n scripts/run_final_report_24m_baselines_colab.sh
bash scripts/run_final_report_24m_baselines_colab.sh --dry-run
```

## 5. Run MSE And MedSE 24-Month Baselines

```bash
%%bash
set -euo pipefail
cd /content/FYP
bash scripts/run_final_report_24m_baselines_colab.sh \
  --output-root /content/drive/MyDrive/FYP/final_report_24m_baselines
```

The fixed run settings are:

- `train-start`: `1990-01`
- `train-end`: `1994-12`
- `test-start`: `1995-01`
- `test-months`: `24`
- `max-epochs`: `20`
- `batch-size`: `1024`

## 6. Artifact Locations

```text
/content/drive/MyDrive/FYP/final_report_24m_baselines/
  results/mse/
  results/medse/
  logs/mse.log
  logs/medse.log
  reports/baseline_24m_status.txt
  reports/baseline_24m_verification.csv
  manifests/run_manifest.json
  manifests/mse_command.txt
  manifests/medse_command.txt
  manifests/mse_verification.json
  manifests/medse_verification.json
```

Each verification manifest records the metrics CSV path, summary JSON path, row count, first month, and last month. A completed metric must have `24` rows from `1995-01` through `1996-12`.

## 7. Rerun Behavior

Run the same command again after a Colab disconnect. The script skips a metric only when `sanity_metrics_<metric>.csv` and `sanity_summary_<metric>.json` exist and verify as complete. To force both metrics to rerun:

```bash
%%bash
set -euo pipefail
cd /content/FYP
bash scripts/run_final_report_24m_baselines_colab.sh \
  --output-root /content/drive/MyDrive/FYP/final_report_24m_baselines \
  --force
```

## 8. Optional Copy-Back

If you need a local copy of the Drive artifacts inside the Colab VM for zipping or download:

```bash
%%bash
set -euo pipefail
mkdir -p /content/final_report_24m_baselines_copy
rsync -a /content/drive/MyDrive/FYP/final_report_24m_baselines/ \
  /content/final_report_24m_baselines_copy/
find /content/final_report_24m_baselines_copy -maxdepth 3 -type f | sort
```
