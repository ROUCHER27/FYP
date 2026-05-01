# Phase 2 P0 Colab Run Guide

This guide keeps all outputs, checkpoints, logs, and reports on Google Drive.

## 1. Mount Drive And Clone

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%%bash
set -euo pipefail
cd /content
if [ ! -d FYP ]; then
  git clone https://github.com/ROUCHER27/FYP.git FYP
fi
cd /content/FYP
git fetch --all --prune
git checkout phase2-fixes
git pull --ff-only origin phase2-fixes
python --version
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
pip install -r requirements.txt
```

## 2. Configure Drive Paths

```bash
%%bash
set -euo pipefail
export DRIVE_ROOT="/content/drive/MyDrive/FYP"
mkdir -p "$DRIVE_ROOT/phase2_1b/results" \
  "$DRIVE_ROOT/phase2_1b/checkpoints" \
  "$DRIVE_ROOT/phase2_1b/logs" \
  "$DRIVE_ROOT/phase2_1b/reports" \
  "$DRIVE_ROOT/phase2_p0/loss_scale/results" \
  "$DRIVE_ROOT/phase2_p0/loss_scale/checkpoints" \
  "$DRIVE_ROOT/phase2_p0/loss_scale/logs" \
  "$DRIVE_ROOT/phase2_p0/loss_scale/reports"
find "$DRIVE_ROOT" -maxdepth 2 -type d | sort | sed -n '1,80p'
```

If the CSV files are in the repo root, use `--data-dir /content/FYP`. If they are in Drive, use the Drive data directory instead.

## 3. Smoke Test One Alignment Run

```bash
%%bash
set -euo pipefail
cd /content/FYP
python run_phase2_1b_alignment.py \
  --losses imadl \
  --seeds 42 \
  --data-dir /content/FYP \
  --test-months 1 \
  --max-epochs 1 \
  --output-dir /content/drive/MyDrive/FYP/phase2_1b/results \
  --checkpoint-dir /content/drive/MyDrive/FYP/phase2_1b/checkpoints \
  --log-dir /content/drive/MyDrive/FYP/phase2_1b/logs \
  --resume-mode auto \
  --skip-existing
```

## 4. Run Phase 2.1b Alignment

```bash
%%bash
set -euo pipefail
cd /content/FYP
python run_phase2_1b_alignment.py \
  --losses imadl,gmadl,hybrid_mul \
  --seeds 42,52,62 \
  --caps 0.05 \
  --data-dir /content/FYP \
  --test-months 24 \
  --output-dir /content/drive/MyDrive/FYP/phase2_1b/results \
  --checkpoint-dir /content/drive/MyDrive/FYP/phase2_1b/checkpoints \
  --log-dir /content/drive/MyDrive/FYP/phase2_1b/logs \
  --resume-mode auto \
  --skip-existing
```

## 5. Compare Against Phase 1.5

```bash
%%bash
set -euo pipefail
cd /content/FYP
python compare_phase15_phase21b.py \
  --results-root /content/drive/MyDrive/FYP/phase2_1b/results \
  --losses imadl,gmadl,hybrid_mul \
  --seeds 42,52,62 \
  --caps 0.05 \
  --output-dir /content/drive/MyDrive/FYP/phase2_1b/reports
```

The comparison flags any loss whose mean Sharpe deviates by more than 15% from the Phase 1.5 target.

## 6. Run Loss-Scale Diagnostics

This requires the Phase 2 loss runners to exist. If they are not merged yet, use `--analyze-only` against any existing sanity output directory.

```bash
%%bash
set -euo pipefail
cd /content/FYP
python run_loss_scale_diagnostics.py \
  --losses imadl_m2_alpha05,imadl_gmadl_beta05,m2_robust_gamma01,adaptive_lambda50 \
  --data-dir /content/FYP \
  --test-months 6 \
  --seed 42 \
  --output-root /content/drive/MyDrive/FYP/phase2_p0/loss_scale/results \
  --checkpoint-root /content/drive/MyDrive/FYP/phase2_p0/loss_scale/checkpoints \
  --log-root /content/drive/MyDrive/FYP/phase2_p0/loss_scale/logs \
  --analysis-dir /content/drive/MyDrive/FYP/phase2_p0/loss_scale/reports \
  --resume-mode auto \
  --skip-missing-runners
```

Analyze existing outputs only:

```bash
%%bash
set -euo pipefail
cd /content/FYP
python analyze_loss_scales.py \
  --input-root /content/drive/MyDrive/FYP/phase2_p0/loss_scale/results \
  --output-dir /content/drive/MyDrive/FYP/phase2_p0/loss_scale/reports
```
