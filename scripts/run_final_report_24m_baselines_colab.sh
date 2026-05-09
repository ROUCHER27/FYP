#!/usr/bin/env bash
set -euo pipefail

DRIVE_ROOT="/content/drive/MyDrive/FYP/final_report_24m_baselines"
DATA_DIR="."
PATTERN="*.csv"
BEST_CONFIG_PATH="best_hyperparameters.txt"
TRAIN_START="1990-01"
TRAIN_END="1994-12"
TEST_START="1995-01"
TEST_MONTHS="24"
MAX_EPOCHS="20"
BATCH_SIZE="1024"
SEED="42"
FORCE="0"
DRY_RUN="0"

usage() {
  cat <<'USAGE'
Usage: scripts/run_final_report_24m_baselines_colab.sh [options]

Runs final-report 24-month baseline sanity checks for MSE and MedSE.
Default output root:
  /content/drive/MyDrive/FYP/final_report_24m_baselines

Options:
  --output-root PATH       Drive-backed output root.
  --data-dir PATH          CSV data directory, relative to repo root by default.
  --pattern GLOB           CSV glob pattern.
  --best-config-path PATH  best_hyperparameters.txt path.
  --seed INT               Random seed.
  --force                  Rerun even if completed metric CSVs already exist.
  --dry-run                Validate paths and print commands without training.
  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      DRIVE_ROOT="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --best-config-path)
      BEST_CONFIG_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

quote_cmd() {
  printf "%q " "$@"
}

require_repo_root() {
  local missing=0
  for path in run_sanity_check_mse.py run_sanity_check_medse.py sanity_check_signal_tilted.py; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required runner at repo root: $path" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

ensure_drive_root() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  case "$DRIVE_ROOT" in
    /content/drive/*) ;;
    *)
      echo "Refusing non-Drive output root: $DRIVE_ROOT" >&2
      echo "Use a /content/drive/... path after mounting Google Drive." >&2
      exit 1
      ;;
  esac
  mkdir -p "$DRIVE_ROOT/results" "$DRIVE_ROOT/logs" "$DRIVE_ROOT/reports" "$DRIVE_ROOT/manifests"
}

write_run_manifest() {
  local path="$DRIVE_ROOT/manifests/run_manifest.json"
  python - "$path" "$DRIVE_ROOT" "$DATA_DIR" "$PATTERN" "$BEST_CONFIG_PATH" \
    "$TRAIN_START" "$TRAIN_END" "$TEST_START" "$TEST_MONTHS" "$MAX_EPOCHS" \
    "$BATCH_SIZE" "$SEED" "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)" \
    "$(git rev-parse HEAD 2>/dev/null || true)" "$(timestamp_utc)" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    output_root,
    data_dir,
    pattern,
    best_config_path,
    train_start,
    train_end,
    test_start,
    test_months,
    max_epochs,
    batch_size,
    seed,
    branch,
    commit,
    started_at,
) = sys.argv[1:]

payload = {
    "started_at_utc": started_at,
    "branch": branch,
    "commit": commit,
    "output_root": output_root,
    "subdirs": {
        "results": str(Path(output_root) / "results"),
        "logs": str(Path(output_root) / "logs"),
        "reports": str(Path(output_root) / "reports"),
        "manifests": str(Path(output_root) / "manifests"),
    },
    "parameters": {
        "data_dir": data_dir,
        "pattern": pattern,
        "best_config_path": best_config_path,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_months": int(test_months),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

verify_metric() {
  local metric="$1"
  local csv_path="$DRIVE_ROOT/results/$metric/sanity_metrics_$metric.csv"
  local summary_path="$DRIVE_ROOT/results/$metric/sanity_summary_$metric.json"
  local verification_path="$DRIVE_ROOT/manifests/${metric}_verification.json"
  python - "$metric" "$csv_path" "$summary_path" "$verification_path" "$TEST_START" "$TEST_MONTHS" <<'PY'
import csv
import json
import sys
from pathlib import Path

metric, csv_path, summary_path, verification_path, test_start, test_months = sys.argv[1:]
csv_file = Path(csv_path)
summary_file = Path(summary_path)
verification_file = Path(verification_path)
expected_count = int(test_months)

def add_months(yyyy_mm: str, months: int) -> str:
    year, month = map(int, yyyy_mm.split("-"))
    month_index = (year * 12 + month - 1) + months
    return f"{month_index // 12:04d}-{month_index % 12 + 1:02d}"

if not csv_file.exists():
    raise SystemExit(f"Missing metrics CSV for {metric}: {csv_file}")
if not summary_file.exists():
    raise SystemExit(f"Missing summary JSON for {metric}: {summary_file}")

with csv_file.open(newline="") as fh:
    rows = list(csv.DictReader(fh))

months = [row.get("month", "") for row in rows]
expected_end = add_months(test_start, expected_count - 1)
problems = []
if len(rows) != expected_count:
    problems.append(f"row_count={len(rows)} expected={expected_count}")
if months[:1] != [test_start]:
    problems.append(f"first_month={months[:1]} expected={test_start}")
if months[-1:] != [expected_end]:
    problems.append(f"last_month={months[-1:]} expected={expected_end}")

payload = {
    "metric": metric,
    "metrics_csv": str(csv_file),
    "summary_json": str(summary_file),
    "row_count": len(rows),
    "first_month": months[0] if months else None,
    "last_month": months[-1] if months else None,
    "expected_row_count": expected_count,
    "expected_first_month": test_start,
    "expected_last_month": expected_end,
    "status": "failed" if problems else "completed",
    "problems": problems,
}
verification_file.write_text(json.dumps(payload, indent=2) + "\n")
if problems:
    raise SystemExit("; ".join(problems))
print(
    f"Verified {metric}: {len(rows)} rows, "
    f"{payload['first_month']} to {payload['last_month']}"
)
PY
}

metric_completed() {
  local metric="$1"
  verify_metric "$metric" >/dev/null 2>&1
}

run_metric() {
  local metric="$1"
  local runner="run_sanity_check_${metric}.py"
  local output_dir="$DRIVE_ROOT/results/$metric"
  local log_file="$DRIVE_ROOT/logs/${metric}.log"
  local command_file="$DRIVE_ROOT/manifests/${metric}_command.txt"
  local started_at
  local ended_at
  started_at="$(timestamp_utc)"

  local cmd=(
    python "$runner"
    --data-dir "$DATA_DIR"
    --pattern "$PATTERN"
    --train-start "$TRAIN_START"
    --train-end "$TRAIN_END"
    --test-start "$TEST_START"
    --test-months "$TEST_MONTHS"
    --max-epochs "$MAX_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --best-config-path "$BEST_CONFIG_PATH"
    --output-dir "$output_dir"
    --seed "$SEED"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $(quote_cmd "${cmd[@]}")"
    return
  fi

  if [[ "$FORCE" != "1" ]] && metric_completed "$metric"; then
    echo "Skipping $metric: completed output already verified."
    return
  fi

  mkdir -p "$output_dir"
  {
    echo "metric=$metric"
    echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "started_at_utc=$started_at"
    echo "output_dir=$output_dir"
    echo "command=$(quote_cmd "${cmd[@]}")"
    echo
  } | tee "$log_file" >/dev/null
  quote_cmd "${cmd[@]}" > "$command_file"
  echo >> "$command_file"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  ended_at="$(timestamp_utc)"
  {
    echo
    echo "ended_at_utc=$ended_at"
    echo "exit_status=$status"
  } | tee -a "$log_file" >/dev/null
  if [[ "$status" -ne 0 ]]; then
    echo "$metric failed with exit status $status. See $log_file" >&2
    exit "$status"
  fi
  verify_metric "$metric" | tee -a "$log_file"
}

write_report() {
  local report_path="$DRIVE_ROOT/reports/baseline_24m_status.txt"
  local csv_path="$DRIVE_ROOT/reports/baseline_24m_verification.csv"
  python - "$DRIVE_ROOT" "$report_path" "$csv_path" "$(timestamp_utc)" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root, report_path, csv_path, generated_at = sys.argv[1:]
root = Path(output_root)
records = []
for metric in ("mse", "medse"):
    verification = root / "manifests" / f"{metric}_verification.json"
    if verification.exists():
        records.append(json.loads(verification.read_text()))
    else:
        records.append({"metric": metric, "status": "missing"})

lines = [
    "Final-report 24-month baseline status",
    f"generated_at_utc={generated_at}",
    f"output_root={root}",
    "",
]
for record in records:
    lines.append(
        "{metric}: {status}, rows={row_count}, months={first_month}..{last_month}".format(
            metric=record.get("metric"),
            status=record.get("status"),
            row_count=record.get("row_count"),
            first_month=record.get("first_month"),
            last_month=record.get("last_month"),
        )
    )
Path(report_path).write_text("\n".join(lines) + "\n")

fieldnames = [
    "metric",
    "status",
    "row_count",
    "first_month",
    "last_month",
    "metrics_csv",
    "summary_json",
]
with Path(csv_path).open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key) for key in fieldnames})
print(f"Wrote report: {report_path}")
print(f"Wrote CSV: {csv_path}")
PY
}

main() {
  require_repo_root
  ensure_drive_root
  echo "Repo branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "Repo commit: $(git rev-parse HEAD 2>/dev/null || true)"
  echo "Output root: $DRIVE_ROOT"
  echo "Train: $TRAIN_START to $TRAIN_END"
  echo "Test: $TEST_START for $TEST_MONTHS months"
  echo "Epochs: $MAX_EPOCHS"
  echo "Batch size: $BATCH_SIZE"
  if [[ "$DRY_RUN" == "1" ]]; then
    run_metric mse
    run_metric medse
    return
  fi
  write_run_manifest
  run_metric mse
  run_metric medse
  write_report
  echo "All requested baseline runs completed or were already verified."
}

main "$@"
