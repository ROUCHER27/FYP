#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/content/drive/MyDrive/FYP/final_report_all_24m_evidence"
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

RUN_SPECS=(
  "baseline:mse:mse:mse"
  "baseline:medse:medse:medse"
  "baseline:madl:madl:madl"
  "baseline:gmadl:gmadl:gmadl"
  "baseline:imadl:imadl:imadl"
  "baseline:hybrid_mul_m1:hybrid_mul_m1:hybrid_mul_m1"
  "baseline:hybrid_mul_m2:hybrid_mul_m2:hybrid_mul_m2"
  "phase15:A1:hybrid_add_a1:hybrid_add_a1"
  "phase15:A2:hybrid_add_a2:hybrid_add_a2"
  "phase15:A3:hybrid_add_a3:hybrid_add_a3"
  "phase15:A4:hybrid_add_a4:hybrid_add_a4"
  "phase15:A5:hybrid_add_a5:hybrid_add_a5"
  "phase15:M1:hybrid_mul_m1:hybrid_mul_m1"
  "phase15:M2:hybrid_mul_m2:hybrid_mul_m2"
  "phase15:M3:hybrid_mul_m3:hybrid_mul_m3"
  "phase15:M4:hybrid_mul_m4:hybrid_mul_m4"
)

usage() {
  cat <<'USAGE'
Usage: scripts/run_final_report_all_24m_evidence_colab.sh [options]

Runs final-report 24-month evidence for all baseline losses and Phase 1.5
lambda-sweep variants from a Colab checkout.

Default output root:
  /content/drive/MyDrive/FYP/final_report_all_24m_evidence

Groups:
  baseline: mse, medse, madl, gmadl, imadl, hybrid_mul_m1, hybrid_mul_m2
  phase15:  A1, A2, A3, A4, A5, M1, M2, M3, M4

Fixed protocol:
  train 1990-01..1994-12, test 1995-01 for 24 months,
  max-epochs 20, batch-size 1024, seed 42.

Options:
  --output-root PATH       Drive-backed output root.
  --data-dir PATH          CSV data directory, relative to repo root by default.
  --pattern GLOB           CSV glob pattern.
  --best-config-path PATH  best_hyperparameters.txt path.
  --seed INT               Random seed; default 42.
  --force                  Rerun even if completed artifacts verify.
  --dry-run                Print exact commands without training.
  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
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

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

runner_for() {
  local runner_stem="$1"
  local lower
  lower="$(lowercase "$runner_stem")"
  if [[ -f "run_sanity_check_${runner_stem}.py" ]]; then
    printf 'run_sanity_check_%s.py' "$runner_stem"
  elif [[ -f "run_sanity_check_${lower}.py" ]]; then
    printf 'run_sanity_check_%s.py' "$lower"
  else
    printf 'run_sanity_check_%s.py' "$runner_stem"
  fi
}

ensure_repo_root() {
  if [[ ! -f "sanity_check_signal_tilted.py" ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
  fi
}

ensure_output_root() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  case "$OUTPUT_ROOT" in
    /content/drive/*) ;;
    *)
      echo "Refusing non-Drive output root: $OUTPUT_ROOT" >&2
      echo "Mount Google Drive and use a /content/drive/... path." >&2
      exit 1
      ;;
  esac
  mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/reports" "$OUTPUT_ROOT/manifests"
}

write_manifest() {
  local manifest_path="$OUTPUT_ROOT/manifests/run_manifest.json"
  python - "$manifest_path" "$OUTPUT_ROOT" "$DATA_DIR" "$PATTERN" "$BEST_CONFIG_PATH" \
    "$TRAIN_START" "$TRAIN_END" "$TEST_START" "$TEST_MONTHS" "$MAX_EPOCHS" \
    "$BATCH_SIZE" "$SEED" "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)" \
    "$(git rev-parse HEAD 2>/dev/null || true)" "$(timestamp_utc)" "${RUN_SPECS[@]}" <<'PY'
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
    *run_specs,
) = sys.argv[1:]

runs = []
for spec in run_specs:
    group, run_id, runner_stem, artifact_stem = spec.split(":", 3)
    runs.append({
        "group": group,
        "run_id": run_id,
        "runner_stem": runner_stem,
        "artifact_stem": artifact_stem,
    })

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
    "runs": runs,
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

verify_run() {
  local group="$1"
  local run_id="$2"
  local artifact_stem="$3"
  local write_verification="${4:-1}"
  local result_dir="$OUTPUT_ROOT/results/$group/$run_id"
  local verification_path="$OUTPUT_ROOT/manifests/${group}_${run_id}_verification.json"
  python - "$group" "$run_id" "$artifact_stem" "$result_dir" "$verification_path" "$TEST_START" "$TEST_MONTHS" "$write_verification" "$(timestamp_utc)" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    group,
    run_id,
    artifact_stem,
    result_dir,
    verification_path,
    test_start,
    test_months,
    write_verification,
    verified_at,
) = sys.argv[1:]
result_root = Path(result_dir)
verification_file = Path(verification_path)
expected_count = int(test_months)

def add_months(yyyy_mm: str, months: int) -> str:
    year, month = map(int, yyyy_mm.split("-"))
    month_index = (year * 12 + month - 1) + months
    return f"{month_index // 12:04d}-{month_index % 12 + 1:02d}"

def choose_file(kind: str, stems: list[str]) -> Path | None:
    for stem in stems:
        candidate = result_root / f"sanity_{kind}_{stem}.{ 'csv' if kind == 'metrics' else 'json' }"
        if candidate.exists():
            return candidate
    matches = sorted(result_root.glob(f"sanity_{kind}_*.{ 'csv' if kind == 'metrics' else 'json' }"))
    return matches[0] if len(matches) == 1 else None

stems = []
for stem in (artifact_stem, artifact_stem.lower(), run_id, run_id.lower()):
    if stem not in stems:
        stems.append(stem)

csv_file = choose_file("metrics", stems)
summary_file = choose_file("summary", stems)
expected_end = add_months(test_start, expected_count - 1)
problems = []

if csv_file is None:
    problems.append(f"missing metrics CSV under {result_root}")
    rows = []
    months = []
else:
    with csv_file.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    months = [row.get("month", "") for row in rows]
    if len(rows) != expected_count:
        problems.append(f"row_count={len(rows)} expected={expected_count}")
    if months[:1] != [test_start]:
        problems.append(f"first_month={months[:1]} expected={test_start}")
    if months[-1:] != [expected_end]:
        problems.append(f"last_month={months[-1:]} expected={expected_end}")

if summary_file is None:
    problems.append(f"missing summary JSON under {result_root}")

payload = {
    "group": group,
    "run_id": run_id,
    "artifact_stem": artifact_stem,
    "result_dir": str(result_root),
    "metrics_csv": str(csv_file) if csv_file else "",
    "summary_json": str(summary_file) if summary_file else "",
    "row_count": len(rows),
    "first_month": months[0] if months else None,
    "last_month": months[-1] if months else None,
    "expected_row_count": expected_count,
    "expected_first_month": test_start,
    "expected_last_month": expected_end,
    "verified_at_utc": verified_at,
    "status": "failed" if problems else "completed",
    "problems": problems,
}
if write_verification == "1":
    verification_file.write_text(json.dumps(payload, indent=2) + "\n")
if problems:
    raise SystemExit("; ".join(problems))
print(f"Verified {group}/{run_id}: {len(rows)} rows, {months[0]} to {months[-1]}")
PY
}

run_completed() {
  local group="$1"
  local run_id="$2"
  local artifact_stem="$3"
  verify_run "$group" "$run_id" "$artifact_stem" "0" >/dev/null 2>&1
}

write_status_record() {
  local group="$1"
  local run_id="$2"
  local status="$3"
  local detail="$4"
  local status_path="$OUTPUT_ROOT/manifests/${group}_${run_id}_status.json"
  [[ "$DRY_RUN" == "1" ]] && return
  python - "$status_path" "$group" "$run_id" "$status" "$detail" "$(timestamp_utc)" <<'PY'
import json
import sys
from pathlib import Path

path, group, run_id, status, detail, recorded_at = sys.argv[1:]
Path(path).write_text(json.dumps({
    "group": group,
    "run_id": run_id,
    "status": status,
    "detail": detail,
    "recorded_at_utc": recorded_at,
}, indent=2) + "\n")
PY
}

run_one() {
  local group="$1"
  local run_id="$2"
  local runner_stem="$3"
  local artifact_stem="$4"
  local runner
  local result_dir="$OUTPUT_ROOT/results/$group/$run_id"
  local log_file="$OUTPUT_ROOT/logs/${group}_${run_id}.log"
  local command_file="$OUTPUT_ROOT/manifests/${group}_${run_id}_command.txt"
  local started_at
  local ended_at
  runner="$(runner_for "$runner_stem")"
  started_at="$(timestamp_utc)"

  if [[ ! -f "$runner" ]]; then
    echo "Missing runner for $group/$run_id: $runner" >&2
    write_status_record "$group" "$run_id" "missing_runner" "$runner"
    return 1
  fi

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
    --output-dir "$result_dir"
    --seed "$SEED"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run][$group][$run_id] $(quote_cmd "${cmd[@]}")"
    return 0
  fi

  if [[ "$FORCE" != "1" ]] && run_completed "$group" "$run_id" "$artifact_stem"; then
    echo "Skipping $group/$run_id: completed output already verified."
    write_status_record "$group" "$run_id" "skipped" "completed output already verified"
    return 0
  fi

  mkdir -p "$result_dir"
  {
    echo "group=$group"
    echo "run_id=$run_id"
    echo "runner_stem=$runner_stem"
    echo "artifact_stem=$artifact_stem"
    echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "started_at_utc=$started_at"
    echo "output_dir=$result_dir"
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
    write_status_record "$group" "$run_id" "failed" "exit_status=$status"
    return "$status"
  fi

  if verify_run "$group" "$run_id" "$artifact_stem" | tee -a "$log_file"; then
    write_status_record "$group" "$run_id" "completed" "$result_dir"
    return 0
  fi
  write_status_record "$group" "$run_id" "failed_verification" "$result_dir"
  return 1
}

write_report() {
  local txt_path="$OUTPUT_ROOT/reports/final_report_all_24m_evidence_status.txt"
  local csv_path="$OUTPUT_ROOT/reports/final_report_all_24m_evidence_status.csv"
  python - "$OUTPUT_ROOT" "$txt_path" "$csv_path" "$(timestamp_utc)" "${RUN_SPECS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root, txt_path, csv_path, generated_at, *run_specs = sys.argv[1:]
root = Path(output_root)
records = []

for spec in run_specs:
    group, run_id, runner_stem, artifact_stem = spec.split(":", 3)
    verification_path = root / "manifests" / f"{group}_{run_id}_verification.json"
    status_path = root / "manifests" / f"{group}_{run_id}_status.json"
    if verification_path.exists():
        record = json.loads(verification_path.read_text())
    elif status_path.exists():
        record = json.loads(status_path.read_text())
        record.setdefault("artifact_stem", artifact_stem)
        record.setdefault("runner_stem", runner_stem)
    else:
        record = {
            "group": group,
            "run_id": run_id,
            "runner_stem": runner_stem,
            "artifact_stem": artifact_stem,
            "status": "not_run",
        }
    records.append(record)

lines = [
    "Final-report all-gaps 24-month evidence status",
    f"generated_at_utc={generated_at}",
    f"output_root={root}",
    "",
]
for record in records:
    problems = "; ".join(record.get("problems", []))
    detail = record.get("detail", "")
    suffix = problems or detail
    lines.append(
        "{group}/{run_id}: {status}, rows={row_count}, months={first_month}..{last_month}{suffix}".format(
            group=record.get("group"),
            run_id=record.get("run_id"),
            status=record.get("status"),
            row_count=record.get("row_count", ""),
            first_month=record.get("first_month", ""),
            last_month=record.get("last_month", ""),
            suffix=f", detail={suffix}" if suffix else "",
        )
    )
Path(txt_path).write_text("\n".join(lines) + "\n")

fieldnames = [
    "group",
    "run_id",
    "runner_stem",
    "artifact_stem",
    "status",
    "row_count",
    "first_month",
    "last_month",
    "metrics_csv",
    "summary_json",
    "result_dir",
    "detail",
]
with Path(csv_path).open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key, "") for key in fieldnames})

print(f"Wrote report: {txt_path}")
print(f"Wrote CSV: {csv_path}")
PY
}

main() {
  ensure_repo_root
  ensure_output_root
  echo "Repo branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "Repo commit: $(git rev-parse HEAD 2>/dev/null || true)"
  echo "Output root: $OUTPUT_ROOT"
  echo "Train: $TRAIN_START to $TRAIN_END"
  echo "Test: $TEST_START for $TEST_MONTHS months"
  echo "Epochs: $MAX_EPOCHS"
  echo "Batch size: $BATCH_SIZE"
  echo "Seed: $SEED"

  if [[ "$DRY_RUN" != "1" ]]; then
    write_manifest
  fi

  local failures=0
  local spec group run_id runner_stem artifact_stem
  for spec in "${RUN_SPECS[@]}"; do
    IFS=: read -r group run_id runner_stem artifact_stem <<<"$spec"
    echo
    echo "=== $group/$run_id ==="
    if ! run_one "$group" "$run_id" "$runner_stem" "$artifact_stem"; then
      failures=$((failures + 1))
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    write_report
  fi

  if [[ "$failures" -ne 0 ]]; then
    echo "$failures run(s) failed or are missing runners." >&2
    return 1
  fi
  echo "All requested runs completed, skipped as verified, or dry-run printed."
}

main "$@"
