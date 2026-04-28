#!/usr/bin/env python3
"""
Phase 2 robustness batch runner.

Runs sanity-check scripts with stable output/checkpoint/log paths so local and
Colab executions can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SEEDS = (42, 52, 62)
DEFAULT_CAP = 0.05
DEFAULT_PHASE2_LOSSES = (
    "imadl_m2_alpha02",
    "imadl_m2_alpha03",
    "imadl_m2_alpha04",
    "imadl_m2_alpha05",
    "imadl_m2_alpha06",
    "imadl_m2_alpha07",
    "imadl_m2_alpha08",
    "imadl_gmadl_beta03",
    "imadl_gmadl_beta05",
    "imadl_gmadl_beta07",
    "m2_robust_gamma001",
    "m2_robust_gamma01",
    "m2_robust_gamma10",
    "adaptive_lambda10",
    "adaptive_lambda50",
    "adaptive_lambda100",
)
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60


@dataclass(frozen=True)
class BatchRun:
    loss: str
    seed: int
    max_weight: float | None

    @property
    def cap_tag(self) -> str:
        return cap_tag(self.max_weight)

    @property
    def run_id(self) -> str:
        return f"{self.loss}_seed{self.seed}_{self.cap_tag}"


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return items


def parse_seeds(value: str) -> list[int]:
    try:
        return [int(seed) for seed in parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --seeds value: {value}") from exc


def parse_caps(value: str) -> list[float | None]:
    caps: list[float | None] = []
    for item in parse_csv(value):
        if item.lower() == "none":
            caps.append(None)
        else:
            caps.append(float(item))
    return caps


def cap_tag(max_weight: float | None) -> str:
    if max_weight is None:
        return "nocap"
    return f"cap{int(round(max_weight * 100)):02d}"


def build_runs(losses: Iterable[str], seeds: Iterable[int], caps: Iterable[float | None]) -> list[BatchRun]:
    return [BatchRun(loss, seed, cap) for loss in losses for seed in seeds for cap in caps]


def summary_path(output_dir: Path, loss: str) -> Path:
    return output_dir / f"sanity_summary_{loss}.json"


def metrics_path(output_dir: Path, loss: str) -> Path:
    return output_dir / f"sanity_metrics_{loss}.csv"


def is_complete(output_dir: Path, loss: str) -> bool:
    return summary_path(output_dir, loss).exists() and metrics_path(output_dir, loss).exists()


def resolve_runner(loss: str) -> Path:
    runner = Path(f"run_sanity_check_{loss}.py")
    if not runner.exists():
        raise FileNotFoundError(
            f"Missing runner {runner}. Generate or add it before running loss '{loss}'."
        )
    return runner


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_command(args: argparse.Namespace, run: BatchRun, output_dir: Path, checkpoint_dir: Path) -> list[str]:
    max_weight = "None" if run.max_weight is None else str(run.max_weight)
    return [
        sys.executable,
        str(resolve_runner(run.loss)),
        "--data-dir",
        str(args.data_dir),
        "--pattern",
        args.pattern,
        "--lookback-months",
        str(args.lookback_months),
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--test-start",
        args.test_start,
        "--test-months",
        str(args.test_months),
        "--best-config-path",
        str(args.best_config_path),
        "--max-epochs",
        str(args.max_epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(run.seed),
        "--max-weight",
        max_weight,
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--resume-mode",
        args.resume_mode,
    ]


def run_batch(args: argparse.Namespace) -> int:
    losses = parse_csv(args.losses)
    seeds = parse_seeds(args.seeds)
    caps = parse_caps(args.caps)
    runs = build_runs(losses, seeds, caps)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    log_root = Path(args.log_root)
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_root / "latest_status.json"

    status: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs_total": len(runs),
        "successful": 0,
        "skipped": 0,
        "failed": 0,
        "runs": [],
    }
    print(f"Phase 2 robustness batch: {len(runs)} runs")
    print(f"output_root={output_root}")
    print(f"checkpoint_root={checkpoint_root}")
    print(f"log_root={log_root}")

    for index, run in enumerate(runs, start=1):
        output_dir = output_root / run.run_id
        checkpoint_dir = checkpoint_root / run.run_id
        log_file = log_root / f"{run.run_id}.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {
            "run_id": run.run_id,
            "loss": run.loss,
            "seed": run.seed,
            "max_weight": run.max_weight,
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_file": str(log_file),
        }
        print(f"[{index}/{len(runs)}] {run.run_id}")

        if args.skip_existing and is_complete(output_dir, run.loss):
            print("  SKIP complete")
            record["status"] = "skipped"
            status["skipped"] = int(status["skipped"]) + 1
            status["runs"].append(record)  # type: ignore[union-attr]
            write_json(manifest_path, status)
            continue

        try:
            command = build_command(args, run, output_dir, checkpoint_dir)
            record["command"] = command
            with log_file.open("w") as log_handle:
                log_handle.write(f"run_id={run.run_id}\n")
                log_handle.write(f"started_at={datetime.now(timezone.utc).isoformat()}\n")
                log_handle.write("command=" + " ".join(command) + "\n\n")
                subprocess.run(
                    command,
                    check=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                )
            if not is_complete(output_dir, run.loss):
                raise RuntimeError(f"Run finished without expected summary/metrics for {run.run_id}")
            print("  OK")
            record["status"] = "successful"
            status["successful"] = int(status["successful"]) + 1
        except Exception as exc:  # noqa: BLE001 - batch runner records and continues.
            print(f"  FAILED {exc}")
            record["status"] = "failed"
            record["error"] = str(exc)
            status["failed"] = int(status["failed"]) + 1
            status["runs"].append(record)  # type: ignore[union-attr]
            write_json(manifest_path, status)
            if args.stop_on_error:
                break
        else:
            status["runs"].append(record)  # type: ignore[union-attr]
            write_json(manifest_path, status)

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, status)
    print(
        "Summary: "
        f"successful={status['successful']} skipped={status['skipped']} "
        f"failed={status['failed']} total={status['runs_total']}"
    )
    return 1 if int(status["failed"]) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 sanity-check robustness batches.")
    parser.add_argument("--losses", default=",".join(DEFAULT_PHASE2_LOSSES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--caps", default=str(DEFAULT_CAP), help="Comma-separated caps; use None for uncapped.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument("--train-start", default="1990-01")
    parser.add_argument("--train-end", default="1994-12")
    parser.add_argument("--test-start", default="1995-01")
    parser.add_argument("--test-months", type=int, default=24)
    parser.add_argument("--best-config-path", default="best_hyperparameters.txt")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume-mode", choices=("auto", "never", "require"), default="auto")
    parser.add_argument("--output-root", default="/content/drive/MyDrive/FYP/phase2/results")
    parser.add_argument("--checkpoint-root", default="/content/drive/MyDrive/FYP/phase2/checkpoints")
    parser.add_argument("--log-root", default="/content/drive/MyDrive/FYP/phase2/logs")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main() -> None:
    raise SystemExit(run_batch(build_parser().parse_args()))


if __name__ == "__main__":
    main()
