#!/usr/bin/env python3
"""Run representative P0.2 sanity checks and analyze available scale outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analyze_loss_scales import main as analyze_main


DEFAULT_LOSSES = "imadl_m2_alpha06,m2_robust_gamma01,m2_robust_gamma10"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def runner_exists(loss: str) -> bool:
    return Path(f"run_sanity_check_{loss}.py").exists()


def run_sanity(args: argparse.Namespace, loss: str) -> int:
    output_dir = Path(args.output_root) / loss
    checkpoint_dir = Path(args.checkpoint_root) / loss
    log_file = Path(args.log_root) / f"{loss}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        f"run_sanity_check_{loss}.py",
        "--data-dir",
        args.data_dir,
        "--pattern",
        args.pattern,
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--test-start",
        args.test_start,
        "--test-months",
        str(args.test_months),
        "--best-config-path",
        args.best_config_path,
        "--max-epochs",
        str(args.max_epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--max-weight",
        str(args.max_weight),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--resume-mode",
        args.resume_mode,
    ]
    with log_file.open("w") as handle:
        handle.write("command=" + " ".join(command) + "\n\n")
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0.2 representative diagnostics and analyze outputs.")
    parser.add_argument("--losses", default=DEFAULT_LOSSES)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--train-start", default="1990-01")
    parser.add_argument("--train-end", default="1994-12")
    parser.add_argument("--test-start", default="1995-01")
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--best-config-path", default="best_hyperparameters.txt")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-weight", default="0.05")
    parser.add_argument("--resume-mode", choices=("auto", "never", "require"), default="auto")
    parser.add_argument("--output-root", default="/content/drive/MyDrive/FYP/phase2_p0/loss_scale/results")
    parser.add_argument("--checkpoint-root", default="/content/drive/MyDrive/FYP/phase2_p0/loss_scale/checkpoints")
    parser.add_argument("--log-root", default="/content/drive/MyDrive/FYP/phase2_p0/loss_scale/logs")
    parser.add_argument("--analysis-dir", default="/content/drive/MyDrive/FYP/phase2_p0/loss_scale/reports")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-missing-runners", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=2 * 60 * 60)
    args = parser.parse_args()

    failures = 0
    if not args.analyze_only:
        for loss in parse_csv(args.losses):
            if not runner_exists(loss):
                message = f"Missing runner for {loss}; expected run_sanity_check_{loss}.py"
                if args.skip_missing_runners:
                    print(f"SKIP {message}")
                    continue
                raise SystemExit(message)
            code = run_sanity(args, loss)
            if code != 0:
                print(f"FAILED {loss} exit={code}")
                failures += 1

    sys.argv = [
        "analyze_loss_scales.py",
        "--input-root",
        args.output_root,
        "--output-dir",
        args.analysis_dir,
    ]
    analyze_main()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
