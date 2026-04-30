#!/usr/bin/env python3
"""Focused Phase 2.2 gamma refinement runner around robust M2 winners."""

from __future__ import annotations

import argparse
import sys

from run_phase2_robustness import run_batch


DEFAULT_GAMMA_REFINEMENT_LOSSES = (
    "m2_robust_gamma03,"
    "m2_robust_gamma05,"
    "m2_robust_gamma07,"
    "m2_robust_gamma10,"
    "m2_robust_gamma15"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2.2 robust-M2 gamma refinement around the current winners."
    )
    parser.add_argument("--losses", default=DEFAULT_GAMMA_REFINEMENT_LOSSES)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--caps", default="0.05")
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
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_root",
        default="/content/drive/MyDrive/FYP/phase2_2/gamma_refinement/results",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint-root",
        dest="checkpoint_root",
        default="/content/drive/MyDrive/FYP/phase2_2/gamma_refinement/checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        "--log-root",
        dest="log_root",
        default="/content/drive/MyDrive/FYP/phase2_2/gamma_refinement/logs",
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=2 * 60 * 60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(run_batch(args))


if __name__ == "__main__":
    main()
