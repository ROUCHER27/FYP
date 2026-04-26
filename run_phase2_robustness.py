#!/usr/bin/env python3
"""
Phase 2 Robustness Experiments Runner

Executes Phase 2 loss function experiments with Google Drive integration.
Supports checkpoint/resume, skip-existing, and matrix modes (light/full).
"""
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FYP")
DEFAULT_SEEDS = (42, 52, 62)
DEFAULT_CAPPED_WEIGHT = 0.05
RUN_TIMEOUT_SECONDS = 7200

# Phase 2 loss functions (13 total)
PHASE2_LOSS_NAMES = [
    # Variant 1: IMADL + M2 Linear (7)
    "imadl_m2_alpha02",
    "imadl_m2_alpha03",
    "imadl_m2_alpha04",
    "imadl_m2_alpha05",
    "imadl_m2_alpha06",
    "imadl_m2_alpha07",
    "imadl_m2_alpha08",
    # Variant 2: IMADL + GMADL Weighted (3)
    "imadl_gmadl_beta03",
    "imadl_gmadl_beta05",
    "imadl_gmadl_beta07",
    # Variant 3: M2 Robustness Enhanced (3)
    "m2_robust_gamma001",
    "m2_robust_gamma01",
    "m2_robust_gamma10",
    # Variant 4: Adaptive Hybrid (3)
    "adaptive_lambda10",
    "adaptive_lambda50",
    "adaptive_lambda100",
]


@dataclass(frozen=True)
class Phase2Run:
    """Represents a single Phase 2 experiment run."""
    loss_name: str
    seed: int
    max_weight: float | None


def parse_seeds(value: str) -> List[int]:
    """Parse comma-separated seed values."""
    seeds = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_losses(value: str) -> List[str]:
    """Parse comma-separated loss function names."""
    losses = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in PHASE2_LOSS_NAMES:
            raise ValueError(f"Unknown Phase 2 loss function: {token}")
        losses.append(token)
    if not losses:
        raise ValueError("At least one loss function is required.")
    return losses


def cap_tag_from_weight(max_weight: float | None) -> str:
    """Generate a tag string for the weight cap."""
    if max_weight is None:
        return "nocap"
    return f"cap{int(round(max_weight * 100)):02d}"


def build_run_matrix(
    losses: Iterable[str],
    seeds: Iterable[int],
    matrix_mode: str,
) -> List[Phase2Run]:
    """
    Build the experiment matrix.

    Args:
        losses: Loss function names
        seeds: Random seeds
        matrix_mode: "light" (Phase 2.1) or "full" (Phase 2.2)

    Returns:
        List of Phase2Run configurations
    """
    ordered_losses = list(losses)
    ordered_seeds = list(seeds)
    runs: List[Phase2Run] = []

    if matrix_mode == "light":
        # Phase 2.1: 3 seeds × 1 weight cap (0.05)
        for loss_name in ordered_losses:
            for seed in ordered_seeds:
                runs.append(
                    Phase2Run(
                        loss_name=loss_name,
                        seed=seed,
                        max_weight=DEFAULT_CAPPED_WEIGHT,
                    )
                )
    elif matrix_mode == "full":
        # Phase 2.2: 6 seeds × 2 weight caps (0.05 and None)
        for loss_name in ordered_losses:
            for seed in ordered_seeds:
                # Capped runs
                runs.append(
                    Phase2Run(
                        loss_name=loss_name,
                        seed=seed,
                        max_weight=DEFAULT_CAPPED_WEIGHT,
                    )
                )
                # Uncapped runs
                runs.append(
                    Phase2Run(
                        loss_name=loss_name,
                        seed=seed,
                        max_weight=None,
                    )
                )
    else:
        raise ValueError(f"Unsupported matrix mode: {matrix_mode}")

    return runs


def derive_run_paths(
    drive_root: Path,
    run: Phase2Run,
) -> Dict[str, Path]:
    """
    Derive output and checkpoint paths for a run.

    Directory structure:
    {drive_root}/phase2/
        ├── results/{loss_name}_seed{seed}_{cap_tag}/
        ├── checkpoints/{loss_name}_seed{seed}_{cap_tag}/
        └── logs/{loss_name}_seed{seed}_{cap_tag}.log
    """
    cap_tag = cap_tag_from_weight(run.max_weight)
    run_id = f"{run.loss_name}_seed{run.seed}_{cap_tag}"

    base_root = Path(drive_root) / "phase2"
    output_dir = base_root / "results" / run_id
    checkpoint_dir = base_root / "checkpoints" / run_id
    log_file = base_root / "logs" / f"{run_id}.log"

    return {
        "base_root": base_root,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_file": log_file,
    }


def is_run_complete(output_dir: Path, loss_name: str) -> bool:
    """Check if a run has completed successfully."""
    summary_file = output_dir / f"sanity_summary_{loss_name}.json"
    metrics_file = output_dir / f"sanity_metrics_{loss_name}.csv"
    return summary_file.exists() and metrics_file.exists()


def resolve_runner_path(loss_name: str) -> Path:
    """Resolve the path to the runner script for a loss function."""
    runner_script = f"run_sanity_check_{loss_name}.py"
    runner_path = Path(runner_script)
    if not runner_path.exists():
        raise FileNotFoundError(
            f"Runner script not found: {runner_script}. "
            f"Please create it before running Phase 2 experiments."
        )
    return runner_path


def build_command_for_run(
    *,
    run: Phase2Run,
    data_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
    paths: Dict[str, Path],
) -> List[str]:
    """Build the command to execute a single run."""
    cmd = [
        sys.executable,
        str(resolve_runner_path(run.loss_name)),
        "--data-dir", str(data_dir),
        "--train-start", train_start,
        "--train-end", train_end,
        "--test-start", test_start,
        "--test-months", str(test_months),
        "--max-epochs", str(max_epochs),
        "--batch-size", str(batch_size),
        "--seed", str(run.seed),
        "--output-dir", str(paths["output_dir"]),
        "--checkpoint-dir", str(paths["checkpoint_dir"]),
        "--resume-mode", resume_mode,
    ]

    # Add max-weight parameter
    if run.max_weight is None:
        cmd.extend(["--max-weight", "None"])
    else:
        cmd.extend(["--max-weight", str(run.max_weight)])

    return cmd


def run_phase2_robustness(
    *,
    losses: Iterable[str],
    seeds: Iterable[int],
    matrix_mode: str,
    drive_root: Path,
    data_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
    skip_existing: bool,
    stop_on_error: bool,
) -> Dict[str, int]:
    """
    Execute Phase 2 robustness experiments.

    Returns:
        Dictionary with execution statistics
    """
    losses = list(losses)
    runs = build_run_matrix(losses, seeds, matrix_mode)

    successful = 0
    skipped = 0
    failed = 0

    print(f"Phase 2 Robustness Experiments")
    print(f"Matrix mode: {matrix_mode}")
    print(f"Total runs: {len(runs)}")
    print(f"Losses: {', '.join(losses)}")
    print(f"Seeds: {', '.join(map(str, seeds))}")
    print("-" * 80)

    for i, run in enumerate(runs, 1):
        cap_tag = cap_tag_from_weight(run.max_weight)
        print(f"\n[{i}/{len(runs)}] {run.loss_name} | seed={run.seed} | cap={cap_tag}")

        paths = derive_run_paths(drive_root, run)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        paths["log_file"].parent.mkdir(parents=True, exist_ok=True)

        # Skip if already complete
        if skip_existing and is_run_complete(paths["output_dir"], run.loss_name):
            print(f"  [SKIP] Already complete")
            skipped += 1
            continue

        # Build and execute command
        command = build_command_for_run(
            run=run,
            data_dir=data_dir,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_months=test_months,
            max_epochs=max_epochs,
            batch_size=batch_size,
            resume_mode=resume_mode,
            paths=paths,
        )

        try:
            # Run with output redirected to log file
            with open(paths["log_file"], "w") as log_f:
                subprocess.run(
                    command,
                    check=True,
                    timeout=RUN_TIMEOUT_SECONDS,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )

            # Verify completion
            if not is_run_complete(paths["output_dir"], run.loss_name):
                raise RuntimeError(
                    f"Run finished without complete outputs: {run.loss_name}"
                )

            print(f"  [SUCCESS]")
            successful += 1

        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            print(f"  [FAILED] {exc}")
            failed += 1
            if stop_on_error:
                print("\nStopping due to error (--stop-on-error enabled)")
                break

    # Print summary
    print("\n" + "=" * 80)
    print("Phase 2 Execution Summary")
    print("=" * 80)
    print(f"Total runs:    {len(runs)}")
    print(f"Successful:    {successful}")
    print(f"Skipped:       {skipped}")
    print(f"Failed:        {failed}")
    print(f"Completed:     {successful + skipped}/{len(runs)}")
    print("=" * 80)

    return {
        "completed": successful + skipped,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "total": len(runs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Robustness Experiments with Google Drive Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 2.1 (light mode): 13 losses × 3 seeds × 1 cap = 39 runs
  python run_phase2_robustness.py \\
    --matrix-mode light \\
    --drive-root /content/drive/MyDrive/FYP \\
    --data-dir /content/drive/MyDrive/FYP/data \\
    --skip-existing

  # Phase 2.2 (full mode): 3 losses × 6 seeds × 2 caps = 36 runs
  python run_phase2_robustness.py \\
    --matrix-mode full \\
    --losses imadl_m2_alpha05,imadl_gmadl_beta05,adaptive_lambda50 \\
    --seeds 42,52,62,72,82,92 \\
    --test-months 48 \\
    --skip-existing
        """
    )

    # Experiment control
    parser.add_argument(
        "--matrix-mode",
        choices=("light", "full"),
        default="light",
        help="Light: Phase 2.1 (3 seeds, 1 cap). Full: Phase 2.2 (6 seeds, 2 caps).",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=",".join(PHASE2_LOSS_NAMES),
        help="Comma-separated loss function names (default: all 13 Phase 2 losses).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated random seeds (default: 42,52,62).",
    )

    # Paths
    parser.add_argument(
        "--drive-root",
        type=str,
        default=str(DEFAULT_DRIVE_ROOT),
        help="Google Drive root path for outputs and checkpoints.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing CSV data files.",
    )

    # Time windows
    parser.add_argument(
        "--train-start",
        default="1990-01",
        help="Training start date (YYYY-MM format).",
    )
    parser.add_argument(
        "--train-end",
        default="1994-12",
        help="Training end date (YYYY-MM format).",
    )
    parser.add_argument(
        "--test-start",
        default="1995-01",
        help="Testing start date (YYYY-MM format).",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=24,
        help="Number of test months (default: 24 for Phase 2.1, 48 for Phase 2.2).",
    )

    # Training parameters
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size.",
    )

    # Execution control
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "never", "require"),
        default="auto",
        help="Checkpoint resume mode (auto: resume if exists, never: always fresh, require: fail if no checkpoint).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that have already completed successfully.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution on first error (default: continue to next run).",
    )

    return parser


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parse arguments
    losses = parse_losses(args.losses)
    seeds = parse_seeds(args.seeds)
    drive_root = Path(args.drive_root)
    data_dir = Path(args.data_dir)

    # Validate data directory
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Execute experiments
    stats = run_phase2_robustness(
        losses=losses,
        seeds=seeds,
        matrix_mode=args.matrix_mode,
        drive_root=drive_root,
        data_dir=data_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_months=args.test_months,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        resume_mode=args.resume_mode,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error,
    )

    # Exit with appropriate code
    if stats["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
