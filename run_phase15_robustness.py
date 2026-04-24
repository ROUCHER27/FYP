import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from Model_Train.losses import EXPERIMENT_LOSS_NAMES
from run_all_experiments import (
    is_loss_complete,
    parse_losses,
    resolve_runner_path,
    validate_runner_scripts_exist,
)


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FYP")
DEFAULT_SEEDS = (42, 52, 62)
DEFAULT_CAPPED_WEIGHT = 0.05
RUN_TIMEOUT_SECONDS = 7200


@dataclass(frozen=True)
class Phase15Run:
    loss_name: str
    seed: int
    max_weight: float | None


def parse_seeds(value: str) -> List[int]:
    seeds = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def cap_tag_from_weight(max_weight: float | None) -> str:
    if max_weight is None:
        return "nocap"
    return f"cap{int(round(max_weight * 100)):03d}"


def build_run_matrix(
    losses: Iterable[str],
    seeds: Iterable[int],
    matrix_mode: str,
    nocap_seed: int,
) -> List[Phase15Run]:
    ordered_losses = list(losses)
    ordered_seeds = list(seeds)
    runs: List[Phase15Run] = []

    for loss_name in ordered_losses:
        for seed in ordered_seeds:
            runs.append(
                Phase15Run(
                    loss_name=loss_name,
                    seed=seed,
                    max_weight=DEFAULT_CAPPED_WEIGHT,
                )
            )
        if matrix_mode == "light":
            runs.append(
                Phase15Run(loss_name=loss_name, seed=nocap_seed, max_weight=None)
            )
        elif matrix_mode == "full":
            for seed in ordered_seeds:
                runs.append(Phase15Run(loss_name=loss_name, seed=seed, max_weight=None))
        else:
            raise ValueError(f"Unsupported matrix mode: {matrix_mode}")
    return runs


def derive_run_paths(
    drive_root: Path,
    matrix_mode: str,
    run: Phase15Run,
) -> Dict[str, Path]:
    cap_tag = cap_tag_from_weight(run.max_weight)
    base_root = Path(drive_root) / "outputs" / "phase1_5_robustness" / matrix_mode
    run_root = base_root / "runs" / f"seed{run.seed}_{cap_tag}" / run.loss_name
    checkpoint_root = (
        base_root / "checkpoints" / f"seed{run.seed}_{cap_tag}" / run.loss_name
    )
    return {
        "base_root": base_root,
        "output_dir": run_root,
        "checkpoint_dir": checkpoint_root,
    }


def build_command_for_run(
    *,
    run: Phase15Run,
    data_dir: Path,
    pattern: str,
    lookback_months: int,
    best_config_path: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
    paths: Dict[str, Path],
) -> List[str]:
    return [
        sys.executable,
        str(resolve_runner_path(run.loss_name)),
        "--data-dir",
        str(data_dir),
        "--pattern",
        pattern,
        "--lookback-months",
        str(lookback_months),
        "--best-config-path",
        str(best_config_path),
        "--train-start",
        train_start,
        "--train-end",
        train_end,
        "--test-start",
        test_start,
        "--test-months",
        str(test_months),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(run.seed),
        "--max-weight",
        "None" if run.max_weight is None else str(run.max_weight),
        "--output-dir",
        str(paths["output_dir"]),
        "--checkpoint-dir",
        str(paths["checkpoint_dir"]),
        "--resume-mode",
        resume_mode,
    ]


def run_phase15_robustness(
    *,
    losses: Iterable[str],
    seeds: Iterable[int],
    matrix_mode: str,
    drive_root: Path,
    data_dir: Path,
    pattern: str,
    lookback_months: int,
    best_config_path: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
    nocap_seed: int,
    skip_existing: bool,
    stop_on_error: bool,
) -> Dict[str, int]:
    losses = list(losses)
    validate_runner_scripts_exist(losses)
    runs = build_run_matrix(losses, seeds, matrix_mode, nocap_seed)
    successful = 0
    skipped = 0
    failed = 0

    for run in runs:
        paths = derive_run_paths(drive_root, matrix_mode, run)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

        if skip_existing and is_loss_complete(paths["output_dir"], run.loss_name):
            skipped += 1
            continue

        command = build_command_for_run(
            run=run,
            data_dir=data_dir,
            pattern=pattern,
            lookback_months=lookback_months,
            best_config_path=best_config_path,
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
            subprocess.run(command, check=True, timeout=RUN_TIMEOUT_SECONDS)
            if not is_loss_complete(paths["output_dir"], run.loss_name):
                raise RuntimeError(
                    f"Run finished without complete outputs: {run.loss_name} seed={run.seed} max_weight={run.max_weight}"
                )
            successful += 1
        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            failed += 1
            print(
                f"FAILED {run.loss_name} seed={run.seed} max_weight={run.max_weight}: {exc}"
            )
            if stop_on_error:
                break

    return {
        "completed": successful + skipped,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "total": len(runs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase1.5 robustness experiments with a Drive-friendly Colab layout."
    )
    parser.add_argument(
        "--matrix-mode",
        choices=("light", "full"),
        default="light",
        help="Light: capped seeds plus one seed-42 no-cap comparison. Full: full seed x weight matrix.",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=str(DEFAULT_DRIVE_ROOT),
        help="Mounted Google Drive root used to derive outputs and checkpoints.",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=",".join(EXPERIMENT_LOSS_NAMES),
        help="Comma-separated loss list.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--nocap-seed",
        type=int,
        default=42,
        help="Seed used for the extra no-cap comparison in light mode.",
    )
    parser.add_argument("--data-dir", type=str, default=".", help="CSV data directory.")
    parser.add_argument(
        "--pattern", type=str, default="*.csv", help="Glob pattern for CSV files."
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        help="Lookback window for Feature Set X1.",
    )
    parser.add_argument(
        "--best-config-path",
        type=str,
        default="best_hyperparameters.txt",
        help="Path to the locked best hyperparameter config.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="1990-01",
        help="Inclusive train period start (YYYY-MM).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="1994-12",
        help="Inclusive train period end (YYYY-MM).",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="1995-01",
        help="Inclusive test period start (YYYY-MM).",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=24,
        help="Number of consecutive test months.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Training epochs for each run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size for each run.",
    )
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        help="Resume policy forwarded to each single-loss runner.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-run outputs that are already complete.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed run.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    losses = parse_losses(args.losses)
    seeds = parse_seeds(args.seeds)
    results = run_phase15_robustness(
        losses=losses,
        seeds=seeds,
        matrix_mode=args.matrix_mode,
        drive_root=Path(args.drive_root),
        data_dir=Path(args.data_dir),
        pattern=args.pattern,
        lookback_months=args.lookback_months,
        best_config_path=Path(args.best_config_path),
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_months=args.test_months,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        resume_mode=args.resume_mode,
        nocap_seed=args.nocap_seed,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error,
    )
    print(
        "Phase1.5 robustness runs "
        f"completed={results['completed']} successful={results['successful']} "
        f"skipped={results['skipped']} failed={results['failed']} total={results['total']}"
    )
    print(
        "Outputs root:",
        (Path(args.drive_root) / "outputs" / "phase1_5_robustness" / args.matrix_mode),
    )


if __name__ == "__main__":
    main()
