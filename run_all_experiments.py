import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from Model_Train.losses import EXPERIMENT_LOSS_NAMES


SUMMARY_COLUMNS = [
    "loss",
    "avg_mse",
    "avg_medse",
    "avg_r2",
    "avg_directional_accuracy",
    "avg_sign_mismatch_large_y",
    "avg_long_short",
    "long_short_cumulative_return",
    "long_short_std",
    "long_short_sharpe",
]

RUNNER_BY_LOSS = {
    "mse": "run_sanity_check_mse.py",
    "medse": "run_sanity_check_medse.py",
    "gmadl": "run_sanity_check_gmadl.py",
    "imadl": "run_sanity_check_imadl.py",
    "dirhuber": "run_sanity_check_dirhuber.py",
    "hybrid_add": "run_sanity_check_hybrid_add.py",
    "hybrid_mul": "run_sanity_check_hybrid_mul.py",
}


def parse_losses(value: str) -> List[str]:
    losses = [token.strip().lower() for token in value.split(",") if token.strip()]
    unsupported = sorted(set(losses) - set(EXPERIMENT_LOSS_NAMES))
    if unsupported:
        raise ValueError(
            f"Unsupported losses: {', '.join(unsupported)}. "
            f"Supported losses: {', '.join(EXPERIMENT_LOSS_NAMES)}"
        )
    return losses


def expected_output_paths(output_dir: Path, loss_name: str) -> Dict[str, Path]:
    return {
        "metrics": output_dir / f"sanity_metrics_{loss_name}.csv",
        "summary": output_dir / f"sanity_summary_{loss_name}.json",
        "loss_curve": output_dir / f"{loss_name}_loss_curve.png",
        "returns_curve": output_dir / f"{loss_name}_returns_curve.png",
    }


def load_summary(output_dir: Path, loss_name: str) -> Dict[str, float]:
    summary_path = expected_output_paths(output_dir, loss_name)["summary"]
    payload = json.loads(summary_path.read_text())
    missing = [key for key in SUMMARY_COLUMNS if key not in payload]
    if missing:
        raise ValueError(
            f"Summary for {loss_name} missing required keys: {', '.join(missing)}"
        )
    if payload["loss"] != loss_name:
        raise ValueError(
            f"Summary loss mismatch for {loss_name}: found {payload['loss']}"
        )
    return payload


def is_loss_complete(output_dir: Path, loss_name: str) -> bool:
    paths = expected_output_paths(output_dir, loss_name)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        load_summary(output_dir, loss_name)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def build_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for loss_name, result in results.items():
        if "error" in result:
            continue
        rows.append({column: result[column] for column in SUMMARY_COLUMNS})
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        "long_short_sharpe", ascending=False, na_position="last"
    )


def build_command(
    loss_name: str,
    output_dir: Path,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str | None = None,
    checkpoint_dir: Path | None = None,
) -> List[str]:
    command = [
        sys.executable,
        RUNNER_BY_LOSS[loss_name],
        "--output-dir",
        str(output_dir),
        "--test-months",
        str(test_months),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
    ]
    if resume_mode:
        command.extend(["--resume-mode", resume_mode])
    if checkpoint_dir:
        command.extend(["--checkpoint-dir", str(checkpoint_dir)])
    return command


def run_experiments(
    losses: Iterable[str],
    output_dir: Path,
    test_months: int,
    max_epochs: int,
    batch_size: int = 1024,
    skip_existing: bool = False,
    stop_on_error: bool = False,
    resume_mode: str | None = None,
    checkpoint_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for loss_name in losses:
        if skip_existing and is_loss_complete(output_dir, loss_name):
            results[loss_name] = load_summary(output_dir, loss_name)
            continue

        cmd = build_command(
            loss_name=loss_name,
            output_dir=output_dir,
            test_months=test_months,
            max_epochs=max_epochs,
            batch_size=batch_size,
            resume_mode=resume_mode,
            checkpoint_dir=checkpoint_dir,
        )
        try:
            subprocess.run(cmd, check=True)
            if not is_loss_complete(output_dir, loss_name):
                raise RuntimeError(
                    f"Loss {loss_name} finished without producing complete outputs."
                )
            results[loss_name] = load_summary(output_dir, loss_name)
        except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
            results[loss_name] = {"loss": loss_name, "error": str(exc)}
            if stop_on_error:
                break

    comparison = build_comparison_table(results)
    comparison.to_csv(output_dir / "all_losses_comparison.csv", index=False)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple sanity-check loss experiments with resumable output collection."
    )
    parser.add_argument(
        "--losses",
        type=str,
        default=",".join(EXPERIMENT_LOSS_NAMES),
        help="Comma-separated experiment losses to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sanity_outputs",
        help="Directory used by all experiment runners.",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=24,
        help="Number of test months passed to each runner.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Training epochs passed to each runner.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size passed to each runner.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip losses whose four output artifacts and summary schema already exist.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch immediately after the first failed loss.",
    )
    parser.add_argument(
        "--resume-mode",
        type=str,
        default=None,
        help="Resume policy forwarded to each single-loss runner.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory forwarded to each single-loss runner.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    losses = parse_losses(args.losses)
    results = run_experiments(
        losses=losses,
        output_dir=Path(args.output_dir),
        test_months=args.test_months,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error,
        resume_mode=args.resume_mode,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    completed = [name for name, result in results.items() if "error" not in result]
    failed = [name for name, result in results.items() if "error" in result]
    print(f"Completed losses: {', '.join(completed) if completed else 'none'}")
    print(f"Failed losses: {', '.join(failed) if failed else 'none'}")
    print(f"Comparison table: {(Path(args.output_dir) / 'all_losses_comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
