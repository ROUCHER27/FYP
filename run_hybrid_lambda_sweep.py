import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class VariantSpec:
    base_loss: str
    loss_kwargs: Dict[str, float]


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "A1": VariantSpec("hybrid_add", {"lambda_dir": 5.0, "lambda_hub": 1.0}),
    "A2": VariantSpec("hybrid_add", {"lambda_dir": 10.0, "lambda_hub": 1.0}),
    "A3": VariantSpec("hybrid_add", {"lambda_dir": 1.0, "lambda_hub": 0.1}),
    "A4": VariantSpec("hybrid_add", {"lambda_dir": 5.0, "lambda_hub": 0.1}),
    "A5": VariantSpec("hybrid_add", {"lambda_dir": 10.0, "lambda_hub": 0.1}),
    "M1": VariantSpec("hybrid_mul", {"lambda_dir": 2.0}),
    "M2": VariantSpec("hybrid_mul", {"lambda_dir": 5.0}),
    "M3": VariantSpec("hybrid_mul", {"lambda_dir": 0.5}),
    "M4": VariantSpec("hybrid_mul", {"lambda_dir": 0.1}),
}

PRESET_VARIANTS = {
    "minimal": ["A4", "A5", "M2"],
    "full": list(VARIANT_SPECS.keys()),
}

RUNNER_BY_LOSS = {
    "hybrid_add": "run_sanity_check_hybrid_add.py",
    "hybrid_mul": "run_sanity_check_hybrid_mul.py",
}

COMPARISON_COLUMNS = [
    "variant_id",
    "base_loss",
    "lambda_dir",
    "lambda_hub",
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


def resolve_runner_path(base_loss: str) -> Path:
    runner_name = RUNNER_BY_LOSS[base_loss]
    runner_path = Path(__file__).resolve().parent / runner_name
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")
    return runner_path


def validate_runner_paths(variant_ids: List[str]) -> None:
    for variant_id in variant_ids:
        variant = VARIANT_SPECS[variant_id]
        resolve_runner_path(variant.base_loss)


def resolve_variant_ids(preset: str, variants: str | None) -> List[str]:
    if variants:
        variant_ids = [token.strip().upper() for token in variants.split(",") if token.strip()]
    else:
        variant_ids = PRESET_VARIANTS[preset]
    unsupported = sorted(set(variant_ids) - set(VARIANT_SPECS))
    if unsupported:
        raise ValueError(
            f"Unsupported variants: {', '.join(unsupported)}. "
            f"Supported variants: {', '.join(VARIANT_SPECS)}"
        )
    return variant_ids


def derive_variant_paths(output_root: Path, variant_id: str) -> Dict[str, Path]:
    return {
        "output_dir": Path(output_root) / "runs" / variant_id,
        "comparison_csv": Path(output_root) / "lambda_sweep_comparison.csv",
    }


def build_command_for_variant(
    variant_id: str,
    variant: VariantSpec,
    output_root: Path,
    best_config_path: Path,
    checkpoint_root: Path,
    archive_root: Path | None,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
) -> List[str]:
    paths = derive_variant_paths(output_root, variant_id)
    command = [
        sys.executable,
        str(resolve_runner_path(variant.base_loss)),
        "--output-dir",
        str(paths["output_dir"]),
        "--checkpoint-dir",
        str(Path(checkpoint_root) / variant_id),
        "--best-config-path",
        str(best_config_path),
        "--loss-kwargs",
        json.dumps(variant.loss_kwargs, sort_keys=True),
        "--test-months",
        str(test_months),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--resume-mode",
        resume_mode,
    ]
    if archive_root is not None:
        command.extend(["--archive-root", str(Path(archive_root) / variant_id)])
    return command


def load_variant_summary(output_dir: Path, variant: VariantSpec) -> Dict[str, float]:
    summary_path = Path(output_dir) / f"sanity_summary_{variant.base_loss}.json"
    try:
        return json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read summary for {variant.base_loss}: {summary_path}") from exc


def run_lambda_sweep(
    output_root: Path,
    variant_ids: List[str],
    best_config_path: Path,
    checkpoint_root: Path,
    archive_root: Path | None,
    test_months: int,
    max_epochs: int,
    batch_size: int,
    resume_mode: str,
    skip_existing: bool,
    stop_on_error: bool,
) -> Path:
    rows = []
    output_root = Path(output_root)
    validate_runner_paths(variant_ids)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    if archive_root is not None:
        Path(archive_root).mkdir(parents=True, exist_ok=True)

    for variant_id in variant_ids:
        variant = VARIANT_SPECS[variant_id]
        paths = derive_variant_paths(output_root, variant_id)
        summary_path = paths["output_dir"] / f"sanity_summary_{variant.base_loss}.json"
        if not (skip_existing and summary_path.exists()):
            cmd = build_command_for_variant(
                variant_id=variant_id,
                variant=variant,
                output_root=output_root,
                best_config_path=best_config_path,
                checkpoint_root=checkpoint_root,
                archive_root=archive_root,
                test_months=test_months,
                max_epochs=max_epochs,
                batch_size=batch_size,
                resume_mode=resume_mode,
            )
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                if stop_on_error:
                    raise
                continue

        if not summary_path.exists():
            if stop_on_error:
                raise FileNotFoundError(f"Missing summary for variant {variant_id}: {summary_path}")
            continue

        try:
            summary = load_variant_summary(paths["output_dir"], variant)
        except ValueError:
            if stop_on_error:
                raise
            continue
        rows.append(
            {
                "variant_id": variant_id,
                "base_loss": variant.base_loss,
                "lambda_dir": variant.loss_kwargs["lambda_dir"],
                "lambda_hub": variant.loss_kwargs.get("lambda_hub"),
                **summary,
            }
        )

    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    comparison.to_csv(output_root / "lambda_sweep_comparison.csv", index=False)
    return output_root / "lambda_sweep_comparison.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hybrid lambda sweep with resumable per-variant outputs."
    )
    parser.add_argument("--preset", choices=sorted(PRESET_VARIANTS), default="minimal")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated subset such as A4,A5,M2. Overrides --preset when provided.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="doc/phase1/lambda_sweep",
        help="Root directory for per-variant run outputs and the comparison CSV.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="sanity_outputs/lambda_sweep_checkpoints",
        help="Root directory for per-variant checkpoint state.",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Optional root copied to persistent storage such as Google Drive.",
    )
    parser.add_argument(
        "--best-config-path",
        type=str,
        default="best_hyperparameters.txt",
        help="Best hyperparameter file forwarded to each single-loss runner.",
    )
    parser.add_argument("--test-months", type=int, default=24)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume-mode", type=str, default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    variant_ids = resolve_variant_ids(args.preset, args.variants)
    comparison_path = run_lambda_sweep(
        output_root=Path(args.output_root),
        variant_ids=variant_ids,
        best_config_path=Path(args.best_config_path),
        checkpoint_root=Path(args.checkpoint_root),
        archive_root=Path(args.archive_root) if args.archive_root else None,
        test_months=args.test_months,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        resume_mode=args.resume_mode,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error,
    )
    print(f"Completed variants: {', '.join(variant_ids)}")
    print(f"Comparison table: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
