import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FYP")
RAW_COLUMNS = [
    "loss",
    "seed",
    "max_weight",
    "max_weight_label",
    "avg_long_short",
    "long_short_cumulative_return",
    "long_short_sharpe",
    "summary_path",
    "run_spec_path",
]


def format_max_weight_label(max_weight: float | None) -> str:
    if max_weight is None:
        return "None"
    return f"{float(max_weight):.2f}"


def resolve_matrix_root(
    matrix_root: str | None,
    drive_root: str,
    matrix_mode: str,
) -> Path:
    if matrix_root:
        return Path(matrix_root)
    return Path(drive_root) / "outputs" / "phase1_5_robustness" / matrix_mode


def collect_run_rows(matrix_root: Path) -> pd.DataFrame:
    matrix_root = Path(matrix_root)
    runs_root = matrix_root / "runs"
    checkpoint_root = matrix_root / "checkpoints"
    rows = []

    if not runs_root.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)

    for summary_path in sorted(runs_root.rglob("sanity_summary_*.json")):
        relative_run_dir = summary_path.parent.relative_to(runs_root)
        run_spec_path = checkpoint_root / relative_run_dir / "run_spec.json"
        if not run_spec_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            run_spec = json.loads(run_spec_path.read_text())
        except json.JSONDecodeError:
            continue

        max_weight = run_spec.get("max_weight")
        rows.append(
            {
                "loss": summary.get("loss", run_spec.get("loss_name")),
                "seed": int(run_spec["seed"]),
                "max_weight": max_weight,
                "max_weight_label": format_max_weight_label(max_weight),
                "avg_long_short": summary.get("avg_long_short"),
                "long_short_cumulative_return": summary.get(
                    "long_short_cumulative_return"
                ),
                "long_short_sharpe": summary.get("long_short_sharpe"),
                "summary_path": str(summary_path),
                "run_spec_path": str(run_spec_path),
            }
        )

    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)

    return pd.DataFrame(rows).sort_values(
        ["loss", "seed", "max_weight_label"], na_position="last"
    ).reset_index(drop=True)


def build_grouped_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "loss",
                "max_weight_label",
                "runs",
                "sharpe_mean",
                "sharpe_std",
                "cumret_mean",
                "cumret_std",
                "avg_ls_mean",
                "avg_ls_std",
            ]
        )
    grouped = (
        raw_df.groupby(["loss", "max_weight_label"], dropna=False)
        .agg(
            runs=("seed", "count"),
            sharpe_mean=("long_short_sharpe", "mean"),
            sharpe_std=("long_short_sharpe", "std"),
            cumret_mean=("long_short_cumulative_return", "mean"),
            cumret_std=("long_short_cumulative_return", "std"),
            avg_ls_mean=("avg_long_short", "mean"),
            avg_ls_std=("avg_long_short", "std"),
        )
        .reset_index()
    )
    return grouped.sort_values(["max_weight_label", "loss"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase1.5 robustness outputs into raw and grouped CSV tables."
    )
    parser.add_argument(
        "--matrix-root",
        type=str,
        default=None,
        help="Optional explicit Phase1.5 matrix root. Defaults to <drive-root>/outputs/phase1_5_robustness/<matrix-mode>.",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=str(DEFAULT_DRIVE_ROOT),
        help="Mounted Google Drive root used when --matrix-root is omitted.",
    )
    parser.add_argument(
        "--matrix-mode",
        choices=("light", "full"),
        default="light",
        help="Matrix mode used to derive the default Phase1.5 root.",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Optional explicit CSV path for the raw per-run table.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional explicit CSV path for the grouped robustness summary.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    matrix_root = resolve_matrix_root(args.matrix_root, args.drive_root, args.matrix_mode)
    raw_df = collect_run_rows(matrix_root)
    summary_df = build_grouped_summary(raw_df)

    raw_output = (
        Path(args.raw_output)
        if args.raw_output
        else matrix_root / "phase15_raw_runs.csv"
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else matrix_root / "phase15_grouped_summary.csv"
    )

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"Raw runs CSV: {raw_output}")
    print(f"Grouped summary CSV: {summary_output}")


if __name__ == "__main__":
    main()
