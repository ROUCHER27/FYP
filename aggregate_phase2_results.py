#!/usr/bin/env python3
"""
Aggregate Phase 2 experiment results.

Collects results from all Phase 2 runs and generates summary statistics.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def collect_run_results(
    drive_root: Path,
    losses: List[str],
    seeds: List[int],
    weight_caps: List[float | None],
) -> pd.DataFrame:
    """
    Collect results from all Phase 2 runs.

    Args:
        drive_root: Google Drive root path
        losses: List of loss function names
        seeds: List of random seeds
        weight_caps: List of weight caps (None for uncapped)

    Returns:
        DataFrame with per-run results
    """
    results = []

    for loss_name in losses:
        for seed in seeds:
            for max_weight in weight_caps:
                # Derive paths
                if max_weight is None:
                    cap_tag = "nocap"
                else:
                    cap_tag = f"cap{int(round(max_weight * 100)):02d}"

                run_id = f"{loss_name}_seed{seed}_{cap_tag}"
                output_dir = drive_root / "phase2" / "results" / run_id
                summary_file = output_dir / f"sanity_summary_{loss_name}.json"

                # Check if run exists
                if not summary_file.exists():
                    print(f"Warning: Missing results for {run_id}")
                    continue

                # Load summary
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)

                    # Extract key metrics
                    results.append({
                        "loss": loss_name,
                        "seed": seed,
                        "max_weight": max_weight,
                        "cap_tag": cap_tag,
                        "cumulative_return": summary.get("cumulative_return", None),
                        "sharpe_ratio": summary.get("sharpe_ratio", None),
                        "avg_monthly_return": summary.get("avg_monthly_return", None),
                        "std_monthly_return": summary.get("std_monthly_return", None),
                        "max_drawdown": summary.get("max_drawdown", None),
                        "win_rate": summary.get("win_rate", None),
                        "avg_r2": summary.get("avg_r2", None),
                        "final_train_loss": summary.get("final_train_loss", None),
                    })

                except Exception as e:
                    print(f"Error loading {summary_file}: {e}")
                    continue

    return pd.DataFrame(results)


def compute_grouped_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute grouped statistics by loss function and weight cap.

    Args:
        df: DataFrame with per-run results

    Returns:
        DataFrame with grouped statistics
    """
    # Group by loss and cap
    grouped = df.groupby(["loss", "cap_tag"]).agg({
        "cumulative_return": ["mean", "std", "min", "max"],
        "sharpe_ratio": ["mean", "std", "min", "max"],
        "avg_monthly_return": ["mean", "std"],
        "std_monthly_return": ["mean", "std"],
        "max_drawdown": ["mean", "min"],
        "win_rate": ["mean", "std"],
        "avg_r2": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]

    # Compute coefficient of variation (CV) for Sharpe ratio
    grouped["sharpe_cv"] = (
        grouped["sharpe_ratio_std"] / grouped["sharpe_ratio_mean"].abs()
    )

    # Compute failure rate (Sharpe < 0)
    failure_counts = df[df["sharpe_ratio"] < 0].groupby(["loss", "cap_tag"]).size()
    total_counts = df.groupby(["loss", "cap_tag"]).size()
    failure_rate = (failure_counts / total_counts * 100).fillna(0)
    grouped["failure_rate"] = grouped.apply(
        lambda row: failure_rate.get((row["loss"], row["cap_tag"]), 0),
        axis=1
    )

    return grouped


def generate_summary_report(
    raw_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    output_path: Path,
):
    """Generate a text summary report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2 Experiment Results Summary\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write(f"Total runs: {len(raw_df)}\n")
        f.write(f"Unique losses: {raw_df['loss'].nunique()}\n")
        f.write(f"Seeds: {sorted(raw_df['seed'].unique())}\n")
        f.write(f"Weight caps: {sorted(raw_df['max_weight'].unique())}\n\n")

        # Top performers by Sharpe ratio
        f.write("-" * 80 + "\n")
        f.write("Top 5 Performers by Mean Sharpe Ratio\n")
        f.write("-" * 80 + "\n")
        top5 = grouped_df.nlargest(5, "sharpe_ratio_mean")
        for idx, row in top5.iterrows():
            f.write(f"\n{row['loss']} ({row['cap_tag']})\n")
            f.write(f"  Mean Sharpe:  {row['sharpe_ratio_mean']:.4f} ± {row['sharpe_ratio_std']:.4f}\n")
            f.write(f"  CV:           {row['sharpe_cv']:.4f}\n")
            f.write(f"  Failure Rate: {row['failure_rate']:.1f}%\n")
            f.write(f"  Cum Return:   {row['cumulative_return_mean']:.2f}% ± {row['cumulative_return_std']:.2f}%\n")

        # Lowest CV (most stable)
        f.write("\n" + "-" * 80 + "\n")
        f.write("Top 5 Most Stable (Lowest CV)\n")
        f.write("-" * 80 + "\n")
        stable5 = grouped_df.nsmallest(5, "sharpe_cv")
        for idx, row in stable5.iterrows():
            f.write(f"\n{row['loss']} ({row['cap_tag']})\n")
            f.write(f"  CV:           {row['sharpe_cv']:.4f}\n")
            f.write(f"  Mean Sharpe:  {row['sharpe_ratio_mean']:.4f} ± {row['sharpe_ratio_std']:.4f}\n")
            f.write(f"  Failure Rate: {row['failure_rate']:.1f}%\n")

        # Comparison with Phase 1.5 baselines
        f.write("\n" + "-" * 80 + "\n")
        f.write("Comparison with Phase 1.5 Baselines\n")
        f.write("-" * 80 + "\n")
        f.write("IMADL Baseline: Sharpe=0.464, CV=0.892, Failure=0%\n")
        f.write("M2 Baseline:    Sharpe=0.914, CV=1.396, Failure=33%\n\n")

        # Count improvements
        better_than_imadl = grouped_df[grouped_df["sharpe_ratio_mean"] > 0.464]
        better_than_m2 = grouped_df[grouped_df["sharpe_ratio_mean"] > 0.914]
        stable_than_imadl = grouped_df[grouped_df["sharpe_cv"] < 0.892]

        f.write(f"Losses better than IMADL (Sharpe > 0.464): {len(better_than_imadl)}\n")
        f.write(f"Losses better than M2 (Sharpe > 0.914): {len(better_than_m2)}\n")
        f.write(f"Losses more stable than IMADL (CV < 0.892): {len(stable_than_imadl)}\n")

        f.write("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 2 experiment results"
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default="/content/drive/MyDrive/FYP",
        help="Google Drive root path",
    )
    parser.add_argument(
        "--losses",
        type=str,
        help="Comma-separated loss names (default: all Phase 2 losses)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,52,62",
        help="Comma-separated seeds (default: 42,52,62)",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default="0.05",
        help="Comma-separated weight caps, use 'None' for uncapped (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: {drive_root}/phase2)",
    )

    args = parser.parse_args()

    # Parse arguments
    drive_root = Path(args.drive_root)

    # Default to all Phase 2 losses
    if args.losses:
        losses = [l.strip() for l in args.losses.split(",")]
    else:
        losses = [
            "imadl_m2_alpha02", "imadl_m2_alpha03", "imadl_m2_alpha04",
            "imadl_m2_alpha05", "imadl_m2_alpha06", "imadl_m2_alpha07",
            "imadl_m2_alpha08", "imadl_gmadl_beta03", "imadl_gmadl_beta05",
            "imadl_gmadl_beta07", "m2_robust_gamma001", "m2_robust_gamma01",
            "m2_robust_gamma10", "adaptive_lambda10", "adaptive_lambda50",
            "adaptive_lambda100",
        ]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Parse weight caps
    weight_caps = []
    for cap in args.caps.split(","):
        cap = cap.strip()
        if cap.lower() == "none":
            weight_caps.append(None)
        else:
            weight_caps.append(float(cap))

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = drive_root / "phase2"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting Phase 2 results...")
    print(f"Losses: {len(losses)}")
    print(f"Seeds: {seeds}")
    print(f"Weight caps: {weight_caps}")
    print("-" * 80)

    # Collect results
    raw_df = collect_run_results(drive_root, losses, seeds, weight_caps)

    if raw_df.empty:
        print("Error: No results found!")
        return

    print(f"\nCollected {len(raw_df)} runs")

    # Compute grouped statistics
    grouped_df = compute_grouped_statistics(raw_df)

    # Save results
    raw_output = output_dir / "phase2_raw_runs.csv"
    grouped_output = output_dir / "phase2_grouped_summary.csv"
    report_output = output_dir / "phase2_summary_report.txt"

    raw_df.to_csv(raw_output, index=False)
    print(f"✓ Saved raw results: {raw_output}")

    grouped_df.to_csv(grouped_output, index=False)
    print(f"✓ Saved grouped summary: {grouped_output}")

    generate_summary_report(raw_df, grouped_df, report_output)
    print(f"✓ Saved summary report: {report_output}")

    print("\n" + "=" * 80)
    print("Aggregation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
