#!/usr/bin/env python3
"""Aggregate sanity-check batch results into raw and grouped CSV reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHARPE_KEYS = ("long_short_sharpe", "sharpe_ratio")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_caps(value: str) -> list[float | None]:
    caps: list[float | None] = []
    for item in parse_csv(value):
        caps.append(None if item.lower() == "none" else float(item))
    return caps


def cap_tag(max_weight: float | None) -> str:
    return "nocap" if max_weight is None else f"cap{int(round(max_weight * 100)):02d}"


def first(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def load_rows(results_root: Path, losses: list[str], seeds: list[int], caps: list[float | None]) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for loss in losses:
        for seed in seeds:
            for cap in caps:
                tag = cap_tag(cap)
                run_id = f"{loss}_seed{seed}_{tag}"
                summary_file = results_root / run_id / f"sanity_summary_{loss}.json"
                if not summary_file.exists():
                    print(f"Missing {summary_file}")
                    continue
                summary = json.loads(summary_file.read_text())
                rows.append(
                    {
                        "run_id": run_id,
                        "loss": loss,
                        "seed": seed,
                        "max_weight": cap,
                        "cap_tag": tag,
                        "avg_mse": first(summary, "avg_mse"),
                        "avg_medse": first(summary, "avg_medse"),
                        "avg_r2": first(summary, "avg_r2"),
                        "avg_directional_accuracy": first(summary, "avg_directional_accuracy"),
                        "avg_long_short": first(summary, "avg_long_short", "avg_monthly_return"),
                        "long_short_cumulative_return": first(
                            summary, "long_short_cumulative_return", "cumulative_return"
                        ),
                        "long_short_std": first(summary, "long_short_std", "std_monthly_return"),
                        "long_short_sharpe": first(summary, *SHARPE_KEYS),
                    }
                )
    return pd.DataFrame(rows)


def grouped_stats(raw: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    numeric_cols = [
        "avg_mse",
        "avg_medse",
        "avg_r2",
        "avg_directional_accuracy",
        "avg_long_short",
        "long_short_cumulative_return",
        "long_short_std",
        "long_short_sharpe",
    ]
    for col in numeric_cols:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    grouped = (
        raw.groupby(["loss", "cap_tag"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            sharpe_mean=("long_short_sharpe", "mean"),
            sharpe_std=("long_short_sharpe", "std"),
            sharpe_min=("long_short_sharpe", "min"),
            sharpe_max=("long_short_sharpe", "max"),
            cumulative_return_mean=("long_short_cumulative_return", "mean"),
            avg_long_short_mean=("avg_long_short", "mean"),
            avg_r2_mean=("avg_r2", "mean"),
        )
        .reset_index()
    )
    grouped["sharpe_cv"] = grouped["sharpe_std"] / grouped["sharpe_mean"].abs()
    return grouped


def write_report(grouped: pd.DataFrame, output_path: Path) -> None:
    lines = ["Phase 2 Results Summary", "=" * 80, ""]
    if grouped.empty:
        lines.append("No runs found.")
    else:
        top = grouped.sort_values("sharpe_mean", ascending=False)
        lines.append("Top losses by mean Sharpe:")
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.loss} ({row.cap_tag}): Sharpe {row.sharpe_mean:.4f} "
                f"+/- {row.sharpe_std:.4f}, runs={row.runs}"
            )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2 batch output directories.")
    parser.add_argument("--results-root", default="/content/drive/MyDrive/FYP/phase2/results")
    parser.add_argument("--losses", required=True, help="Comma-separated loss names to collect.")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--caps", default="0.05")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_rows(results_root, parse_csv(args.losses), parse_seeds(args.seeds), parse_caps(args.caps))
    grouped = grouped_stats(raw) if not raw.empty else pd.DataFrame()
    raw_path = output_dir / "phase2_raw_runs.csv"
    grouped_path = output_dir / "phase2_grouped_summary.csv"
    report_path = output_dir / "phase2_summary_report.txt"
    raw.to_csv(raw_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    write_report(grouped, report_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {grouped_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
