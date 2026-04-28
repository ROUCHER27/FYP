#!/usr/bin/env python3
"""
Analyze loss-scale diagnostics from available sanity outputs.

Per-batch component CSVs are preferred. If they are absent, the script falls
back to a metrics proxy from sanity_metrics_*.csv so P0 diagnostics still
produce a CSV without modifying the training loop.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


COMPONENT_GLOB = "*loss_scale*.csv"
METRICS_GLOB = "sanity_metrics_*.csv"
SKIP_COLUMNS = {"epoch", "batch", "step", "month", "loss"}
PROXY_COLUMNS = ("mse", "medse", "long_short_return", "directional_accuracy", "r2")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_abs_by_column(rows: list[dict[str, str]], columns: list[str]) -> dict[str, float]:
    values: dict[str, list[float]] = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            try:
                value = abs(float(row.get(column, "")))
            except ValueError:
                continue
            if math.isfinite(value):
                values[column].append(value)
    return {
        column: sum(items) / len(items)
        for column, items in values.items()
        if items and sum(items) / len(items) > 0
    }


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = [column for column in rows[0] if column not in SKIP_COLUMNS]
    usable: list[str] = []
    for column in columns:
        for row in rows:
            try:
                float(row.get(column, ""))
            except ValueError:
                continue
            usable.append(column)
            break
    return usable


def summarize_means(source: Path, diagnostic_type: str, means: dict[str, float], loss: str | None = None) -> dict[str, Any] | None:
    if not means:
        return None
    smallest = min(means, key=means.get)
    largest = max(means, key=means.get)
    min_value = means[smallest]
    max_value = means[largest]
    ratio = max_value / min_value if min_value > 0 else math.inf
    row: dict[str, Any] = {
        "source": str(source),
        "diagnostic_type": diagnostic_type,
        "component_count": len(means),
        "min_abs_mean": min_value,
        "max_abs_mean": max_value,
        "scale_ratio": ratio,
        "largest_component": largest,
        "smallest_component": smallest,
        "scale_class": classify_ratio(ratio),
    }
    if loss is not None:
        row["loss"] = loss
    return row


def summarize_components(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob(COMPONENT_GLOB)):
        csv_rows = read_rows(path)
        means = mean_abs_by_column(csv_rows, numeric_columns(csv_rows))
        summary = summarize_means(path, "component", means)
        if summary:
            rows.append(summary)
    return rows


def infer_loss_name(path: Path) -> str:
    prefix = "sanity_metrics_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem


def summarize_metric_proxies(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob(METRICS_GLOB)):
        csv_rows = read_rows(path)
        columns = [column for column in PROXY_COLUMNS if csv_rows and column in csv_rows[0]]
        means = mean_abs_by_column(csv_rows, columns)
        summary = summarize_means(path, "metric_proxy", means, loss=infer_loss_name(path))
        if summary:
            rows.append(summary)
    return rows


def classify_ratio(ratio: float) -> str:
    if not math.isfinite(ratio):
        return "invalid"
    if ratio < 5:
        return "balanced"
    if ratio <= 10:
        return "moderate_imbalance"
    return "severe_imbalance"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "loss",
        "source",
        "diagnostic_type",
        "component_count",
        "min_abs_mean",
        "max_abs_mean",
        "scale_ratio",
        "scale_class",
        "largest_component",
        "smallest_component",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ModuleNotFoundError:
        print("matplotlib not installed; skipped loss_scale_ratios.png")
        return
    labels = [str(row.get("loss") or Path(str(row["source"])).stem) for row in rows]
    ratios = [float(row["scale_ratio"]) for row in rows]
    plt.figure(figsize=(max(8, len(rows) * 0.7), 4))
    plt.bar(labels, ratios)
    plt.axhline(5.0, color="orange", linestyle="--", linewidth=1, label="5x")
    plt.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10x")
    plt.ylabel("Scale ratio")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_scale_ratios.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze available loss-scale diagnostics and sanity outputs.")
    parser.add_argument("--input-root", default="sanity_outputs")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "loss_scale_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    component_rows = summarize_components(input_root)
    proxy_rows = summarize_metric_proxies(input_root)
    rows = component_rows + proxy_rows
    summary_path = output_dir / "loss_scale_summary.csv"
    write_csv(summary_path, rows)
    plot_summary(rows, output_dir)
    print(f"Wrote {summary_path}")
    if (output_dir / "loss_scale_ratios.png").exists():
        print(f"Wrote {output_dir / 'loss_scale_ratios.png'}")
    if not component_rows:
        print("No per-batch component diagnostics found; used sanity metric proxy rows where available.")


if __name__ == "__main__":
    main()
