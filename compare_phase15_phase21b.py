#!/usr/bin/env python3
"""Compare Phase 2.1b summary results with Phase 1.5 Sharpe targets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


TARGETS = {
    "imadl": 0.464,
    "hybrid_mul": 0.914,
    "gmadl": 0.307,
}
ALIASES = {
    "m2": "hybrid_mul",
    "M2": "hybrid_mul",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def cap_tag(cap: float | None) -> str:
    return "nocap" if cap is None else f"cap{int(round(cap * 100)):02d}"


def parse_caps(value: str) -> list[float | None]:
    caps: list[float | None] = []
    for item in parse_csv(value):
        caps.append(None if item.lower() == "none" else float(item))
    return caps


def first(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def load_comparison(results_root: Path, losses: list[str], seeds: list[int], caps: list[float | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loss in losses:
        canonical_loss = ALIASES.get(loss, loss)
        target = TARGETS[canonical_loss]
        for seed in seeds:
            for cap in caps:
                run_id = f"{loss}_seed{seed}_{cap_tag(cap)}"
                summary_file = results_root / run_id / f"sanity_summary_{loss}.json"
                if not summary_file.exists():
                    rows.append(
                        {
                            "loss": loss,
                            "seed": seed,
                            "cap_tag": cap_tag(cap),
                            "target_sharpe": target,
                            "observed_sharpe": None,
                            "deviation_pct": None,
                            "flag_gt_15pct": True,
                            "status": "missing",
                            "summary_file": str(summary_file),
                        }
                    )
                    continue
                summary = json.loads(summary_file.read_text())
                observed = first(summary, "long_short_sharpe", "sharpe_ratio")
                deviation = None if observed is None else (float(observed) - target) / abs(target) * 100.0
                rows.append(
                    {
                        "loss": loss,
                        "seed": seed,
                        "cap_tag": cap_tag(cap),
                        "target_sharpe": target,
                        "observed_sharpe": observed,
                        "deviation_pct": deviation,
                        "flag_gt_15pct": True if deviation is None else abs(deviation) > 15.0,
                        "status": "ok",
                        "summary_file": str(summary_file),
                    }
                )
    return rows


def grouped_comparison(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw:
        if row["status"] != "ok" or row["observed_sharpe"] is None:
            continue
        buckets.setdefault((str(row["loss"]), str(row["cap_tag"])), []).append(row)
    grouped: list[dict[str, Any]] = []
    for (loss, tag), rows in sorted(buckets.items()):
        values = [float(row["observed_sharpe"]) for row in rows]
        target = float(rows[0]["target_sharpe"])
        mean_value = statistics.fmean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        deviation = (mean_value - target) / abs(target) * 100.0
        grouped.append(
            {
                "loss": loss,
                "cap_tag": tag,
                "runs": len(values),
                "target_sharpe": target,
                "mean_observed_sharpe": mean_value,
                "std_observed_sharpe": std_value,
                "deviation_pct": deviation,
                "flag_gt_15pct": abs(deviation) > 15.0,
            }
        )
    return grouped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 2.1b results with Phase 1.5 Sharpe targets.")
    parser.add_argument("--results-root", default="/content/drive/MyDrive/FYP/phase2_1b/results")
    parser.add_argument("--losses", default="imadl,gmadl,hybrid_mul")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--caps", default="0.05")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fail-on-deviation", action="store_true")
    args = parser.parse_args()

    losses = [ALIASES.get(loss, loss) for loss in parse_csv(args.losses)]
    unsupported = sorted(set(losses) - set(TARGETS))
    if unsupported:
        raise SystemExit(f"Unsupported comparison losses: {', '.join(unsupported)}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_comparison(Path(args.results_root), losses, parse_seeds(args.seeds), parse_caps(args.caps))
    grouped = grouped_comparison(raw)
    raw_path = output_dir / "phase21b_vs_phase15_raw.csv"
    grouped_path = output_dir / "phase21b_vs_phase15_grouped.csv"
    write_csv(raw_path, raw)
    write_csv(grouped_path, grouped)
    print(f"Wrote {raw_path}")
    print(f"Wrote {grouped_path}")
    if not grouped:
        print("No completed summaries found.")
    else:
        for row in grouped:
            print(
                f"{row['loss']} {row['cap_tag']}: mean Sharpe "
                f"{row['mean_observed_sharpe']:.4f} vs target {row['target_sharpe']:.4f}, "
                f"deviation {row['deviation_pct']:.2f}%, flag={row['flag_gt_15pct']}"
            )
    has_bad = any(bool(row["flag_gt_15pct"]) for row in raw + grouped)
    if args.fail_on_deviation and has_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
