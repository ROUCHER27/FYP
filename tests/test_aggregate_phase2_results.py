import json
import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "aggregate_phase2_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_phase2_results", MODULE_PATH)
aggregate_phase2_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate_phase2_results)
collect_run_results = aggregate_phase2_results.collect_run_results


def test_collect_run_results_reads_existing_long_short_summary_schema(tmp_path: Path):
    loss_name = "imadl_m2_alpha05"
    output_dir = (
        tmp_path / "phase2" / "results" / f"{loss_name}_seed42_cap05"
    )
    output_dir.mkdir(parents=True)
    (output_dir / f"sanity_summary_{loss_name}.json").write_text(
        json.dumps(
            {
                "loss": loss_name,
                "avg_medse": 0.012,
                "avg_r2": 0.34,
                "avg_long_short": 0.056,
                "long_short_cumulative_return": 0.78,
                "long_short_std": 0.09,
                "long_short_sharpe": 1.23,
            }
        )
    )

    df = collect_run_results(
        tmp_path,
        losses=[loss_name],
        seeds=[42],
        weight_caps=[0.05],
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["cumulative_return"] == 0.78
    assert row["sharpe_ratio"] == 1.23
    assert row["avg_monthly_return"] == 0.056
    assert row["std_monthly_return"] == 0.09
    assert row["avg_r2"] == 0.34
    assert not pd.isna(row["cumulative_return"])
    assert not pd.isna(row["sharpe_ratio"])
