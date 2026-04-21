import json
import subprocess
from pathlib import Path

import pandas as pd

from run_all_experiments import is_loss_complete, run_experiments


def _write_success_outputs(output_dir: Path, loss_name: str) -> None:
    metrics_path = output_dir / f"sanity_metrics_{loss_name}.csv"
    metrics_path.write_text("month,mse,medse,long_short_return\n1995-01,1.0,0.8,0.1\n")

    summary = {
        "loss": loss_name,
        "avg_mse": 1.0,
        "avg_medse": 0.8,
        "avg_r2": 0.1,
        "avg_directional_accuracy": 0.6,
        "avg_sign_mismatch_large_y": 0.4,
        "avg_long_short": 0.1,
        "long_short_cumulative_return": 0.1,
        "long_short_std": 0.05,
        "long_short_sharpe": 2.0,
    }
    (output_dir / f"sanity_summary_{loss_name}.json").write_text(json.dumps(summary))
    (output_dir / f"{loss_name}_loss_curve.png").write_bytes(b"png")
    (output_dir / f"{loss_name}_returns_curve.png").write_bytes(b"png")


def test_is_loss_complete_requires_all_artifacts_and_valid_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    _write_success_outputs(output_dir, "mse")
    assert is_loss_complete(output_dir, "mse") is True

    (output_dir / "sanity_summary_mse.json").write_text("{bad json")
    assert is_loss_complete(output_dir, "mse") is False


def test_run_experiments_skips_existing_losses(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_success_outputs(output_dir, "mse")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        _write_success_outputs(output_dir, "medse")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = run_experiments(
        losses=["mse", "medse"],
        output_dir=output_dir,
        test_months=2,
        max_epochs=1,
        skip_existing=True,
    )

    assert len(calls) == 1
    assert Path(calls[0][1]).name == "run_sanity_check_medse.py"
    comparison = pd.read_csv(output_dir / "all_losses_comparison.csv")
    assert set(comparison["loss"]) == {"mse", "medse"}
    assert set(results) == {"mse", "medse"}


def test_run_experiments_excludes_failed_losses_from_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    def fake_run(cmd, check):
        runner_name = Path(cmd[1]).name
        if runner_name == "run_sanity_check_hybrid_add.py":
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        _write_success_outputs(output_dir, "hybrid_mul")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = run_experiments(
        losses=["hybrid_add", "hybrid_mul"],
        output_dir=output_dir,
        test_months=2,
        max_epochs=1,
        skip_existing=False,
        stop_on_error=False,
    )

    comparison = pd.read_csv(output_dir / "all_losses_comparison.csv")
    assert list(comparison["loss"]) == ["hybrid_mul"]
    assert "error" in results["hybrid_add"]
    assert results["hybrid_mul"]["loss"] == "hybrid_mul"
