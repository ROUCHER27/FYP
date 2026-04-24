import json
import subprocess
from pathlib import Path

import pandas as pd

import run_hybrid_lambda_sweep as sweep


def _write_variant_outputs(output_dir: Path, base_loss: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "loss": base_loss,
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
    (output_dir / f"sanity_summary_{base_loss}.json").write_text(json.dumps(summary))


def test_resolve_variant_ids_supports_minimal_and_full_presets() -> None:
    assert sweep.resolve_variant_ids("minimal", None) == ["A4", "A5", "M2"]
    assert sweep.resolve_variant_ids("full", None) == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "M1",
        "M2",
        "M3",
        "M4",
    ]


def test_build_command_for_variant_forwards_loss_kwargs_and_drive_paths(tmp_path: Path) -> None:
    variant = sweep.VARIANT_SPECS["A4"]
    paths = sweep.derive_variant_paths(tmp_path, "A4")

    command = sweep.build_command_for_variant(
        variant_id="A4",
        variant=variant,
        output_root=tmp_path,
        best_config_path=Path("/tmp/best_hyperparameters.txt"),
        checkpoint_root=tmp_path / "checkpoints",
        archive_root=tmp_path / "archive",
        test_months=24,
        max_epochs=20,
        batch_size=1024,
        resume_mode="auto",
    )

    assert Path(command[1]).name == "run_sanity_check_hybrid_add.py"
    assert "--loss-kwargs" in command
    assert json.loads(command[command.index("--loss-kwargs") + 1]) == variant.loss_kwargs
    assert str(paths["output_dir"]) in command
    assert str((tmp_path / "checkpoints" / "A4")) in command
    assert str((tmp_path / "archive" / "A4")) in command


def test_run_lambda_sweep_writes_comparison_csv(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        variant_id = Path(cmd[cmd.index("--output-dir") + 1]).name
        variant = sweep.VARIANT_SPECS[variant_id]
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        _write_variant_outputs(output_dir, variant.base_loss)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    comparison_path = sweep.run_lambda_sweep(
        output_root=tmp_path,
        variant_ids=["A4", "M2"],
        best_config_path=Path("/tmp/best_hyperparameters.txt"),
        checkpoint_root=tmp_path / "checkpoints",
        archive_root=tmp_path / "archive",
        test_months=24,
        max_epochs=20,
        batch_size=1024,
        resume_mode="auto",
        skip_existing=False,
        stop_on_error=False,
    )

    comparison = pd.read_csv(comparison_path)
    assert len(calls) == 2
    assert set(comparison["variant_id"]) == {"A4", "M2"}
    assert set(comparison["base_loss"]) == {"hybrid_add", "hybrid_mul"}
    assert set(comparison["lambda_dir"]) == {5.0}
