import json
import subprocess
from pathlib import Path

import pandas as pd

import run_all_experiments
from run_all_experiments import build_command, is_loss_complete, run_experiments


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


def test_run_experiments_does_not_skip_partial_checkpoint_state(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mse_epoch5.pt").write_bytes(b"checkpoint")
    (output_dir / "sanity_metrics_mse.csv").write_text(
        "month,mse,medse,long_short_return\n1995-01,1.0,0.8,0.1\n"
    )

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        _write_success_outputs(output_dir, "mse")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = run_experiments(
        losses=["mse"],
        output_dir=output_dir,
        test_months=2,
        max_epochs=1,
        skip_existing=True,
        resume_mode="latest",
        checkpoint_dir=checkpoint_dir,
    )

    assert len(calls) == 1
    assert results["mse"]["loss"] == "mse"


def test_build_command_appends_resume_flags() -> None:
    checkpoint_dir = Path("/tmp/checkpoints")
    best_config_path = Path("/tmp/best_hyperparameters.txt")

    command = build_command(
        loss_name="medse",
        output_dir=Path("/tmp/outputs"),
        test_months=3,
        max_epochs=4,
        batch_size=512,
        best_config_path=best_config_path,
        resume_mode="latest",
        checkpoint_dir=checkpoint_dir,
    )

    assert "--best-config-path" in command
    assert str(best_config_path) in command
    assert command[-6:] == [
        "--best-config-path",
        str(best_config_path),
        "--resume-mode",
        "latest",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]


def test_main_passes_resume_args_to_run_experiments(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_experiments(**kwargs):
        captured.update(kwargs)
        return {"mse": {"loss": "mse"}}

    monkeypatch.setattr(run_all_experiments, "run_experiments", fake_run_experiments)
    monkeypatch.setattr(
        run_all_experiments.argparse.ArgumentParser,
        "parse_args",
        lambda self: run_all_experiments.argparse.Namespace(
            losses="mse",
            output_dir=str(tmp_path / "outputs"),
            test_months=6,
            max_epochs=10,
            batch_size=128,
            best_config_path=str(tmp_path / "best_hyperparameters.txt"),
            skip_existing=False,
            stop_on_error=False,
            resume_mode="latest",
            checkpoint_dir=str(tmp_path / "ckpts"),
        ),
    )

    run_all_experiments.main()

    assert captured["resume_mode"] == "latest"
    assert captured["checkpoint_dir"] == Path(tmp_path / "ckpts")
    assert captured["best_config_path"] == Path(tmp_path / "best_hyperparameters.txt")
