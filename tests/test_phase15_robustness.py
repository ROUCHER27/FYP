import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import aggregate_phase15_results as aggregate_phase15
import run_phase15_robustness as phase15


def _write_run_artifacts(
    root: Path,
    *,
    matrix_mode: str,
    loss_name: str,
    seed: int,
    max_weight: float | None,
    sharpe: float,
    cumulative_return: float,
    avg_long_short: float,
) -> None:
    cap_tag = phase15.cap_tag_from_weight(max_weight)
    run_dir = root / "outputs" / "phase1_5_robustness" / matrix_mode / "runs" / f"seed{seed}_{cap_tag}" / loss_name
    checkpoint_dir = root / "outputs" / "phase1_5_robustness" / matrix_mode / "checkpoints" / f"seed{seed}_{cap_tag}" / loss_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "loss": loss_name,
        "avg_mse": 1.0,
        "avg_medse": 0.5,
        "avg_r2": 0.1,
        "avg_directional_accuracy": 0.6,
        "avg_sign_mismatch_large_y": 0.3,
        "avg_long_short": avg_long_short,
        "long_short_cumulative_return": cumulative_return,
        "long_short_std": 0.2,
        "long_short_sharpe": sharpe,
    }
    run_spec = {
        "loss_name": loss_name,
        "seed": seed,
        "max_weight": max_weight,
    }

    (run_dir / f"sanity_summary_{loss_name}.json").write_text(json.dumps(summary))
    (checkpoint_dir / "run_spec.json").write_text(json.dumps(run_spec))


def test_build_run_matrix_light_mode_adds_seed42_nocap() -> None:
    runs = phase15.build_run_matrix(
        losses=["mse", "medse"],
        seeds=[42, 52, 62],
        matrix_mode="light",
        nocap_seed=42,
    )

    assert len(runs) == 8
    assert runs[0] == phase15.Phase15Run(loss_name="mse", seed=42, max_weight=0.05)
    assert runs[-1] == phase15.Phase15Run(loss_name="medse", seed=42, max_weight=None)


def test_build_run_matrix_full_mode_builds_cartesian_product() -> None:
    runs = phase15.build_run_matrix(
        losses=["mse", "medse"],
        seeds=[42, 52],
        matrix_mode="full",
        nocap_seed=42,
    )

    assert len(runs) == 8
    assert phase15.Phase15Run(loss_name="mse", seed=52, max_weight=None) in runs
    assert phase15.Phase15Run(loss_name="medse", seed=42, max_weight=0.05) in runs


def test_default_losses_cover_all_experiment_losses() -> None:
    parser = phase15.build_arg_parser()
    args = parser.parse_args([])

    assert args.losses == ",".join(phase15.EXPERIMENT_LOSS_NAMES)
    assert args.matrix_mode == "light"


def test_derive_paths_uses_drive_layout_and_normalizes_nocap() -> None:
    paths = phase15.derive_run_paths(
        drive_root=Path("/tmp/FYP"),
        matrix_mode="light",
        run=phase15.Phase15Run(loss_name="gmadl", seed=42, max_weight=None),
    )

    assert paths["base_root"] == Path("/tmp/FYP/outputs/phase1_5_robustness/light")
    assert paths["output_dir"] == Path(
        "/tmp/FYP/outputs/phase1_5_robustness/light/runs/seed42_nocap/gmadl"
    )
    assert paths["checkpoint_dir"] == Path(
        "/tmp/FYP/outputs/phase1_5_robustness/light/checkpoints/seed42_nocap/gmadl"
    )


def test_build_command_forwards_seed_weight_and_paths(tmp_path: Path) -> None:
    run = phase15.Phase15Run(loss_name="imadl", seed=52, max_weight=0.05)
    command = phase15.build_command_for_run(
        run=run,
        data_dir=Path("/data"),
        pattern="*.csv",
        lookback_months=12,
        best_config_path=Path("/best.txt"),
        train_start="1990-01",
        train_end="1994-12",
        test_start="1995-01",
        test_months=24,
        max_epochs=20,
        batch_size=1024,
        resume_mode="auto",
        paths=phase15.derive_run_paths(tmp_path, "light", run),
    )

    assert command[0] == phase15.sys.executable
    assert Path(command[1]).name == "run_sanity_check_imadl.py"
    assert "--seed" in command
    assert "52" in command
    assert "--max-weight" in command
    assert "0.05" in command
    assert "--output-dir" in command
    assert str(tmp_path / "outputs" / "phase1_5_robustness" / "light" / "runs" / "seed52_cap005" / "imadl") in command
    assert "--checkpoint-dir" in command
    assert str(tmp_path / "outputs" / "phase1_5_robustness" / "light" / "checkpoints" / "seed52_cap005" / "imadl") in command


def test_run_phase15_robustness_skips_complete_runs(tmp_path: Path, monkeypatch) -> None:
    drive_root = tmp_path / "drive"
    run = phase15.Phase15Run(loss_name="mse", seed=42, max_weight=0.05)
    paths = phase15.derive_run_paths(drive_root, "light", run)
    paths["output_dir"].mkdir(parents=True)
    (paths["output_dir"] / "sanity_metrics_mse.csv").write_text("month,long_short_return\n1995-01,0.1\n")
    (paths["output_dir"] / "sanity_summary_mse.json").write_text(
        json.dumps(
            {
                "loss": "mse",
                "avg_mse": 1.0,
                "avg_medse": 0.5,
                "avg_r2": 0.2,
                "avg_directional_accuracy": 0.6,
                "avg_sign_mismatch_large_y": 0.3,
                "avg_long_short": 0.1,
                "long_short_cumulative_return": 0.1,
                "long_short_std": 0.1,
                "long_short_sharpe": 1.0,
            }
        )
    )
    (paths["output_dir"] / "mse_loss_curve.png").write_bytes(b"png")
    (paths["output_dir"] / "mse_returns_curve.png").write_bytes(b"png")

    calls = []

    def fake_run(cmd, check, timeout=None):
        calls.append(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sanity_metrics_mse.csv").write_text(
            "month,long_short_return\n1995-01,0.1\n"
        )
        (output_dir / "sanity_summary_mse.json").write_text(
            json.dumps(
                {
                    "loss": "mse",
                    "avg_mse": 1.0,
                    "avg_medse": 0.5,
                    "avg_r2": 0.2,
                    "avg_directional_accuracy": 0.6,
                    "avg_sign_mismatch_large_y": 0.3,
                    "avg_long_short": 0.1,
                    "long_short_cumulative_return": 0.1,
                    "long_short_std": 0.1,
                    "long_short_sharpe": 1.0,
                }
            )
        )
        (output_dir / "mse_loss_curve.png").write_bytes(b"png")
        (output_dir / "mse_returns_curve.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = phase15.run_phase15_robustness(
        losses=["mse"],
        seeds=[42, 52, 62],
        matrix_mode="light",
        drive_root=drive_root,
        data_dir=Path("/data"),
        pattern="*.csv",
        lookback_months=12,
        best_config_path=Path("/best.txt"),
        train_start="1990-01",
        train_end="1994-12",
        test_start="1995-01",
        test_months=24,
        max_epochs=20,
        batch_size=1024,
        resume_mode="auto",
        nocap_seed=42,
        skip_existing=True,
        stop_on_error=False,
    )

    assert len(calls) == 3
    assert results["completed"] == 4
    assert results["failed"] == 0


def test_run_phase15_robustness_passes_timeout_to_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    drive_root = tmp_path / "drive"
    captured = {}

    def fake_run(cmd, check, timeout):
        captured["timeout"] = timeout
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        loss_name = Path(cmd[1]).stem.replace("run_sanity_check_", "")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"sanity_metrics_{loss_name}.csv").write_text(
            "month,long_short_return\n1995-01,0.1\n"
        )
        (output_dir / f"sanity_summary_{loss_name}.json").write_text(
            json.dumps(
                {
                    "loss": loss_name,
                    "avg_mse": 1.0,
                    "avg_medse": 0.5,
                    "avg_r2": 0.2,
                    "avg_directional_accuracy": 0.6,
                    "avg_sign_mismatch_large_y": 0.3,
                    "avg_long_short": 0.1,
                    "long_short_cumulative_return": 0.1,
                    "long_short_std": 0.1,
                    "long_short_sharpe": 1.0,
                }
            )
        )
        (output_dir / f"{loss_name}_loss_curve.png").write_bytes(b"png")
        (output_dir / f"{loss_name}_returns_curve.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = phase15.run_phase15_robustness(
        losses=["mse"],
        seeds=[42],
        matrix_mode="light",
        drive_root=drive_root,
        data_dir=Path("/data"),
        pattern="*.csv",
        lookback_months=12,
        best_config_path=Path("/best.txt"),
        train_start="1990-01",
        train_end="1994-12",
        test_start="1995-01",
        test_months=24,
        max_epochs=20,
        batch_size=1024,
        resume_mode="auto",
        nocap_seed=42,
        skip_existing=False,
        stop_on_error=False,
    )

    assert results["failed"] == 0
    assert captured["timeout"] == phase15.RUN_TIMEOUT_SECONDS


def test_collect_run_rows_reads_seed_and_weight_from_run_spec(tmp_path: Path) -> None:
    _write_run_artifacts(
        tmp_path,
        matrix_mode="light",
        loss_name="mse",
        seed=42,
        max_weight=0.05,
        sharpe=1.2,
        cumulative_return=0.3,
        avg_long_short=0.05,
    )
    _write_run_artifacts(
        tmp_path,
        matrix_mode="light",
        loss_name="mse",
        seed=52,
        max_weight=0.05,
        sharpe=0.8,
        cumulative_return=0.1,
        avg_long_short=0.02,
    )
    _write_run_artifacts(
        tmp_path,
        matrix_mode="light",
        loss_name="mse",
        seed=42,
        max_weight=None,
        sharpe=1.0,
        cumulative_return=0.2,
        avg_long_short=0.03,
    )

    runs_root = tmp_path / "outputs" / "phase1_5_robustness" / "light"
    raw_df = aggregate_phase15.collect_run_rows(runs_root)

    assert list(raw_df["seed"]) == [42, 42, 52]
    assert list(raw_df["max_weight_label"]) == ["0.05", "None", "0.05"]


def test_build_grouped_summary_computes_mean_and_std() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "loss": "mse",
                "seed": 42,
                "max_weight": 0.05,
                "max_weight_label": "0.05",
                "avg_long_short": 0.02,
                "long_short_cumulative_return": 0.1,
                "long_short_sharpe": 1.0,
            },
            {
                "loss": "mse",
                "seed": 52,
                "max_weight": 0.05,
                "max_weight_label": "0.05",
                "avg_long_short": 0.04,
                "long_short_cumulative_return": 0.3,
                "long_short_sharpe": 2.0,
            },
        ]
    )

    summary_df = aggregate_phase15.build_grouped_summary(raw_df)

    row = summary_df.iloc[0]
    assert row["runs"] == 2
    assert row["sharpe_mean"] == pytest.approx(1.5)
    assert row["cumret_mean"] == pytest.approx(0.2)
    assert row["avg_ls_mean"] == pytest.approx(0.03)


def test_collect_run_rows_skips_missing_run_spec(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "phase1_5_robustness" / "light" / "runs" / "seed42_cap005" / "mse"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sanity_summary_mse.json").write_text(json.dumps({"loss": "mse"}))

    raw_df = aggregate_phase15.collect_run_rows(
        tmp_path / "outputs" / "phase1_5_robustness" / "light"
    )

    assert raw_df.empty
