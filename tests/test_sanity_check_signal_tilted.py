from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import sanity_check_signal_tilted as scst
from Model_Train.models import MLPConfig


def _make_args(
    tmp_path: Path,
    *,
    output_name: str,
    checkpoint_name: str,
    **overrides,
) -> SimpleNamespace:
    args = {
        "data_dir": ".",
        "pattern": "*.csv",
        "lookback_months": 12,
        "train_start": "1990-01",
        "train_end": "1994-12",
        "test_start": "1995-01",
        "test_months": 3,
        "best_config_path": "best_hyperparameters.txt",
        "batch_size": 8,
        "max_epochs": 3,
        "output_dir": str(tmp_path / output_name),
        "checkpoint_dir": str(tmp_path / checkpoint_name),
        "archive_root": None,
        "resume_mode": "resume",
        "seed": 123,
        "max_weight": None,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _make_synthetic_arrays(
    *,
    rows_per_month: int = 6,
    missing_test_months: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    all_months = pd.period_range("1990-01", "1995-03", freq="M")
    skipped = {pd.Period(month, freq="M") for month in missing_test_months}
    months = [period for period in all_months if period not in skipped]

    rng = np.random.default_rng(7)
    total_rows = len(months) * rows_per_month
    x_all = rng.normal(size=(total_rows, 3)).astype(np.float32)
    coeffs = np.array([0.35, -0.2, 0.15], dtype=np.float32)
    y_all = (x_all @ coeffs + 0.03 * rng.normal(size=total_rows)).astype(np.float32)
    dates = pd.Series(
        [period.to_timestamp() for period in months for _ in range(rows_per_month)]
    )
    return x_all, y_all, dates


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    dates: pd.Series,
) -> None:
    ids = pd.Series(np.arange(len(dates)), dtype=int)

    monkeypatch.setattr(
        scst,
        "prepare_panel_data",
        lambda **_kwargs: pd.DataFrame({"stub": [1]}),
    )
    monkeypatch.setattr(scst, "build_feature_set_x1", lambda panel, feature_cfg: panel)
    monkeypatch.setattr(
        scst,
        "assemble_feature_matrix",
        lambda _df_x1: (x_all, y_all, ids, dates),
    )
    monkeypatch.setattr(
        scst,
        "read_best_config",
        lambda path, input_dim: MLPConfig(
            input_dim=input_dim,
            hidden_dims=[6],
            activation="relu",
            dropout=0.0,
        ),
    )
    monkeypatch.setattr(scst, "plot_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(scst, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(scst, "configure_matplotlib", lambda *_args, **_kwargs: None)


def test_compute_directional_metrics_reports_expected_fields() -> None:
    y_true = np.array([0.02, -0.03, 0.01, -0.04], dtype=float)
    y_pred = np.array([0.01, 0.02, -0.02, -0.03], dtype=float)

    metrics = scst.compute_directional_metrics(y_true, y_pred)

    assert metrics["directional_accuracy"] == 0.5
    assert metrics["sign_mismatch_large_y"] == 0.0


def test_normalize_resume_mode_supports_plan_modes_and_legacy_aliases() -> None:
    assert scst.normalize_resume_mode("auto") == "auto"
    assert scst.normalize_resume_mode("never") == "never"
    assert scst.normalize_resume_mode("require") == "require"
    assert scst.normalize_resume_mode("resume") == "auto"
    assert scst.normalize_resume_mode("off") == "never"


def test_summarize_results_uses_unified_output_schema() -> None:
    df_result = pd.DataFrame(
        [
            {
                "month": "1995-01",
                "sample_size": 10,
                "mse": 1.0,
                "medse": 0.8,
                "r2": 0.1,
                "directional_accuracy": 0.6,
                "sign_mismatch_large_y": 0.4,
                "long_return": 0.02,
                "short_return": -0.01,
                "long_short_return": 0.03,
            },
            {
                "month": "1995-02",
                "sample_size": 12,
                "mse": 1.2,
                "medse": 0.9,
                "r2": 0.2,
                "directional_accuracy": 0.5,
                "sign_mismatch_large_y": 0.3,
                "long_return": 0.01,
                "short_return": -0.02,
                "long_short_return": 0.03,
            },
        ]
    )

    summary = scst.summarize_results("hybrid_add", df_result)

    assert summary["loss"] == "hybrid_add"
    assert summary["avg_mse"] == pytest.approx(1.1)
    assert summary["avg_medse"] == pytest.approx(0.85)
    assert summary["avg_r2"] == pytest.approx(0.15)
    assert summary["avg_directional_accuracy"] == pytest.approx(0.55)
    assert summary["avg_sign_mismatch_large_y"] == pytest.approx(0.35)
    assert summary["avg_long_short"] == pytest.approx(0.03)
    assert summary["long_short_cumulative_return"] == pytest.approx(0.0609)
    assert summary["long_short_std"] == pytest.approx(0.0)
    assert np.isnan(summary["long_short_sharpe"])


@pytest.mark.parametrize(
    "loss_name",
    ["mse", "medse", "gmadl", "imadl", "dirhuber", "hybrid_add", "hybrid_mul"],
)
def test_run_sanity_check_writes_unified_summary_schema(
    loss_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "sanity_outputs"
    months = pd.period_range("1990-01", "1995-02", freq="M")
    rows_per_month = 4
    dates = pd.Index(
        [period.to_timestamp() for period in months for _ in range(rows_per_month)]
    )
    ids = pd.Series(np.arange(len(dates)), dtype=int)
    x_all = np.arange(len(dates) * 2, dtype=np.float32).reshape(len(dates), 2)
    y_all = np.linspace(-0.04, 0.05, len(dates), dtype=np.float32)

    def fake_prepare_panel_data(**_kwargs):
        return pd.DataFrame({"stub": [1]})

    def fake_build_feature_set_x1(panel, feature_cfg):
        assert panel.shape == (1, 1)
        assert feature_cfg.lookback_months == 12
        return panel

    def fake_assemble_feature_matrix(_df_x1):
        return x_all, y_all, ids, pd.Series(dates)

    def fake_read_best_config(path, input_dim):
        assert path == Path("best_hyperparameters.txt")
        return MLPConfig(input_dim=input_dim, hidden_dims=[4, 2], activation="relu")

    def fake_train_model(*_args, **_kwargs):
        return object()

    def fake_predict(_model, x, device=None):
        return np.linspace(0.01, 0.02, x.shape[0], dtype=np.float32)

    monkeypatch.setattr(scst, "prepare_panel_data", fake_prepare_panel_data)
    monkeypatch.setattr(scst, "build_feature_set_x1", fake_build_feature_set_x1)
    monkeypatch.setattr(scst, "assemble_feature_matrix", fake_assemble_feature_matrix)
    monkeypatch.setattr(scst, "read_best_config", fake_read_best_config)
    monkeypatch.setattr(scst, "train_model", fake_train_model)
    monkeypatch.setattr(scst, "predict", fake_predict)
    monkeypatch.setattr(scst, "plot_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(scst, "detect_device", lambda: "cpu")
    monkeypatch.setattr(scst, "configure_matplotlib", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scst, "set_seed", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        data_dir=".",
        pattern="*.csv",
        lookback_months=12,
        train_start="1990-01",
        train_end="1994-12",
        test_start="1995-01",
        test_months=2,
        best_config_path="best_hyperparameters.txt",
        batch_size=8,
        max_epochs=1,
        output_dir=str(output_dir),
        seed=42,
        max_weight=None,
    )

    scst.run_sanity_check(loss_name, args)

    metrics_path = output_dir / f"sanity_metrics_{loss_name}.csv"
    summary_path = output_dir / f"sanity_summary_{loss_name}.json"
    df_metrics = pd.read_csv(metrics_path)
    summary = pd.Series(json.loads(summary_path.read_text()))

    assert list(df_metrics.columns) == [
        "month",
        "sample_size",
        "mse",
        "medse",
        "r2",
        "directional_accuracy",
        "sign_mismatch_large_y",
        "long_return",
        "short_return",
        "long_short_return",
        "cumulative_long_short_return",
    ]
    assert set(summary.index) == {
        "loss",
        "avg_mse",
        "avg_medse",
        "avg_r2",
        "avg_directional_accuracy",
        "avg_sign_mismatch_large_y",
        "avg_long_short",
        "long_short_cumulative_return",
        "long_short_std",
        "long_short_sharpe",
    }
    assert summary["loss"] == loss_name
    assert df_metrics["sample_size"].tolist() == [4, 4]
    assert np.isfinite(summary["avg_mse"])
    assert np.isfinite(summary["avg_long_short"])


def test_run_sanity_check_resumes_training_after_completed_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays()

    baseline_args = _make_args(
        tmp_path,
        output_name="baseline_outputs",
        checkpoint_name="baseline_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)
    scst.run_sanity_check("mse", baseline_args)
    baseline_metrics = pd.read_csv(
        Path(baseline_args.output_dir) / "sanity_metrics_mse.csv"
    )

    interrupted_args = _make_args(
        tmp_path,
        output_name="resume_outputs",
        checkpoint_name="resume_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    original_save_train_state = scst.save_train_state
    raised = {"value": False}

    def interrupting_save_train_state(path: Path, state: dict) -> None:
        original_save_train_state(path, state)
        if state["completed_epochs"] == 1 and not raised["value"]:
            raised["value"] = True
            raise RuntimeError("simulated training interruption")

    monkeypatch.setattr(scst, "save_train_state", interrupting_save_train_state)

    with pytest.raises(RuntimeError, match="simulated training interruption"):
        scst.run_sanity_check("mse", interrupted_args)

    train_state_path = Path(interrupted_args.checkpoint_dir) / "mse" / "train_state.json"
    train_state = json.loads(train_state_path.read_text())
    assert train_state["completed_epochs"] == 1

    monkeypatch.setattr(scst, "save_train_state", original_save_train_state)
    scst.run_sanity_check("mse", interrupted_args)

    resumed_metrics = pd.read_csv(
        Path(interrupted_args.output_dir) / "sanity_metrics_mse.csv"
    )
    pd.testing.assert_frame_equal(resumed_metrics, baseline_metrics)


def test_run_sanity_check_resumes_evaluation_from_metrics_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays(missing_test_months=("1995-02",))

    baseline_args = _make_args(
        tmp_path,
        output_name="baseline_eval_outputs",
        checkpoint_name="baseline_eval_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)
    scst.run_sanity_check("mse", baseline_args)
    baseline_metrics = pd.read_csv(
        Path(baseline_args.output_dir) / "sanity_metrics_mse.csv"
    )

    interrupted_args = _make_args(
        tmp_path,
        output_name="resume_eval_outputs",
        checkpoint_name="resume_eval_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    original_save_progress = scst.save_progress
    raised = {"value": False}

    def interrupting_save_progress(path: Path, state: dict) -> None:
        original_save_progress(path, state)
        if (
            state["stage"] == "evaluating"
            and state["completed_months"] == ["1995-01"]
            and not raised["value"]
        ):
            raised["value"] = True
            raise RuntimeError("simulated evaluation interruption")

    monkeypatch.setattr(scst, "save_progress", interrupting_save_progress)

    with pytest.raises(RuntimeError, match="simulated evaluation interruption"):
        scst.run_sanity_check("mse", interrupted_args)

    metrics_path = Path(interrupted_args.output_dir) / "sanity_metrics_mse.csv"
    partial_metrics = pd.read_csv(metrics_path)
    assert partial_metrics["month"].tolist() == ["1995-01"]

    monkeypatch.setattr(scst, "save_progress", original_save_progress)
    scst.run_sanity_check("mse", interrupted_args)

    resumed_metrics = pd.read_csv(metrics_path)
    pd.testing.assert_frame_equal(resumed_metrics, baseline_metrics)

    progress_path = Path(interrupted_args.checkpoint_dir) / "mse" / "progress.json"
    progress = json.loads(progress_path.read_text())
    assert progress["completed_months"] == ["1995-01", "1995-02", "1995-03"]


def test_run_sanity_check_resume_does_not_duplicate_completed_months(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays(missing_test_months=("1995-02",))
    args = _make_args(
        tmp_path,
        output_name="dedupe_outputs",
        checkpoint_name="dedupe_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    scst.run_sanity_check("mse", args)
    metrics_path = Path(args.output_dir) / "sanity_metrics_mse.csv"
    first_metrics = pd.read_csv(metrics_path)

    predict_calls = {"count": 0}
    original_predict = scst.predict

    def counting_predict(*args, **kwargs):
        predict_calls["count"] += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(scst, "predict", counting_predict)
    scst.run_sanity_check("mse", args)

    second_metrics = pd.read_csv(metrics_path)
    pd.testing.assert_frame_equal(second_metrics, first_metrics)
    assert predict_calls["count"] == 0

    progress_path = Path(args.checkpoint_dir) / "mse" / "progress.json"
    progress = json.loads(progress_path.read_text())
    assert progress["completed_months"] == ["1995-01", "1995-02", "1995-03"]


def test_run_sanity_check_rejects_mismatched_run_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays()
    original_args = _make_args(
        tmp_path,
        output_name="spec_outputs",
        checkpoint_name="spec_checkpoints",
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)
    scst.run_sanity_check("mse", original_args)

    mismatched_args = _make_args(
        tmp_path,
        output_name="spec_outputs",
        checkpoint_name="spec_checkpoints",
        test_months=4,
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    with pytest.raises(ValueError, match="Run spec mismatch"):
        scst.run_sanity_check("mse", mismatched_args)


def test_run_sanity_check_persists_loss_kwargs_in_run_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays()
    args = _make_args(
        tmp_path,
        output_name="loss_kwargs_outputs",
        checkpoint_name="loss_kwargs_checkpoints",
        loss_kwargs='{"lambda_dir": 5.0, "lambda_hub": 0.1}',
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    scst.run_sanity_check("hybrid_add", args)

    run_spec_path = Path(args.checkpoint_dir) / "hybrid_add" / "run_spec.json"
    run_spec = json.loads(run_spec_path.read_text())
    assert run_spec["loss_kwargs"] == {"lambda_dir": 5.0, "lambda_hub": 0.1}


def test_run_sanity_check_rejects_mismatched_loss_kwargs_in_run_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays()
    original_args = _make_args(
        tmp_path,
        output_name="spec_kwargs_outputs",
        checkpoint_name="spec_kwargs_checkpoints",
        loss_kwargs='{"lambda_dir": 5.0, "lambda_hub": 0.1}',
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)
    scst.run_sanity_check("hybrid_add", original_args)

    mismatched_args = _make_args(
        tmp_path,
        output_name="spec_kwargs_outputs",
        checkpoint_name="spec_kwargs_checkpoints",
        loss_kwargs='{"lambda_dir": 10.0, "lambda_hub": 0.1}',
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    with pytest.raises(ValueError, match="Run spec mismatch"):
        scst.run_sanity_check("hybrid_add", mismatched_args)


def test_run_sanity_check_archives_single_loss_outputs_after_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_all, y_all, dates = _make_synthetic_arrays()
    archive_root = tmp_path / "drive" / "FYP"
    args = _make_args(
        tmp_path,
        output_name="archive_outputs",
        checkpoint_name="archive_checkpoints",
        archive_root=str(archive_root),
    )
    _patch_pipeline(monkeypatch, x_all=x_all, y_all=y_all, dates=dates)

    def fake_plot_curves(_df, loss_name, output_dir):
        (Path(output_dir) / f"{loss_name}_loss_curve.png").write_bytes(b"loss")
        (Path(output_dir) / f"{loss_name}_returns_curve.png").write_bytes(b"returns")

    monkeypatch.setattr(scst, "plot_curves", fake_plot_curves)
    scst.run_sanity_check("mse", args)

    archived_dir = archive_root / "mse"
    assert archived_dir.exists()
    assert (archived_dir / "sanity_metrics_mse.csv").exists()
    assert (archived_dir / "sanity_summary_mse.json").exists()
    assert (archived_dir / "mse_loss_curve.png").exists()
    assert (archived_dir / "mse_returns_curve.png").exists()

    archived_summary = json.loads(
        (archived_dir / "sanity_summary_mse.json").read_text()
    )
    assert archived_summary["loss"] == "mse"
