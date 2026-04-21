from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sanity_check_signal_tilted as scst
from Model_Train.models import MLPConfig


def test_compute_directional_metrics_reports_expected_fields() -> None:
    y_true = np.array([0.02, -0.03, 0.01, -0.04], dtype=float)
    y_pred = np.array([0.01, 0.02, -0.02, -0.03], dtype=float)

    metrics = scst.compute_directional_metrics(y_true, y_pred)

    assert metrics["directional_accuracy"] == 0.5
    assert metrics["sign_mismatch_large_y"] == 0.0


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
