import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Model_Train.losses import EXPERIMENT_LOSS_NAMES, get_experiment_loss_fn


@pytest.mark.parametrize("loss_name", EXPERIMENT_LOSS_NAMES)
def test_experiment_loss_dispatch_returns_finite_scalar(loss_name: str) -> None:
    y_true = torch.tensor([0.02, -0.03, 0.01, -0.04], dtype=torch.float32)
    y_pred = torch.tensor(
        [0.01, -0.02, -0.01, 0.03], dtype=torch.float32, requires_grad=True
    )

    loss = get_experiment_loss_fn(loss_name)(y_true, y_pred)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_phase15_lambda_sweep_names_are_explicit() -> None:
    expected = {
        "hybrid_add_a1",
        "hybrid_add_a2",
        "hybrid_add_a3",
        "hybrid_add_a4",
        "hybrid_add_a5",
        "hybrid_mul_m1",
        "hybrid_mul_m2",
        "hybrid_mul_m3",
        "hybrid_mul_m4",
    }

    assert expected.issubset(set(EXPERIMENT_LOSS_NAMES))


def test_existing_mse_and_medse_dispatch_values_are_preserved() -> None:
    y_true = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32)
    y_pred = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)

    mse = get_experiment_loss_fn("mse")(y_true, y_pred)
    medse = get_experiment_loss_fn("medse")(y_true, y_pred)

    assert mse.item() == pytest.approx(((1.0 + 0.0 + 16.0) / 3.0))
    assert medse.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "module_name",
    [
        "run_sanity_check_madl",
        "run_sanity_check_gmadl",
        "run_sanity_check_imadl",
        "run_sanity_check_hybrid_add_a1",
        "run_sanity_check_hybrid_add_a2",
        "run_sanity_check_hybrid_add_a3",
        "run_sanity_check_hybrid_add_a4",
        "run_sanity_check_hybrid_add_a5",
        "run_sanity_check_hybrid_mul_m1",
        "run_sanity_check_hybrid_mul_m2",
        "run_sanity_check_hybrid_mul_m3",
        "run_sanity_check_hybrid_mul_m4",
    ],
)
def test_supported_loss_runner_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert callable(module.main)
