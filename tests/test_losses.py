import pytest
import torch

from Model_Train.losses import (
    EXPERIMENT_LOSS_NAMES,
    adaptive_lambda100_loss,
    adaptive_lambda10_loss,
    adaptive_lambda50_loss,
    directional_huber_loss,
    gmadl_loss,
    get_experiment_loss_fn,
    hybrid_dir_huber_add_loss,
    hybrid_dir_huber_mul_loss,
    imadl_rebalanced_loss,
    madl_loss,
    m2_loss,
    m2_robust_gamma01_loss,
    medse_loss,
    mse_loss,
)


@pytest.mark.parametrize(
    ("loss_name", "loss_fn"),
    [
        ("mse", mse_loss),
        ("medse", medse_loss),
        ("gmadl", gmadl_loss),
        ("imadl", imadl_rebalanced_loss),
        ("dirhuber", directional_huber_loss),
        ("hybrid_add", hybrid_dir_huber_add_loss),
        ("hybrid_mul", hybrid_dir_huber_mul_loss),
    ],
)
def test_experiment_loss_catalog_supports_all_seven_losses(
    loss_name: str, loss_fn
) -> None:
    resolved = get_experiment_loss_fn(loss_name)
    y_true = torch.tensor([0.02, -0.03, 0.01], dtype=torch.float32)
    y_pred = torch.tensor([0.01, -0.01, 0.02], dtype=torch.float32, requires_grad=True)

    direct = loss_fn(y_true, y_pred)
    via_resolver = resolved(y_true, y_pred)
    via_resolver.backward()

    assert loss_name in EXPERIMENT_LOSS_NAMES
    assert callable(resolved)
    assert torch.isfinite(direct)
    assert torch.isfinite(via_resolver)
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_directional_losses_reward_correct_sign_alignment() -> None:
    y_true = torch.tensor([0.02, -0.03], dtype=torch.float32)
    aligned_pred = torch.tensor([0.015, -0.02], dtype=torch.float32)
    flipped_pred = torch.tensor([-0.015, 0.02], dtype=torch.float32)

    madl_aligned = madl_loss(y_true, aligned_pred)
    madl_flipped = madl_loss(y_true, flipped_pred)
    gmadl_aligned = gmadl_loss(y_true, aligned_pred)
    gmadl_flipped = gmadl_loss(y_true, flipped_pred)

    assert madl_aligned.item() < madl_flipped.item()
    assert gmadl_aligned.item() < gmadl_flipped.item()


def test_mse_and_medse_reduce_consistently_for_small_tensors() -> None:
    y_true = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)
    y_pred = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)

    squared_errors = (y_true - y_pred) ** 2

    assert torch.isclose(
        mse_loss(y_true, y_pred, reduction="sum"), squared_errors.sum()
    )
    assert torch.isclose(
        mse_loss(y_true, y_pred, reduction="mean"), squared_errors.mean()
    )
    assert torch.isclose(
        medse_loss(y_true, y_pred, reduction="median"), squared_errors.median()
    )


def test_experiment_loss_resolver_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment loss"):
        get_experiment_loss_fn("hybrid")


@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "median"])
def test_m2_loss_matches_phase15_hybrid_mul_lambda_dir_2(reduction: str) -> None:
    y_true = torch.tensor([0.03, -0.02, 0.0, 0.015], dtype=torch.float32)
    y_pred = torch.tensor([0.01, -0.03, 0.02, -0.005], dtype=torch.float32)

    actual = m2_loss(y_true, y_pred, reduction=reduction)
    expected = hybrid_dir_huber_mul_loss(
        y_true, y_pred, lambda_dir=2.0, reduction=reduction
    )

    assert torch.allclose(actual, expected)


def test_robustness_penalty_increases_loss_for_higher_prediction_variance() -> None:
    low_var_pred = torch.full((4,), 0.01, dtype=torch.float32)
    high_var_pred = torch.tensor([-0.03, 0.05, -0.03, 0.05], dtype=torch.float32)

    low_loss = m2_robust_gamma01_loss(low_var_pred, low_var_pred)
    high_loss = m2_robust_gamma01_loss(high_var_pred, high_var_pred)

    assert torch.isclose(m2_loss(low_var_pred, low_var_pred), torch.tensor(0.0))
    assert torch.isclose(m2_loss(high_var_pred, high_var_pred), torch.tensor(0.0))
    assert high_var_pred.var() > low_var_pred.var()
    assert high_loss > low_loss


@pytest.mark.parametrize(
    ("loss_name", "loss_fn"),
    [
        ("m2", m2_loss),
        ("imadl_m2_alpha02", None),
        ("imadl_m2_alpha03", None),
        ("imadl_m2_alpha04", None),
        ("imadl_m2_alpha05", None),
        ("imadl_m2_alpha06", None),
        ("imadl_m2_alpha07", None),
        ("imadl_m2_alpha08", None),
        ("imadl_gmadl_beta03", None),
        ("imadl_gmadl_beta05", None),
        ("imadl_gmadl_beta07", None),
        ("m2_robust_gamma001", None),
        ("m2_robust_gamma01", m2_robust_gamma01_loss),
        ("m2_robust_gamma10", None),
        ("adaptive_lambda10", adaptive_lambda10_loss),
        ("adaptive_lambda50", adaptive_lambda50_loss),
        ("adaptive_lambda100", adaptive_lambda100_loss),
    ],
)
def test_experiment_loss_catalog_includes_phase2_losses(loss_name: str, loss_fn) -> None:
    resolved = get_experiment_loss_fn(loss_name)
    y_true = torch.tensor([0.02, -0.03, 0.01], dtype=torch.float32)
    y_pred = torch.tensor([0.01, -0.01, 0.02], dtype=torch.float32, requires_grad=True)

    via_resolver = resolved(y_true, y_pred)
    via_resolver.backward()

    assert loss_name in EXPERIMENT_LOSS_NAMES
    assert torch.isfinite(via_resolver)
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    if loss_fn is not None:
        assert torch.allclose(via_resolver.detach(), loss_fn(y_true, y_pred.detach()))
