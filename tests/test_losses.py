import pytest
import torch

from Model_Train.losses import (
    EXPERIMENT_LOSS_NAMES,
    directional_huber_loss,
    gmadl_loss,
    get_experiment_loss_fn,
    hybrid_dir_huber_add_loss,
    hybrid_dir_huber_mul_loss,
    imadl_rebalanced_loss,
    madl_loss,
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
        (
            "hybrid_add_a4",
            lambda y_true, y_pred: hybrid_dir_huber_add_loss(
                y_true, y_pred, lambda_dir=5.0, lambda_hub=0.1
            ),
        ),
        (
            "hybrid_mul_m1",
            lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
                y_true, y_pred, lambda_dir=2.0
            ),
        ),
        (
            "hybrid_mul_m2",
            lambda y_true, y_pred: hybrid_dir_huber_mul_loss(
                y_true, y_pred, lambda_dir=5.0
            ),
        ),
    ],
)
def test_experiment_loss_catalog_supports_registered_losses(
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
