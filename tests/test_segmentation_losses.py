import pytest
import torch
import torch.nn.functional as F

from skp.losses.segmentation import PositiveDiceNegativeFocalLoss


def _loss(**params):
    defaults = {
        "activation_fn": "sigmoid",
        "compute_method": "per_sample",
        "negative_focal_weight": 0.05,
        "gamma": 0.0,
        "alpha": None,
    }
    defaults.update(params)
    return PositiveDiceNegativeFocalLoss(defaults)


def test_positive_dice_negative_focal_ignores_empty_samples_for_dice():
    target = torch.zeros(2, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0
    logits_a = torch.zeros_like(target)
    logits_b = logits_a.clone()
    logits_b[1] = 10.0

    loss_fn = _loss(focal_weight=0.0)

    dice_a = loss_fn({"logits": logits_a}, {"y": target})["dice_loss"]
    dice_b = loss_fn({"logits": logits_b}, {"y": target})["dice_loss"]

    assert dice_a == pytest.approx(dice_b)


def test_positive_dice_negative_focal_normalizes_focal_by_effective_weight():
    target = torch.zeros(3, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0
    logits = torch.zeros_like(target)
    loss_fn = _loss(
        dice_weight=0.0,
        focal_weight=1.0,
        positive_focal_weight=1.0,
        negative_focal_weight=0.1,
    )

    out = loss_fn({"logits": logits}, {"y": target})
    bce_per_sample = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        .flatten(1)
        .mean(dim=1)
    )
    weights = torch.tensor([1.0, 0.1, 0.1])
    expected = (bce_per_sample * weights).sum() / weights.sum()

    assert out["focal_loss"] == pytest.approx(expected)
    assert out["loss"] == pytest.approx(expected)
    assert out["positive_fraction"] == pytest.approx(torch.tensor(1 / 3))


def test_positive_dice_negative_focal_handles_all_empty_batch():
    target = torch.zeros(2, 1, 2, 2)
    logits = torch.zeros_like(target)
    loss_fn = _loss()

    out = loss_fn({"logits": logits}, {"y": target})

    assert torch.isfinite(out["loss"])
    assert out["dice_loss"] == pytest.approx(torch.tensor(0.0))
    assert out["positive_fraction"] == pytest.approx(torch.tensor(0.0))
