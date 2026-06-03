import pytest
import torch

from skp.configs import Config
from skp.optim import get_scheduler
from skp.optim.linear_warmup_cosine_annealing import LinearWarmupCosineAnnealingLR


def test_linear_warmup_cosine_scheduler_reaches_expected_bounds():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        total_steps=10,
        max_lr=0.1,
        init_lr=0.0,
        final_lr=0.01,
        warmup_steps=2,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    optimizer._opt_called = True
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    for _ in range(20):
        optimizer._opt_called = True
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_get_scheduler_enforces_max_lr_matches_optimizer_lr():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    cfg = Config(
        scheduler="LinearWarmupCosineAnnealingLR",
        scheduler_params={"max_lr": 0.2, "pct_start": 0.1},
        num_iterations_per_epoch=None,
        batch_size=2,
        world_size=1,
        n_train=10,
        accumulate_grad_batches=1,
        num_epochs=1,
    )

    with pytest.raises(ValueError, match="max_lr"):
        get_scheduler(cfg, optimizer)


def test_get_scheduler_returns_none_when_disabled():
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=0.1)
    assert get_scheduler(Config(scheduler=None), optimizer) is None
