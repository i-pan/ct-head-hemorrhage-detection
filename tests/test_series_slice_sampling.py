import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from skp.configs import Config
from skp.tasks.samplers import (
    SeriesSliceSampler,
    DistributedSamplerWrapper,
    configure_series_sampling_schedule,
)
from skp.tasks.utils import build_dataloader
from skp.optim import get_scheduler


class SeriesDataset(Dataset):
    def __init__(self, count=10):
        self.df = pd.DataFrame(
            [
                {"series_uid": str(s), "any": label}
                for s in range(count)
                for label in (
                    [0] * 5
                    if s % 3 == 0
                    else [1] * 3
                    if s % 3 == 1
                    else [0] * 7 + [1] * 2
                )
            ]
        )
        self.collate_fn = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {"index": index, "y": int(self.df.iloc[index]["any"])}


def config(**overrides):
    c = Config(
        seed=88,
        series_positive_slice_probability=0.5,
        num_iterations_per_epoch=None,
        sampler="SeriesSliceSampler",
        train_drop_last=False,
        world_size=1,
        batch_size=4,
        num_workers=0,
        args={"strategy": "auto"},
        num_epochs=50,
        accumulate_grad_batches=1,
        scheduler="LinearWarmupCosineAnnealingLR",
        scheduler_interval="step",
        scheduler_params={"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6},
        n_train=1000000,
    )
    c.__dict__.update(overrides)
    return c


def test_one_per_series_and_independent_bins_without_replacement():
    ds = SeriesDataset()
    sampler = SeriesSliceSampler(ds, config())
    history = {}
    positive_draws = []
    for epoch in range(300):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert indices == list(sampler)  # loader reconstruction cannot advance bins
        rows = ds.df.iloc[indices]
        assert rows.series_uid.nunique() == len(sampler) == len(indices)
        for index, row in zip(indices, rows.itertuples()):
            key = (row.series_uid, row.any)
            history.setdefault(key, []).append(index)
            if row.series_uid == "2":
                positive_draws.append(row.any)
    for (series, label), draws in history.items():
        members = set(
            np.flatnonzero((ds.df.series_uid == series) & (ds.df["any"] == label))
        )
        for start in range(0, len(draws), len(members)):
            chunk = draws[start : start + len(members)]
            assert len(chunk) == len(set(chunk))
            if len(chunk) == len(members):
                assert set(chunk) == members
    assert (
        0.4 < np.mean(positive_draws) < 0.6
    )  # distinct bin sizes must not dictate 2/9 frequency


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_extremes_and_single_category_fallback(probability):
    ds = SeriesDataset()
    sampler = SeriesSliceSampler(
        ds, config(series_positive_slice_probability=probability)
    )
    for epoch in range(15):
        sampler.set_epoch(epoch)
        rows = ds.df.iloc[list(sampler)].set_index("series_uid")
        assert rows.loc["0", "any"] == 0
        assert rows.loc["1", "any"] == 1
        assert rows.loc["2", "any"] == probability


def test_resume_replays_bin_positions_and_ddp_ranks_partition_same_plan():
    ds = SeriesDataset()
    cfg = config()
    original = SeriesSliceSampler(ds, cfg)
    for epoch in range(19):
        original.set_epoch(epoch)
        list(original)
    resumed = SeriesSliceSampler(ds, cfg)
    resumed.set_epoch(18)
    assert list(resumed) == list(original)
    ranks = [
        DistributedSamplerWrapper(SeriesSliceSampler(ds, cfg), num_replicas=2, rank=r)
        for r in range(2)
    ]
    for epoch in [0, 1, 18, 19]:
        for rank in ranks:
            rank.set_epoch(epoch)
        selections = [list(rank) for rank in ranks]
        assert set(selections[0]).isdisjoint(selections[1])
        rows = ds.df.iloc[selections[0] + selections[1]]
        assert len(rows) == rows.series_uid.nunique() == 10


@pytest.mark.parametrize("epochs", [50, 75, 100])
@pytest.mark.parametrize("world,batch,accumulation", [(1, 4, 1), (2, 2, 2), (2, 4, 3)])
def test_lr_budget_uses_series_partial_batches_and_accumulation(
    epochs, world, batch, accumulation
):
    ds = SeriesDataset()
    cfg = config(
        num_epochs=epochs,
        world_size=world,
        batch_size=batch,
        accumulate_grad_batches=accumulation,
    )
    configure_series_sampling_schedule(cfg, ds)
    rank_samples = 10 // world
    batches = (rank_samples + batch - 1) // batch
    steps = (batches + accumulation - 1) // accumulation
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = get_scheduler(cfg, optimizer)
    assert scheduler.total_steps == steps * epochs
    assert scheduler.warmup_steps == round(0.05 * steps * epochs)
    for step in range(scheduler.total_steps):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert scheduler.last_epoch == steps * epochs


def test_loader_keeps_last_batch_and_validation_keeps_all_slices():
    ds = SeriesDataset()
    cfg = config()
    loader = build_dataloader(cfg, ds, "train")
    assert len(loader) == 3
    rows = torch.cat([batch["index"] for batch in loader])
    assert len(rows) == 10
    val = build_dataloader(cfg, ds, "val")
    assert sum(len(batch["index"]) for batch in val) == len(ds)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"train_drop_last": True}, "drop_last"),
        ({"world_size": 3}, "split exactly"),
        ({"scheduler_params": {"total_steps": 123}}, "Stale"),
        (
            {"accumulate_grad_batches": 2, "scheduler_params": {"total_steps": 150}},
            "Stale",
        ),
        ({"args": {"limit_train_batches": 1}}, "full training epochs"),
        ({"series_positive_slice_probability": float("nan")}, "probability"),
        ({"num_iterations_per_epoch": 20}, "full-series epochs"),
    ],
)
def test_invalid_epoch_or_schedule_configuration_fails(overrides, match):
    with pytest.raises(ValueError, match=match):
        configure_series_sampling_schedule(config(**overrides), SeriesDataset())
