import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from skp.configs import Config
from skp.tasks import samplers
from skp.tasks.base import BaseTask
from skp.tasks.utils import build_dataloader


class TinyDataset(Dataset):
    def __init__(self, n=5):
        self.n = n
        self.sampling_weights = np.ones(n)
        self.collate_fn = None

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"x": torch.tensor(index), "y": torch.tensor(index)}


def _loader_cfg(**overrides):
    cfg = Config(
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        batch_size=2,
        val_batch_size=None,
        sampler=None,
        args={"strategy": "auto"},
    )
    cfg.__dict__.update(overrides)
    return cfg


def test_build_dataloader_omits_persistent_workers_with_zero_workers():
    loader = build_dataloader(_loader_cfg(), TinyDataset(), mode="train")

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.batch_size == 2


def test_weighted_sampler_validates_weights():
    dataset = TinyDataset()
    dataset.sampling_weights = np.zeros(len(dataset))

    with pytest.raises(ValueError, match="positive"):
        samplers.WeightedSampler(dataset, Config())


def test_distributed_sampler_wrapper_handles_single_index(monkeypatch):
    base_sampler = torch.utils.data.SequentialSampler([0])

    monkeypatch.setattr(
        torch.distributed,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    wrapper = samplers.DistributedSamplerWrapper(base_sampler, num_replicas=1, rank=0)

    assert list(wrapper) == [0]


def test_base_task_returns_optimizer_only_when_scheduler_is_disabled():
    task = BaseTask(Config())
    task.optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=1e-3)

    assert task.configure_optimizers() == {"optimizer": task.optimizer}
