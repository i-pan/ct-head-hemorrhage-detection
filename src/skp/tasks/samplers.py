import numpy as np

from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Iterator, Optional

from skp.configs.base import Config


# From https://github.com/catalyst-team/catalyst
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# From https://github.com/catalyst-team/catalyst
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Yield original sampler indices assigned to the current distributed rank."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = list(super().__iter__())
        subsampler_indexes = self.dataset
        return iter([subsampler_indexes[index] for index in indexes_of_indexes])

    def __repr__(self) -> str:
        return f"DistributedSamplerWrapper.{self.sampler}"


class IterationBasedSampler(Sampler):
    """
    Define epochs based on # of iterations.
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.total_samples_per_epoch = (
            cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
        )
        self.available_indices = list(range(self.len_dataset))

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        num_samples = self.total_samples_per_epoch
        iteration_indices = []
        while num_samples > len(iteration_indices):
            difference = num_samples - len(iteration_indices)
            to_sample = min(difference, len(self.available_indices))
            sampled_indices = np.random.choice(
                self.available_indices, to_sample, replace=False
            )
            iteration_indices.extend(list(sampled_indices))
            self.available_indices = list(
                set(self.available_indices) - set(iteration_indices)
            )
            if len(self.available_indices) == 0:
                self.available_indices = list(range(self.len_dataset))
        return iter(iteration_indices)

    def __repr__(self) -> str:
        return "IterationBasedSampler"


def _normalized_sampling_weights(dataset: Dataset, expected_len: int) -> np.ndarray:
    if not hasattr(dataset, "sampling_weights"):
        raise AttributeError(
            f"{dataset.__class__.__name__} must define `sampling_weights` to use a "
            "weighted sampler."
        )
    weights = np.asarray(dataset.sampling_weights, dtype=np.float64)
    if weights.shape != (expected_len,):
        raise ValueError(
            f"Expected {expected_len} sampling weights, got shape {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Sampling weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("Sampling weights must be nonnegative.")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Sampling weights must sum to a positive value.")
    return weights / weight_sum


class WeightedSampler(Sampler):
    """
    Samples using individual sample weights.

    If class-weighted sampling is desired, can just assign weights to samples based on class.
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.sampling_weights = _normalized_sampling_weights(dataset, self.len_dataset)
        if cfg.get("num_iterations_per_epoch") is not None:
            self.total_samples_per_epoch = (
                cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
            )
        else:
            self.total_samples_per_epoch = self.len_dataset

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.len_dataset)
        sampled_indices = np.random.choice(
            indices, self.total_samples_per_epoch, replace=True, p=self.sampling_weights
        )
        return iter(sampled_indices.tolist())

    def __repr__(self) -> str:
        return "WeightedSampler"


class WeightedWithDecaySampler(Sampler):
    """
    Samples using individual sample weights.

    Decays sampling weights by alpha after each epoch:
      new_wt = current_wt ** alpha

    Lower alpha = more aggressive decay
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.sampling_weights = _normalized_sampling_weights(dataset, self.len_dataset)
        self.alpha = cfg.get("sampler_decay_alpha", 0.8) or 0.8
        if cfg.get("num_iterations_per_epoch") is not None:
            self.total_samples_per_epoch = (
                cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
            )
        else:
            self.total_samples_per_epoch = self.len_dataset

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.len_dataset)
        sampled_indices = np.random.choice(
            indices,
            self.total_samples_per_epoch,
            replace=True,
            p=self.sampling_weights,
        )
        self.sampling_weights = self.sampling_weights**self.alpha
        self.sampling_weights = self.sampling_weights / self.sampling_weights.sum()
        return iter(sampled_indices.tolist())

    def __repr__(self) -> str:
        return f"WeightedWithDecaySampler, decay_alpha={self.alpha:0.2f}"
