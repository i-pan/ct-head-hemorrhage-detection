"""Memory-mapped series dataset for pre-extracted RSNA ICH features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from skp.datasets._utils import get_collate_fn, validate_mode


class Dataset(TorchDataset):
    def __init__(self, cfg, mode: str):
        validate_mode(mode)
        if mode not in {"train", "val", "test"}:
            raise ValueError("ich_sequence_features supports train, val, and test.")
        self.cfg = cfg
        self.mode = mode
        self.root = Path(cfg.feature_dir)
        self.max_length = int(cfg.max_sequence_length)
        self.reverse_p = float(cfg.get("sequence_reverse_p", 0.0) or 0.0)
        self.collate_fn = get_collate_fn(mode)

        self.series = pd.read_csv(self.root / f"{mode}_series.csv")
        self.features = np.load(
            self.root / f"{mode}_features.npy", mmap_mode="r"
        )
        self.base_logits = np.load(
            self.root / f"{mode}_base_logits.npy", mmap_mode="r"
        )
        self.labels = np.load(self.root / f"{mode}_labels.npy", mmap_mode="r")
        if not (len(self.features) == len(self.base_logits) == len(self.labels)):
            raise ValueError(f"Mismatched extracted array lengths for split={mode}.")

    def __len__(self) -> int:
        return len(self.series)

    @staticmethod
    def _resample_indices(length: int, max_length: int) -> np.ndarray:
        if length <= max_length:
            return np.arange(length, dtype=np.int64)
        return np.rint(np.linspace(0, length - 1, max_length)).astype(np.int64)

    @staticmethod
    def restore_predictions(
        predictions: np.ndarray,
        sampled_indices: np.ndarray,
        original_length: int,
    ) -> np.ndarray:
        """Nearest-neighbor map sampled predictions to original slice indices."""
        predictions = np.asarray(predictions)
        sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
        if len(predictions) != len(sampled_indices):
            raise ValueError("predictions and sampled_indices must have equal length.")
        original_indices = np.arange(original_length, dtype=np.int64)
        right = np.searchsorted(sampled_indices, original_indices, side="left")
        right = np.clip(right, 0, len(sampled_indices) - 1)
        left = np.clip(right - 1, 0, len(sampled_indices) - 1)
        choose_left = (
            np.abs(original_indices - sampled_indices[left])
            <= np.abs(sampled_indices[right] - original_indices)
        )
        nearest = np.where(choose_left, left, right)
        return predictions[nearest]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.series.iloc[index]
        offset, original_length = int(row.offset), int(row.length)
        sampled = self._resample_indices(original_length, self.max_length)
        source_rows = offset + sampled

        features = np.asarray(self.features[source_rows], dtype=np.float32).copy()
        base_logits = np.asarray(
            self.base_logits[source_rows], dtype=np.float32
        ).copy()
        labels = np.asarray(self.labels[source_rows], dtype=np.float32).copy()
        positions = sampled.astype(np.float32) / max(original_length - 1, 1)

        if self.mode == "train" and self.reverse_p > 0:
            if np.random.random() < self.reverse_p:
                features = features[::-1].copy()
                base_logits = base_logits[::-1].copy()
                labels = labels[::-1].copy()
                positions = positions[::-1].copy()
                sampled = sampled[::-1].copy()

        length = len(sampled)
        feature_dim = features.shape[1]
        num_classes = labels.shape[1]
        x = np.zeros((self.max_length, feature_dim), dtype=np.float32)
        y = np.zeros((self.max_length, num_classes), dtype=np.float32)
        baseline = np.zeros((self.max_length, num_classes), dtype=np.float32)
        position = np.zeros(self.max_length, dtype=np.float32)
        source_index = np.full(self.max_length, -1, dtype=np.int64)
        valid_mask = np.zeros(self.max_length, dtype=bool)
        x[:length] = features
        y[:length] = labels
        baseline[:length] = base_logits
        position[:length] = positions
        source_index[:length] = sampled
        valid_mask[:length] = True

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "series_y": torch.from_numpy(labels.max(axis=0)),
            "base_logits": torch.from_numpy(baseline),
            "position": torch.from_numpy(position),
            "valid_mask": torch.from_numpy(valid_mask),
            "length": torch.tensor(length, dtype=torch.long),
            "original_length": torch.tensor(original_length, dtype=torch.long),
            "source_index": torch.from_numpy(source_index),
            "group_index": torch.tensor(index, dtype=torch.long),
        }
