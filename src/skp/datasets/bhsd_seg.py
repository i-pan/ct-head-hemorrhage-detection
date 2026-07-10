"""BHSD ICH multilabel segmentation dataset."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset as TorchDataset

from skp.configs import Config
from skp.datasets._utils import (
    filter_by_mode,
    get_collate_fn,
    get_transforms,
    read_annotations,
    validate_mode,
)


MASK_VALUES = [1, 2, 3, 4, 5]


class Dataset(TorchDataset):
    def __init__(self, cfg: Config, mode: str):
        validate_mode(mode)
        self.cfg = cfg
        self.mode = mode
        self.transforms = get_transforms(cfg, mode)
        self.collate_fn = get_collate_fn(mode)
        self.image_size = (cfg.image_height, cfg.image_width)
        self.num_slices = cfg.get("num_slices") or 1
        if self.num_slices not in {1, 3}:
            raise ValueError("bhsd_seg expects cfg.num_slices to be 1 or 3.")
        self.mask_cache_size = int(cfg.get("mask_cache_size", 16) or 0)
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        df = read_annotations(cfg)
        self.series_to_index = {
            series_uid: index
            for index, series_uid in enumerate(sorted(df["series_uid"].unique()))
        }
        df = filter_by_mode(df, cfg, mode).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No BHSD rows selected for mode={mode!r}.")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}: {path}")
        if image.shape[:2] != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {image.shape[:2]}: {path}"
            )
        return image.astype(np.float32) / 255.0

    def _get_mask_volume(self, path: str | Path) -> np.ndarray:
        path = str(path)
        if self.mask_cache_size <= 0:
            return np.load(path, mmap_mode="r")
        if path in self._mask_cache:
            volume = self._mask_cache.pop(path)
            self._mask_cache[path] = volume
            return volume
        volume = np.load(path, mmap_mode="r")
        self._mask_cache[path] = volume
        while len(self._mask_cache) > self.mask_cache_size:
            self._mask_cache.popitem(last=False)
        return volume

    def _load_mask(self, row) -> np.ndarray:
        volume = self._get_mask_volume(row.mask_volume_path)
        mask = np.asarray(volume[..., int(row.slice_index)])
        channels = [(mask == value).astype(np.float32) for value in MASK_VALUES]
        channels.append((mask > 0).astype(np.float32))
        return np.stack(channels, axis=-1)

    def _load_input(self, row) -> np.ndarray:
        if self.num_slices == 1:
            return self._load_image(row.image_path)
        images = [
            self._load_image(row.prev_image_path),
            self._load_image(row.image_path),
            self._load_image(row.next_image_path),
        ]
        return np.concatenate(images, axis=-1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        x = self._load_input(row)
        y = self._load_mask(row)

        if self.transforms is not None:
            transformed = self.transforms(image=x, mask=y)
            x, y = transformed["image"], transformed["mask"]

        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        y = np.ascontiguousarray(y.transpose(2, 0, 1))
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
            "index": torch.tensor(index),
            "group_index": torch.tensor(self.series_to_index[row.series_uid]).long(),
            "slice_index": torch.tensor(int(row.slice_index)).long(),
            "spacing_2d": torch.tensor(
                [float(row.row_spacing_mm), float(row.col_spacing_mm)],
                dtype=torch.float32,
            ),
            "spacing_3d": torch.tensor(
                [
                    float(row.row_spacing_mm),
                    float(row.col_spacing_mm),
                    float(row.slice_spacing_mm),
                ],
                dtype=torch.float32,
            ),
        }
