"""RSNA ICH joint classification and pseudolabel segmentation dataset."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from einops import rearrange

from skp.datasets.rsna_ich_2p5d import Dataset as RSNAICH2p5DDataset


class Dataset(RSNAICH2p5DDataset):
    def __init__(self, cfg, mode: str):
        super().__init__(cfg, mode)
        self.pseudolabel_dir = Path(cfg.pseudolabel_dir)
        self.mask_cache_size = int(cfg.get("mask_cache_size", 32) or 0)
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.df = self._merge_pseudolabel_manifest(self.df)

    def _merge_pseudolabel_manifest(self, df: pd.DataFrame) -> pd.DataFrame:
        manifest = pd.read_csv(self.cfg.pseudolabel_manifest)
        merge_columns = ["patient_id", "study_id", "series_id", "slice_path"]
        manifest_columns = [
            *merge_columns,
            "mask_path",
            "mask_index",
            "mask_nonzero_pixels",
        ]
        manifest = manifest[manifest_columns].copy()
        for column in ["patient_id", "study_id", "series_id", "slice_path"]:
            manifest[column] = manifest[column].astype(str)
        manifest["has_pseudolabel"] = True

        merged = df.merge(
            manifest,
            on=merge_columns,
            how="left",
            validate="one_to_one",
        )
        positive = merged["any"].astype(float) > 0
        missing_positive = positive & merged["has_pseudolabel"].isna()
        if missing_positive.any() and self.cfg.get(
            "require_positive_pseudolabels", True
        ):
            raise ValueError(
                f"Missing pseudolabel masks for {int(missing_positive.sum())} positive slices."
            )
        merged["has_pseudolabel"] = merged["has_pseudolabel"].notna()
        return merged

    def _get_mask_array(self, relative_path: str) -> np.ndarray:
        path = str(self.pseudolabel_dir / relative_path)
        if self.mask_cache_size <= 0:
            return np.load(path)["masks"]
        if path in self._mask_cache:
            masks = self._mask_cache.pop(path)
            self._mask_cache[path] = masks
            return masks
        masks = np.load(path)["masks"]
        self._mask_cache[path] = masks
        while len(self._mask_cache) > self.mask_cache_size:
            self._mask_cache.popitem(last=False)
        return masks

    def _load_mask(self, row: Any) -> np.ndarray:
        shape = (*self.image_size, len(self.label_columns))
        if not isinstance(row.mask_path, str) or pd.isna(row.mask_index):
            return np.zeros(shape, dtype=np.float32)

        masks = self._get_mask_array(row.mask_path)
        mask = masks[int(row.mask_index)].astype(np.float32) / 255.0
        mask = mask.transpose(1, 2, 0)
        if mask.shape != shape:
            raise ValueError(f"Expected mask shape {shape}, got {mask.shape}.")
        return mask

    def _apply_random_flips_to_image_and_mask(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim == 4 and self.depth_flip_p and np.random.random() < self.depth_flip_p:
            x = x[:, :, ::-1, :]
        if self.horizontal_flip_p and np.random.random() < self.horizontal_flip_p:
            x = x[:, ::-1, ...]
            y = y[:, ::-1, ...]
        if self.vertical_flip_p and np.random.random() < self.vertical_flip_p:
            x = x[::-1, :, ...]
            y = y[::-1, :, ...]
        return x, y

    def _load_input(self, row: Any) -> np.ndarray:
        if self.depth == 1:
            x = self._load_windowed_slice(
                row.slice_path,
                slope=row.rescale_slope,
                intercept=row.rescale_intercept,
            )
            return x.transpose(1, 2, 0)

        slices = [
            self._load_windowed_slice(
                row.prev_slice_path,
                slope=row.rescale_slope,
                intercept=row.rescale_intercept,
            ),
            self._load_windowed_slice(
                row.slice_path,
                slope=row.rescale_slope,
                intercept=row.rescale_intercept,
            ),
            self._load_windowed_slice(
                row.next_slice_path,
                slope=row.rescale_slope,
                intercept=row.rescale_intercept,
            ),
        ]
        x = np.stack(slices, axis=0)
        return rearrange(x, "d c h w -> h w (d c)")

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        row = self.df.iloc[index]
        x = self._load_input(row)
        y_seg = self._load_mask(row)

        if self.transforms is not None:
            transformed = self.transforms(image=x, mask=y_seg)
            x, y_seg = transformed["image"], transformed["mask"]

        if self.depth == 1:
            x, y_seg = self._apply_random_flips_to_image_and_mask(x, y_seg)
            x = x.transpose(2, 0, 1)
        else:
            x = x.reshape(*x.shape[:2], self.depth, len(self.windows))
            x, y_seg = self._apply_random_flips_to_image_and_mask(x, y_seg)
            if self.flatten_depth_to_channels:
                x = x.transpose(2, 3, 0, 1).reshape(
                    self.depth * len(self.windows),
                    *x.shape[:2],
                )
            else:
                x = x.transpose(3, 2, 0, 1)

        x = np.ascontiguousarray(x)
        y_seg = np.ascontiguousarray(y_seg.transpose(2, 0, 1))
        y_cls = row[self.label_columns].to_numpy(dtype=np.float32)
        series_uid = row.series_uid

        return {
            "cls": {
                "y": torch.from_numpy(y_cls).float(),
                "group_index": torch.tensor(self.series_index[series_uid]).long(),
            },
            "seg": {
                "x": torch.from_numpy(x).float(),
                "y": torch.from_numpy(y_seg).float(),
            },
        }
