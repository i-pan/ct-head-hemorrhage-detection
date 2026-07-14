"""RSNA ICH CT classification dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from skp.configs import Config
from skp.datasets._utils import get_collate_fn, get_transforms, validate_mode
from skp.toolbox.images import center_crop_or_pad_borders


DEFAULT_LABEL_COLUMNS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


def _slice_sort_key(filename: str) -> str:
    stem = Path(filename).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return f"~|{stem}"
    return f"{int(numbers[0]):020d}|{stem}"


def _series_uid(df: pd.DataFrame) -> pd.Series:
    return df["patient_id"] + "/" + df["study_id"] + "/" + df["series_id"]


def _stringify_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["patient_id", "study_id", "series_id"]:
        df[column] = df[column].astype(str)
    return df


class DepthFlip(A.ImageOnlyTransform):
    """Reverse slice order for flattened 2.5D images."""

    def __init__(self, depth: int, channels_per_slice: int, p: float = 0.5):
        super().__init__(p=p)
        self.depth = depth
        self.channels_per_slice = channels_per_slice

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        x = rearrange(
            image,
            "h w (d c) -> h w d c",
            d=self.depth,
            c=self.channels_per_slice,
        )
        x = x[:, :, ::-1, :]
        x = rearrange(x, "h w d c -> h w (d c)")
        return np.ascontiguousarray(x)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("depth", "channels_per_slice")


class Dataset(TorchDataset):
    def __init__(self, cfg: Config, mode: str):
        validate_mode(mode)
        self.cfg = cfg
        self.mode = mode
        self.transforms = get_transforms(cfg, mode)
        self.collate_fn = get_collate_fn(mode)
        self.label_columns = cfg.get("label_columns") or DEFAULT_LABEL_COLUMNS
        self.image_size = (cfg.image_height, cfg.image_width)
        self.large_image_resize_threshold = int(
            cfg.get("large_image_resize_threshold", 640) or 640
        )
        self.windows = cfg.get("ct_windows") or [
            (40, 80),
            (80, 200),
            (600, 2800),
        ]
        windows = np.asarray(self.windows, dtype=np.float32)
        self.window_lowers = windows[:, 0] - windows[:, 1] / 2
        self.window_uppers = windows[:, 0] + windows[:, 1] / 2
        self.window_widths = windows[:, 1]
        self.depth = cfg.get("num_slices") or 3
        if self.depth not in {1, 3}:
            raise ValueError("rsna_ich_2p5d currently expects cfg.num_slices = 1 or 3.")
        self.flatten_depth_to_channels = cfg.get("flatten_depth_to_channels", False)
        self.depth_flip_p = cfg.get("depth_flip_p", 0.0) if mode == "train" else 0.0
        self.horizontal_flip_p = (
            cfg.get("horizontal_flip_p", 0.0) if mode == "train" else 0.0
        )
        self.vertical_flip_p = (
            cfg.get("vertical_flip_p", 0.0) if mode == "train" else 0.0
        )

        df = self._load_annotations()
        df = self._filter_mode(df, mode).reset_index(drop=True)
        if mode == "train" and cfg.get("train_positive_series_only", False):
            positive_series = df.groupby("series_uid")["any"].transform("max") > 0
            df = df.loc[positive_series].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows selected for mode={mode!r}.")

        self.df = df
        self.series_index = {
            series_uid: idx
            for idx, series_uid in enumerate(sorted(df["series_uid"].unique()))
        }

    def _load_annotations(self) -> pd.DataFrame:
        split_df = pd.read_csv(self.cfg.annotations_file)
        split_df = _stringify_identifiers(split_df)
        split_df = split_df.loc[~split_df["excluded_bhsd"]].copy()
        split_df["filename"] = split_df["slice_path"].map(lambda path: Path(path).name)
        split_df["slice_sort_key"] = split_df["filename"].map(_slice_sort_key)
        split_df["series_uid"] = _series_uid(split_df)

        labels = pd.read_csv(self.cfg.labels_file)
        labels = labels.rename(
            columns={
                "patient_ID": "patient_id",
                "study_ID": "study_id",
                "series_ID": "series_id",
            }
        )
        labels = _stringify_identifiers(labels)
        label_cols = [
            "patient_id",
            "study_id",
            "series_id",
            "filename",
            *self.label_columns,
        ]
        labels = labels[label_cols]

        rescale = pd.read_csv(self.cfg.rescale_file)
        rescale = _stringify_identifiers(rescale)
        df = split_df.merge(
            labels,
            on=["patient_id", "study_id", "series_id", "filename"],
            how="left",
            validate="one_to_one",
        )
        df = df.merge(
            rescale,
            on=["patient_id", "study_id", "series_id"],
            how="left",
            validate="many_to_one",
        )

        missing_labels = df[self.label_columns].isna().any(axis=1)
        if missing_labels.any():
            raise ValueError(f"Missing labels for {int(missing_labels.sum())} slices.")
        missing_rescale = df[["rescale_slope", "rescale_intercept"]].isna().any(axis=1)
        if missing_rescale.any():
            raise ValueError(
                f"Missing rescale values for {int(missing_rescale.sum())} slices."
            )

        df = df.sort_values(
            ["patient_id", "study_id", "series_id", "slice_sort_key", "filename"]
        ).reset_index(drop=True)
        df["prev_slice_path"] = df.groupby("series_uid")["slice_path"].shift(1)
        df["next_slice_path"] = df.groupby("series_uid")["slice_path"].shift(-1)
        return df

    def _filter_mode(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "train":
            if (df["split"] == "val").any():
                return df.loc[df["split"] == "train"]
            return df.loc[(df["split"] == "train") & (df["fold"] != self.cfg.fold)]
        if mode == "val":
            if (df["split"] == "val").any():
                return df.loc[df["split"] == "val"]
            return df.loc[(df["split"] == "train") & (df["fold"] == self.cfg.fold)]
        if mode == "test":
            return df.loc[df["split"] == "test"]
        return df

    def __len__(self) -> int:
        return len(self.df)

    def _load_windowed_slice(
        self,
        relative_path: Any,
        *,
        slope: float,
        intercept: float,
    ) -> np.ndarray:
        if not isinstance(relative_path, str) or relative_path == "":
            return np.zeros((len(self.windows), *self.image_size), dtype=np.float32)

        path = Path(self.cfg.data_dir) / relative_path
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale image, got {image.shape}: {path}"
            )

        if any(
            dimension > self.large_image_resize_threshold
            for dimension in image.shape[:2]
        ):
            image = cv2.resize(
                image.astype(np.float32),
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_AREA,
            )
        elif image.shape != self.image_size:
            image = center_crop_or_pad_borders(image, self.image_size, pad_val=0)

        hu = image.astype(np.float32)
        hu *= float(slope)
        hu += float(intercept)
        windowed = np.clip(
            hu[None],
            self.window_lowers[:, None, None],
            self.window_uppers[:, None, None],
        )
        windowed -= self.window_lowers[:, None, None]
        windowed /= self.window_widths[:, None, None] + 1e-6
        return windowed

    def _apply_random_flips(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 4 and self.depth_flip_p and np.random.random() < self.depth_flip_p:
            x = x[:, :, ::-1, :]
        if self.horizontal_flip_p and np.random.random() < self.horizontal_flip_p:
            x = x[:, ::-1, ...]
        if self.vertical_flip_p and np.random.random() < self.vertical_flip_p:
            x = x[::-1, :, ...]
        return x

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        if self.depth == 1:
            x = self._load_windowed_slice(
                row.slice_path,
                slope=row.rescale_slope,
                intercept=row.rescale_intercept,
            )
            x = x.transpose(1, 2, 0)  # H, W, C
            if self.transforms is not None:
                x = self.transforms(image=x)["image"]
            x = self._apply_random_flips(x)
            x = x.transpose(2, 0, 1)  # C, H, W
        else:
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
            x = np.stack(slices, axis=0)  # D, C, H, W
            x = rearrange(x, "d c h w -> h w (d c)")
            if self.transforms is not None:
                x = self.transforms(image=x)["image"]
            x = x.reshape(*x.shape[:2], self.depth, len(self.windows))
            x = self._apply_random_flips(x)
            if self.flatten_depth_to_channels:
                x = x.transpose(2, 3, 0, 1).reshape(
                    self.depth * len(self.windows),
                    *x.shape[:2],
                )
            else:
                x = x.transpose(3, 2, 0, 1)
        x = np.ascontiguousarray(x)

        y = row[self.label_columns].to_numpy(dtype=np.float32)
        series_uid = row.series_uid
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
            "index": torch.tensor(index),
            "group_index": torch.tensor(self.series_index[series_uid]).long(),
        }
