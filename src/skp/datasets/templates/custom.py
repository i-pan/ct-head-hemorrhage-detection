"""
Template for a custom SKP dataset.

Copy this file into `src/skp/datasets/`, rename it, then point `cfg.dataset`
at it. For example, `src/skp/datasets/my_dataset.py` can be used with:

    cfg.dataset = "my_dataset"

This template mirrors the common reusable loaders: read annotations, filter by
mode, select transforms, load samples, optionally retry failed samples, and
return a dictionary consumed by SKP tasks/models.
"""

import cv2
import numpy as np
import os
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from skp.configs import Config
from skp.datasets._utils import filter_by_mode, get_collate_fn, get_transforms, read_annotations


class Dataset(TorchDataset):
    def __init__(self, cfg: Config, mode: str):
        self.cfg = cfg
        self.mode = mode

        df = filter_by_mode(read_annotations(cfg), cfg, mode)
        self.transforms = get_transforms(cfg, mode)

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values

        if self.cfg.get("vars"):
            self.vars = df[self.cfg.vars].values

        if self.cfg.get("sampling_weight_col"):
            self.sampling_weights = df[self.cfg.sampling_weight_col].values

        if self.cfg.get("group_index_col"):
            self.group_index = df[self.cfg.group_index_col].values

        self.collate_fn = get_collate_fn(mode)

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        x = cv2.imread(path, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        return x

    def _get(self, i: int):
        x = self.load_image(self.inputs[i])
        y = self.labels[i].copy()
        return x, y

    def get(self, i: int):
        if self.cfg.get("skip_failed_data", False):
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        return self._get(i)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data
        x = self.transforms(image=x)["image"]
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")

        y = torch.tensor(y)
        x, y = x.float(), y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        if hasattr(self, "vars"):
            input_dict["var"] = torch.tensor(self.vars[i]).long()

        if hasattr(self, "group_index"):
            input_dict["group_index"] = torch.tensor(self.group_index[i])

        return input_dict
