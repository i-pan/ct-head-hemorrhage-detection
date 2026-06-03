"""
Loads 2D images and masks for segmentation tasks.
"""

import cv2
import numpy as np
import os
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, Optional, Tuple

from skp.configs import Config
from skp.datasets._utils import filter_by_mode, get_collate_fn, get_transforms, read_annotations


class Dataset(TorchDataset):
    def __init__(self, cfg: Config, mode: str):
        self.cfg = cfg
        self.mode = mode
        df = filter_by_mode(read_annotations(cfg), cfg, mode)
        self.transforms = get_transforms(cfg, mode)

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].tolist()

        self.collate_fn = get_collate_fn(mode)

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str, data_dir: str, cv2_load_flag) -> np.ndarray:
        path = os.path.join(data_dir, path)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        # assumes multiclass segmentation
        # if mask has multiple label channels, this won't work
        y = self.load_image(
            self.labels[i],
            self.cfg.get("seg_data_dir") or self.cfg.data_dir,
            cv2.IMREAD_GRAYSCALE,
        )
        if self.cfg.get("rescale_mask"):
            # sometimes for binary masks will save as 255 instead of 1
            y = (y / self.cfg.rescale_mask).astype("int")
        return x, y

    def get(self, i: int) -> Optional[Tuple[np.ndarray]]:
        if self.cfg.get("skip_failed_data", False):
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        else:
            return self._get(i)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data
        trf = self.transforms(image=x, mask=y)
        x, y = trf["image"], trf["mask"]
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")
        y = torch.tensor(y)

        x = x.float()
        # default long unless otherwise specified
        y = y.type(self.cfg.label_dtype) if self.cfg.get("label_dtype") else y.long()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
