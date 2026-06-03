"""
Loads 2D images and masks for combined segmentation-classification tasks.
e.g., pneumonia segmentation and classification
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
        self.masks = df[self.cfg.masks].tolist()
        self.labels = df[self.cfg.targets].values

        self.collate_fn = get_collate_fn(mode)

    def __len__(self) -> int:
        return len(self.inputs)

    @staticmethod
    def load_image(
        path: str,
        data_dir: str,
        cv2_load_flag,
        assume_empty_if_not_present: bool = False,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        path = os.path.join(data_dir, path)
        if assume_empty_if_not_present and not os.path.exists(path):
            return np.zeros(img_shape)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        # assumes multiclass segmentation
        # if mask has multiple label channels, this won't work
        y = self.load_image(
            self.masks[i],
            self.cfg.get("seg_data_dir") or self.cfg.data_dir,
            cv2.IMREAD_GRAYSCALE,
            assume_empty_if_not_present=self.cfg.assume_mask_empty_if_not_present,
            img_shape=x.shape[:2],
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

        # shared encoder and x goes through segmentation model first
        # so cls does not need x
        input_dict = {"cls": {"y": torch.tensor(self.labels[i])}, "seg": {"x": x, "y": y}}
        return input_dict
