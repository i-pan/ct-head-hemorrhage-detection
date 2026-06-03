"""
Loads 2Dc images and masks (for center channel) for segmentation tasks.
"""

import cv2
import numpy as np
import os
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Optional, Tuple

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

    @staticmethod
    def load_image(path: str, data_dir: str, cv2_load_flag) -> List[np.ndarray]:
        # format is usually a string with filepaths separated by commas
        paths = path.split(",")
        paths = [os.path.join(data_dir, p) for p in paths]
        imgs = [cv2.imread(p, cv2_load_flag) for p in paths]
        return imgs

    @staticmethod
    def load_segmentation(path: str, data_dir: str, cv2_load_flag) -> np.ndarray:
        path = os.path.join(data_dir, path)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        x = [rearrange(img, "h w -> h w 1") if img.ndim == 2 else img for img in x]
        # assumes multiclass segmentation
        # if mask has multiple label channels, this won't work
        if self.labels[i] == "empty":
            y = np.zeros((x[0].shape[0], x[0].shape[1]))
        else:
            y = self.load_segmentation(
                self.labels[i],
                self.cfg.get("seg_data_dir") or self.cfg.data_dir,
                cv2.IMREAD_GRAYSCALE,
            )
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
        to_trf = {
            "image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)
        }
        to_trf["mask"] = y
        trf = self.transforms(**to_trf)
        x = [trf["image"]] + [trf[f"image{idx}"] for idx in range(1, len(x))]
        y = trf["mask"]
        x = np.stack(x)
        if self.mode == "train" and self.cfg.get("reverse_slice_aug") is not None:
            # reverses the slice order
            if np.random.rand() < self.cfg.reverse_slice_aug:
                x = np.ascontiguousarray(x[::-1])
        if x.shape[3] == 1:
            x = rearrange(x, "n h w c -> (n c) h w")
        else:
            # If using ConvSqueeze method, then return it as if it were a 3D tensor
            x = rearrange(x, "n h w c -> c n h w")

        x = torch.from_numpy(x)
        y = torch.tensor(y)

        x = x.float()
        # default long unless otherwise specified
        y = y.type(self.cfg.label_dtype) if self.cfg.get("label_dtype") else y.long()
        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
