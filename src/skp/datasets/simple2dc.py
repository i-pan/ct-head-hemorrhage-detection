"""
Loads 2Dc images for simple classification/regression tasks, along extra variable for embedding, if specified.
2Dc is mainly used for 3D images that have been converted to 2D slices.
For example, if we have a CT, we can convert it to 2Dc by taking 3 contiguous slices and forming
a pseudo RGB channel.
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
        self.labels = df[self.cfg.targets].values
        if self.cfg.get("vars"):
            self.vars = df[self.cfg.vars].values

        self.collate_fn = get_collate_fn(mode)

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str) -> List[np.ndarray]:
        # format is usually a string with filepaths separated by commas
        paths = path.split(",")
        paths = [os.path.join(self.cfg.data_dir, p) for p in paths]
        imgs = [cv2.imread(p, self.cfg.cv2_load_flag) for p in paths]
        imgs = [
            rearrange(img, "h w -> h w 1") if img.ndim == 2 else img for img in imgs
        ]
        return imgs

    def _get(self, i: int) -> Tuple[List[np.ndarray], np.ndarray]:
        x = self.load_image(self.inputs[i])
        y = self.labels[i].copy()
        return x, y

    def get(self, i: int) -> Optional[Tuple]:
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
        # apply same transforms to each of the images
        x = {"image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)}
        x_trf = self.transforms(**x)
        x = [x_trf["image"]] + [x_trf[f"image{idx}"] for idx in range(1, len(x))]
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

        x, y = x.float(), y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        if hasattr(self, "vars"):
            input_dict["var"] = torch.tensor(self.vars[i]).long()

        return input_dict
