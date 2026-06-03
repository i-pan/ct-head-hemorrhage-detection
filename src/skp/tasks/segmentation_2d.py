import numpy as np
import torch.nn as nn
import torch

from functools import partial
from typing import Dict

from skp.tasks.base import BaseTask


def _require_monai_sliding_window():
    try:
        from monai.inferers.utils import sliding_window_inference
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MONAI is required for sliding-window inference. "
            "Install the optional 3D/medical-imaging dependencies or set "
            "cfg.use_sliding_window_inference = False."
        ) from e
    return sliding_window_inference


class ModelWrapper(nn.Module):
    # For use with MONAI sliding_window_inference function
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"x": x})["logits"]


class Task(BaseTask):
    def mixup(self, batch: Dict) -> Dict:
        x, y = batch["x"], batch["y"]
        # ensure float tensors
        assert x.dtype == torch.float, f"x.dtype is {x.dtype}, not float"
        assert y.dtype == torch.float, f"y.dtype is {y.dtype}, not float"

        batch_size = y.size(0)
        # generate lambda from beta distribution
        lamb = np.random.beta(self.cfg.mixup, self.cfg.mixup, batch_size)
        lamb = torch.from_numpy(lamb).to(x.device).float()
        # adjust lambda shape for broadcasting when multiplying
        if lamb.ndim < y.ndim:
            for _ in range(y.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        permuted_indices = torch.randperm(batch_size, device=x.device)
        # mixed label
        ymix = lamb * y + (1 - lamb) * y[permuted_indices]
        # adjust lambda shape again for input
        # which should have more dims than label
        if lamb.ndim < x.ndim:
            for _ in range(x.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        assert (
            lamb.ndim == x.ndim
        ), f"lamb has {lamb.ndim} dims whereas x has {x.ndim} dims"
        # mixed input
        xmix = lamb * x + (1 - lamb) * x[permuted_indices]
        # replace original input and label with mixed
        batch["x"] = xmix
        batch["y"] = ymix
        return batch

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.get("mixup"):
            batch = self.mixup(batch)
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k:
                self.log(k, v)
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.get("use_sliding_window_inference", False):
            out = {}
            out["logits"] = self.swi_inferer(batch["x"])
            out.update(self.model.criterion(out, batch))
        else:
            out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k:
                self.val_loss[k].append(v.detach().clone())
        for m in self.metrics:
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            m.update(out, batch)
        return out["loss"]

    def on_validation_epoch_start(self) -> None:
        if self.cfg.get("use_sliding_window_inference", False):
            sliding_window_inference = _require_monai_sliding_window()
            assert (
                self.cfg.val_batch_size == 1
            ), "val_batch_size must be 1 for sliding window inference"
            wrapped_model = ModelWrapper(self.model)
            self.swi_inferer = partial(
                sliding_window_inference,
                roi_size=(self.cfg.image_height, self.cfg.image_width),
                sw_batch_size=self.cfg.batch_size,  # training batch size should be ok
                predictor=wrapped_model,
                overlap=self.cfg.get("sliding_window_overlap", 0.25) or 0.25,
            )
