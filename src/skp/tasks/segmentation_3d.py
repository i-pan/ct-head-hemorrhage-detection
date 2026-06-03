import numpy as np
import torch.nn as nn
import torch

from functools import partial
from lightning.pytorch.utilities import grad_norm
from typing import Dict

from skp.tasks.base import BaseTask


def _require_monai_mixers():
    try:
        from monai.transforms import CutMix, MixUp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MONAI is required for 3D MixUp/CutMix. Install the optional "
            "3D/medical-imaging dependencies or disable cfg.mixup/cfg.cutmix."
        ) from e
    return CutMix, MixUp


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


def _sliding_window_roi_size(cfg):
    roi_size = (
        cfg.get("num_slices") or cfg.get("dim0"),
        cfg.get("image_height") or cfg.get("dim1"),
        cfg.get("image_width") or cfg.get("dim2"),
    )
    if any(size is None for size in roi_size):
        raise AttributeError(
            "Sliding-window inference requires num_slices/image_height/image_width "
            "or legacy dim0/dim1/dim2 config values."
        )
    return roi_size



class ModelWrapper(nn.Module):
    # For use with MONAI sliding_window_inference function
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"x": x})["logits"]


class StochasticModelWrapper(nn.Module):
    # For use with MONAI sliding_window_inference function
    def __init__(self, model, class_indices=None):
        super().__init__()
        self.model = model
        self.class_indices = class_indices

    def forward(self, x):
        return self.model({"x": x}, indices=self.class_indices)["logits"]


class Task(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.mixer = []
        if self.cfg.get("mix_proba") is not None:
            CutMix, MixUp = _require_monai_mixers()
            if self.cfg.get("mixup") is not None:
                self.mixer.append(
                    MixUp(batch_size=self.cfg.batch_size, alpha=self.cfg.mixup)
                )
            if self.cfg.get("cutmix") is not None:
                self.mixer.append(
                    CutMix(batch_size=self.cfg.batch_size, alpha=self.cfg.cutmix)
                )

        if len(self.mixer) == 1:
            self.mixer_weights = [1]
        elif len(self.mixer) == 2:
            self.mixer_weights = [
                self.cfg.get("mixup_weight", 0.5) or 0.5,
                self.cfg.get("cutmix_weight", 0.5) or 0.5,
            ]

        if self.cfg.get("mix_proba") is not None and len(self.mixer) == 0:
            raise ValueError("Set cfg.mixup and/or cfg.cutmix when cfg.mix_proba is set.")

        if len(self.mixer) > 0:
            assert (
                sum(self.mixer_weights) == 1
            ), f"mixer weights must sum to 1, got {sum(self.mixer_weights)}"
            assert (
                len(self.mixer_weights) == len(self.mixer)
            ), f"expected {len(self.mixer)} mixer weights, got {len(self.mixer_weights)}"

    def mixup(self, batch: Dict) -> Dict:
        x, y = batch["x"], batch["y"]
        # ensure float tensors
        assert x.dtype == torch.float, f"x.dtype is {x.dtype}, not float"
        assert y.dtype == torch.float, f"y.dtype is {y.dtype}, not float"

        batch_size = y.size(0)

        # Generate lambda from beta distribution
        lamb = np.random.beta(self.cfg.mixup, self.cfg.mixup, batch_size)
        lamb = torch.from_numpy(lamb).to(x.device)

        # Adjust lambda shape for broadcasting when multiplying
        if lamb.ndim < y.ndim:
            for _ in range(y.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)

        # Permute indices
        permuted_indices = torch.randperm(batch_size, device=x.device)

        # Mixed label
        ymix = lamb * y + (1 - lamb) * y[permuted_indices]

        # Adjust lambda shape again for input
        if lamb.ndim < x.ndim:
            for _ in range(x.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)

        assert (
            lamb.ndim == x.ndim
        ), f"lamb has {lamb.ndim} dims whereas x has {x.ndim} dims"

        # Mixed input
        xmix = lamb * x + (1 - lamb) * x[permuted_indices]

        # Replace original input and label with mixed
        batch["x"] = xmix
        batch["y"] = ymix
        return batch

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.get("mix_proba") is not None:
            if np.random.rand() < self.cfg.mix_proba:
                mix = np.random.choice(self.mixer, p=self.mixer_weights)
                x, y = mix(batch["x"], batch["y"])
                batch["x"] = x
                batch["y"] = y
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k:
                self.log(k, v)
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.get("use_sliding_window_inference", False):
            out = {}
            out["logits"] = self.swi_inferer(batch["x"])
            if "net_stochastic" in self.cfg.model:
                out["class_indices"] = self.cfg.class_indices_for_val
            if self.cfg.get("deep_supervision", False):
                out.update(self.model.criterion(out, batch, mode="val"))
            else:
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
            if "net_stochastic" in self.cfg.model:
                wrapped_model = StochasticModelWrapper(
                    self.model, self.cfg.class_indices_for_val
                )
            else:
                wrapped_model = ModelWrapper(self.model)
            self.swi_inferer = partial(
                sliding_window_inference,
                roi_size=_sliding_window_roi_size(self.cfg),
                sw_batch_size=self.cfg.batch_size,  # training batch size should be ok
                predictor=wrapped_model,
                overlap=self.cfg.get("sliding_window_overlap", 0.25) or 0.25,
            )

    def on_before_optimizer_step(self, optimizer):
        if not self.cfg.get("log_grad_norm", False):
            return
        norms = grad_norm(self, norm_type=2)
        self.log("total_grad_norm_sum", norms["grad_2.0_norm_total"])
