import numpy as np
import torch

from typing import Dict

from skp.tasks.base import BaseTask


class Task(BaseTask):
    @staticmethod
    def _should_log_value(key: str, value: torch.Tensor) -> bool:
        if "loss" in key:
            return True
        if key.endswith(("fraction", "count")) and isinstance(value, torch.Tensor):
            return value.ndim == 0
        return False

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
        assert lamb.ndim == x.ndim, (
            f"lamb has {lamb.ndim} dims whereas x has {x.ndim} dims"
        )
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
            if self._should_log_value(k, v):
                self.log(k, v)
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        classification_only = self.cfg.get("validate_classification_only", False)
        out = self.model(
            batch,
            return_loss=not classification_only,
            return_seg=not classification_only,
        )
        for k, v in out.items():
            if self._should_log_value(k, v):
                self.val_loss[k].append(v.detach().clone())
        for idx, m in enumerate(self.metrics):
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            if self.cfg.metrics[idx].startswith(("segmentation.", "bhsd.")):
                m.update(out["seg"], batch["seg"])
            else:
                m.update(out["cls"], batch["cls"])
        if classification_only:
            return out["cls"]["logits"].new_zeros(())
        return out["loss"]
