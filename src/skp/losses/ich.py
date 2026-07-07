"""Project-specific losses for RSNA ICH classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


DEFAULT_CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.class_names = params.get("class_names") or DEFAULT_CLASS_NAMES
        class_weights = params.get("class_weights") or [1, 1, 1, 1, 1, 5]
        if len(class_weights) != len(self.class_names):
            raise ValueError("class_weights and class_names must have the same length.")
        self.register_buffer(
            "class_weights",
            torch.as_tensor(class_weights, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        logits, targets = out["logits"], batch["y"]
        losses = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="none"
        )
        per_class_loss = losses.mean(dim=0)
        weighted = per_class_loss * self.class_weights.to(per_class_loss.device)
        loss_dict = {
            f"{name}_loss": value
            for name, value in zip(self.class_names, per_class_loss)
        }
        loss_dict["loss"] = weighted.sum() / self.class_weights.sum()
        return loss_dict
