"""Joint slice, sequence-series, and MIL losses for RSNA ICH."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceClassificationLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.class_names = params["class_names"]
        self.slice_weight = float(params.get("slice_weight", 1.0))
        self.series_weight = float(params.get("series_weight", 0.5))
        self.mil_weight = float(params.get("mil_weight", 0.25))
        self.register_buffer(
            "class_weights",
            torch.as_tensor(params["class_weights"], dtype=torch.float32),
            persistent=False,
        )

    def _weighted_class_mean(self, losses: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(losses.device)
        return (losses * weights).sum() / weights.sum()

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        mask = batch["valid_mask"].float()
        slice_losses = F.binary_cross_entropy_with_logits(
            out["slice_logits"].float(), batch["y"].float(), reduction="none"
        )
        per_series_class = (slice_losses * mask.unsqueeze(-1)).sum(dim=1)
        per_series_class /= mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        slice_class_losses = per_series_class.mean(dim=0)

        series_class_losses = F.binary_cross_entropy_with_logits(
            out["series_logits"].float(),
            batch["series_y"].float(),
            reduction="none",
        ).mean(dim=0)
        mil_class_losses = F.binary_cross_entropy_with_logits(
            out["mil_logits"].float(),
            batch["series_y"].float(),
            reduction="none",
        ).mean(dim=0)

        slice_loss = self._weighted_class_mean(slice_class_losses)
        series_loss = self._weighted_class_mean(series_class_losses)
        mil_loss = self._weighted_class_mean(mil_class_losses)
        result = {
            "slice_loss": slice_loss,
            "series_loss": series_loss,
            "mil_loss": mil_loss,
            "loss": self.slice_weight * slice_loss
            + self.series_weight * series_loss
            + self.mil_weight * mil_loss,
        }
        for index, name in enumerate(self.class_names):
            result[f"slice_{name}_loss"] = slice_class_losses[index]
            result[f"series_{name}_loss"] = series_class_losses[index]
            result[f"mil_{name}_loss"] = mil_class_losses[index]
        return result
