"""Slice and series AUROC metrics for the RSNA ICH sequence model."""

from __future__ import annotations

from typing import Dict

import torch
from torchmetrics import Metric

from skp.metrics import utils


def _cat(state) -> torch.Tensor:
    return state if isinstance(state, torch.Tensor) else torch.cat(state, dim=0)


class _AUROC(Metric):
    prediction_key = ""
    target_key = ""
    prefix = ""
    is_slice = False

    def __init__(self, cfg, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)
        self.class_names = cfg.label_columns
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        predictions = out[self.prediction_key].detach().float()
        targets = batch[self.target_key].detach().float()
        if self.is_slice:
            mask = batch["valid_mask"].bool()
            predictions = predictions[mask]
            targets = targets[mask]
        self.p.append(predictions)
        self.t.append(targets)

    def compute(self) -> Dict[str, torch.Tensor]:
        predictions = utils.distributed_concat(_cat(self.p), dim=0).cpu().sigmoid()
        targets = utils.distributed_concat(_cat(self.t), dim=0).cpu()
        metrics = {
            f"{self.prefix}auc_{name}": utils.auc(targets[:, index], predictions[:, index])
            for index, name in enumerate(self.class_names)
        }
        metrics[f"{self.prefix}auc_mean"] = torch.stack(list(metrics.values())).mean()
        return metrics


class ContextualSliceAUROC(_AUROC):
    prediction_key = "slice_logits"
    target_key = "y"
    prefix = "slice_"
    is_slice = True


class ContextualSeriesAUROC(_AUROC):
    prediction_key = "series_logits"
    target_key = "series_y"
    prefix = "series_"


class MILSeriesAUROC(_AUROC):
    prediction_key = "mil_logits"
    target_key = "series_y"
    prefix = "mil_"


class BaselineSliceAUROC(_AUROC):
    prediction_key = "base_logits"
    target_key = "y"
    prefix = "baseline_slice_"
    is_slice = True

    def update(self, out: Dict, batch: Dict) -> None:
        baseline_out = {"base_logits": batch["base_logits"]}
        super().update(baseline_out, batch)


class BaselineMaxSeriesAUROC(_AUROC):
    prediction_key = "base_series_logits"
    target_key = "series_y"
    prefix = "baseline_series_max_"

    def update(self, out: Dict, batch: Dict) -> None:
        logits = batch["base_logits"].detach().float()
        mask = batch["valid_mask"].bool()
        probabilities = logits.sigmoid().masked_fill(~mask.unsqueeze(-1), -1.0)
        max_probabilities = probabilities.amax(dim=1).clamp(1e-6, 1 - 1e-6)
        max_logits = torch.logit(max_probabilities)
        baseline_out = {"base_series_logits": max_logits}
        super().update(baseline_out, batch)
