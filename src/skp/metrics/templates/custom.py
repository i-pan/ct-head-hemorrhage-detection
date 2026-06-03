"""
Template for a custom SKP metric.

Copy this file into `src/skp/metrics/`, rename it, and point `cfg.metrics` at it.
For example, `src/skp/metrics/my_metrics.py` with class `CustomMetric` can be
used as:

    cfg.metrics = ["my_metrics.CustomMetric"]
"""

import torch

from torchmetrics import Metric
from typing import Dict

from skp.configs import Config
from skp.metrics import utils


def _cat_state(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state
    return torch.cat(state, dim=0)


class CustomMetric(Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].detach().float())
        self.t.append(batch["y"].detach().float())

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0)
        t = utils.distributed_concat(_cat_state(self.t), dim=0)
        return {"custom_metric": torch.mean(torch.abs(p - t))}
