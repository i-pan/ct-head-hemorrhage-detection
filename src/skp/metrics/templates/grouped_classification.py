"""
Template for grouped classification metrics.

This is useful for study/patient/breast-level aggregation where the model predicts
on individual images or slices but validation should also score grouped examples.
"""

import torch

from collections import defaultdict
from torchmetrics import Metric
from typing import Callable, Dict, Tuple

from skp.configs import Config
from skp.metrics import utils


def _cat_state(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state
    return torch.cat(state, dim=0)


class GroupedAUROC(Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("group_index", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].detach().float())
        self.t.append(batch["y"].detach().float())
        self.group_index.append(batch["group_index"].detach())

    @staticmethod
    def aggregate(
        p_dict: Dict[int, list[torch.Tensor]],
        t_dict: Dict[int, list[torch.Tensor]],
        agg_func: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert agg_func in {
            torch.mean,
            torch.amax,
        }, "only torch.mean and torch.amax are valid aggregation functions"
        p = torch.stack([agg_func(torch.stack(v), dim=0) for v in p_dict.values()])
        t = torch.stack([agg_func(torch.stack(v), dim=0) for v in t_dict.values()])
        return p, t

    @staticmethod
    def compute_aucs(
        t: torch.Tensor, p: torch.Tensor, suffix: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        for class_idx in range(p.size(1)):
            metrics_dict[f"auc{class_idx}_{suffix}"] = utils.auc(
                t[:, class_idx], p[:, class_idx]
            )
        metrics_dict[f"auc_mean_{suffix}"] = torch.stack(
            list(metrics_dict.values())
        ).mean()
        return metrics_dict

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        group_index = utils.distributed_concat(
            _cat_state(self.group_index), dim=0
        ).cpu()

        metrics_dict = self.compute_aucs(t, p, "sample")
        p_dict, t_dict = defaultdict(list), defaultdict(list)
        for each_p, each_t, each_index in zip(p, t, group_index):
            group_id = each_index.item()
            p_dict[group_id].append(each_p)
            t_dict[group_id].append(each_t)

        p_mean, t_mean = self.aggregate(p_dict, t_dict, torch.mean)
        metrics_dict.update(self.compute_aucs(t_mean, p_mean, "group_mean"))

        p_max, t_max = self.aggregate(p_dict, t_dict, torch.amax)
        metrics_dict.update(self.compute_aucs(t_max, p_max, "group_max"))

        metrics_dict["auc_mean_group"] = torch.stack(
            [
                metrics_dict["auc_mean_group_mean"],
                metrics_dict["auc_mean_group_max"],
            ]
        ).amax()
        return metrics_dict
