"""Project-specific metrics for RSNA ICH classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from torchmetrics import Metric

from skp.configs import Config
from skp.metrics import utils


DEFAULT_CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


def _cat_state(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state
    return torch.cat(state, dim=0)


def _activate(logits: torch.Tensor, activation_fn: str | None) -> torch.Tensor:
    if activation_fn == "sigmoid":
        return logits.sigmoid()
    if activation_fn == "softmax":
        return logits.softmax(dim=1)
    return logits


class SliceAUROC(Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.class_names = cfg.get("label_columns") or DEFAULT_CLASS_NAMES
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].detach().float())
        self.t.append(batch["y"].detach().float())

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        p = _activate(p, self.cfg.get("metric_activation_fn"))

        metrics = {}
        for idx, name in enumerate(self.class_names):
            metrics[f"auc_{name}"] = utils.auc(t[:, idx], p[:, idx])
        metrics["auc_mean"] = torch.stack(list(metrics.values())).mean()
        return metrics


class SeriesAUROC(Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.class_names = cfg.get("label_columns") or DEFAULT_CLASS_NAMES
        self.aggregations = cfg.get("series_metric_aggregations") or [
            "max",
            "mean",
            "top3_mean",
        ]
        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("group_index", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].detach().float())
        self.t.append(batch["y"].detach().float())
        self.group_index.append(batch["group_index"].detach().long())

    @staticmethod
    def _aggregate_predictions(values: torch.Tensor, name: str) -> torch.Tensor:
        if name == "max":
            return values.amax(dim=0)
        if name == "mean":
            return values.mean(dim=0)
        if name == "top3_mean":
            k = min(3, values.shape[0])
            return values.topk(k, dim=0).values.mean(dim=0)
        raise ValueError(f"Unknown series aggregation: {name}")

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        group_index = utils.distributed_concat(
            _cat_state(self.group_index), dim=0
        ).cpu()
        p = _activate(p, self.cfg.get("metric_activation_fn"))

        p_by_group = defaultdict(list)
        t_by_group = defaultdict(list)
        for each_p, each_t, each_group in zip(p, t, group_index):
            key = int(each_group.item())
            p_by_group[key].append(each_p)
            t_by_group[key].append(each_t)

        metrics = {}
        for agg_name in self.aggregations:
            series_p = []
            series_t = []
            for key in sorted(p_by_group):
                group_p = torch.stack(p_by_group[key])
                group_t = torch.stack(t_by_group[key])
                series_p.append(self._aggregate_predictions(group_p, agg_name))
                series_t.append(group_t.amax(dim=0))
            series_p = torch.stack(series_p)
            series_t = torch.stack(series_t)

            agg_metrics = {}
            for idx, class_name in enumerate(self.class_names):
                agg_metrics[f"series_auc_{class_name}_{agg_name}"] = utils.auc(
                    series_t[:, idx], series_p[:, idx]
                )
            agg_metrics[f"series_auc_mean_{agg_name}"] = torch.stack(
                list(agg_metrics.values())
            ).mean()
            metrics.update(agg_metrics)
        return metrics
