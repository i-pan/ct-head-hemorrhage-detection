import torch

from einops import rearrange
from torchmetrics import Metric
from typing import Dict

from skp.configs.base import Config
from skp.metrics import utils


def _cat_state(state) -> torch.Tensor:
    """Return a metric state as one tensor before or after distributed sync."""
    if isinstance(state, torch.Tensor):
        return state
    return torch.cat(state, dim=0)


class BaseMetric(Metric):
    """
    Base class for simple classification/regression metrics
    """

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

    def compute(self):
        raise NotImplementedError


class ScoreBased(BaseMetric):
    """
    Uses model output scores (e.g., logits)
    For example, AUC and AVP
    Can also work with regression tasks, such as MAE
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0)  # (N,C)
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        if self.cfg.get("metric_include_classes"):
            assert isinstance(self.cfg.metric_include_classes, list)
            p = p[:, self.cfg.metric_include_classes]
            t = t[:, self.cfg.metric_include_classes]
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        # activation function is not necessarily required, for example for AUC
        if self.cfg.get("metric_activation_fn") == "sigmoid":
            p = p.sigmoid()
        elif self.cfg.get("metric_activation_fn") == "softmax":
            p = p.softmax(dim=1)
        p = p.cpu()
        if p.size(1) == 1:
            # Binary classification (or simple regression)
            return {f"{self.name}_mean": self.metric_func(t, p)}
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        return metrics_dict


class ClassBased(BaseMetric):
    """
    Uses class prediction
    For example, accuracy, Kappa score
    This is for multiclass/binary classification, not multilabel classification
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0)  # (N,C)
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        # activation function is not necessarily required
        if self.cfg.get("metric_activation_fn") == "sigmoid" or p.size(1) == 1:
            # p.size(1) = 1 implies binary classification
            p = p.sigmoid()
        elif self.cfg.get("metric_activation_fn") == "softmax":
            p = p.softmax(dim=1)
        p = p.cpu()
        if p.size(1) == 1:
            # binary classification
            threshold = self.cfg.get("binary_cls_threshold", 0.5) or 0.5
            p = (p >= threshold).float()
            return {f"{self.name}": self.metric_func(t, p)}
        p = torch.argmax(p, dim=1, keepdims=True)
        return {f"{self.name}": self.metric_func(t, p)}


class AUROC(ScoreBased):
    name = "auc"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.auc(t, p)


class AVP(ScoreBased):
    name = "avp"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.avp(t, p)


class MAE(ScoreBased):
    name = "mae"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.mae(t, p)


class MSE(ScoreBased):
    name = "mse"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.mse(t, p)


class Accuracy(ClassBased):
    name = "accuracy"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.accuracy(t, p)


class Kappa(ClassBased):
    name = "kappa"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.kappa(t, p)


class QWK(ClassBased):
    name = "qwk"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return utils.qwk(t, p)


class AUROCSeq(AUROC):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("mask", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p = out["logits_seq"] if "logits_seq" in out else out["logits"]
        t = batch["y_seq"] if "y_seq" in batch else batch["y"]
        self.p.append(rearrange(p.detach().float(), "b n c -> (b n) c"))
        self.t.append(rearrange(t.detach().float(), "b n c -> (b n) c"))
        if "mask" in batch:
            self.mask.append(rearrange(batch["mask"], "b n -> (b n)"))

    def compute(self) -> dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        if len(self.mask) > 0:
            mask = utils.distributed_concat(_cat_state(self.mask), dim=0).cpu()
            p, t = p[mask], t[mask]
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        return metrics_dict


class ManyClassAUROC(AUROC):
    # If large number of classes, instead of reporting each individual AUC,
    # report summary statistics
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        if self.cfg.get("metric_seq_mode", False):
            self.add_state("mask", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        if self.cfg.get("metric_seq_mode", False):
            self.p.append(
                rearrange(out["logits_seq"].detach().float(), "b n c -> (b n) c")
            )
            self.t.append(
                rearrange(batch["y_seq"].detach().float(), "b n c -> (b n) c")
            )
            if "mask" in batch:
                self.mask.append(rearrange(batch["mask"], "b n -> (b n)"))
        else:
            self.p.append(out["logits"].detach().float())
            self.t.append(batch["y"].detach().float())

    def compute(self) -> dict[str, torch.Tensor]:
        p = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        t = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        if hasattr(self, "mask"):
            mask = utils.distributed_concat(_cat_state(self.mask), dim=0).cpu()
            if len(mask) > 0:
                p, t = p[mask], t[mask]
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        auc_list = []
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            auc_list.append(self.metric_func(tmp_gt, p[:, c]))

        metrics_dict = {}
        metrics_dict["auc_mean"] = torch.stack(auc_list).mean(0)
        metrics_dict["auc_min"] = torch.stack(auc_list).amin(0)
        metrics_dict["auc_max"] = torch.stack(auc_list).amax(0)
        metrics_dict["auc_25p"] = torch.stack(auc_list).quantile(0.25, dim=0)
        metrics_dict["auc_50p"] = torch.stack(auc_list).quantile(0.50, dim=0)
        metrics_dict["auc_75p"] = torch.stack(auc_list).quantile(0.75, dim=0)
        metrics_dict["auc_min_class"] = torch.stack(auc_list).argmin(0)
        metrics_dict["auc_max_class"] = torch.stack(auc_list).argmax(0)
        return metrics_dict
