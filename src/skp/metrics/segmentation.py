import torch
import torchmetrics as tm

from einops import rearrange, reduce
from torch.nn.functional import one_hot
from typing import Dict

from skp.configs import Config
from skp.metrics import utils


def _cat_state(state) -> torch.Tensor:
    """Return a metric state as one tensor before or after distributed sync."""
    if isinstance(state, torch.Tensor):
        return state
    return torch.cat(state, dim=0)


def _mean_ignore_nan(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    valid = ~torch.isnan(x)
    if not valid.any():
        return torch.tensor(float("nan"), device=x.device)
    return x[valid].mean()


def _dice_from_binary_masks(
    p: torch.Tensor,
    t: torch.Tensor,
    reduce_pattern: str,
) -> torch.Tensor:
    intersection = reduce(p * t, reduce_pattern, "sum")
    denominator = reduce(p + t, reduce_pattern, "sum")
    return (2 * intersection) / denominator


class MulticlassDiceScore(tm.Metric):
    """Dice score for mutually exclusive segmentation classes."""

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"].detach(), batch["y"].detach()
        if "mask_present" in batch:
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]

        assert self.cfg.get("metric_activation_fn") == "softmax" or self.cfg.get(
            "activation_fn"
        ) == "softmax"
        num_classes = p.size(1)
        if t.ndim == p.ndim and t.size(1) == 1:
            t = t[:, 0]
        p = one_hot(p.argmax(1), num_classes=num_classes)
        t = one_hot(t.long(), num_classes=num_classes)

        if self.cfg.get("metric_invert_background", False):
            p[..., 0] = 1 - p[..., 0]
            t[..., 0] = 1 - t[..., 0]

        if self.cfg.get("metric_ignore_class0", False):
            p, t = p[..., 1:], t[..., 1:]

        if p.ndim == 5:
            p = rearrange(p, "b x y z c -> b c x y z").long()
            t = rearrange(t, "b x y z c -> b c x y z").long()
            reduce_pattern = "b c x y z -> b c"
        elif p.ndim == 4:
            p = rearrange(p, "b h w c -> b c h w").long()
            t = rearrange(t, "b h w c -> b c h w").long()
            reduce_pattern = "b c h w -> b c"
        else:
            raise ValueError(f"Expected 4D or 5D one-hot masks, got {p.shape}")

        self.dice_scores.append(_dice_from_binary_masks(p, t, reduce_pattern))

    def compute(self) -> Dict[str, torch.Tensor]:
        dice_scores = utils.distributed_concat(_cat_state(self.dice_scores), dim=0)
        metrics_dict = {}
        offset = 1 if self.cfg.get("metric_ignore_class0", False) else 0
        for idx in range(dice_scores.shape[1]):
            metrics_dict[f"dice{idx + offset}"] = _mean_ignore_nan(
                dice_scores[:, idx]
            )
        metrics_dict["dice_mean"] = torch.stack(list(metrics_dict.values())).mean(0)
        return metrics_dict


class MultilabelDiceScore(tm.Metric):
    """Threshold-swept Dice score for multilabel segmentation."""

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.cfg = cfg
        self.thresholds = torch.tensor(cfg.get("metric_thresholds") or [0.5])
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"].detach(), batch["y"].detach()
        if "mask_present" in batch:
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]

        if self.cfg.get("metric_labels_to_onehot", False):
            if t.ndim == p.ndim and t.size(1) == 1:
                t = t[:, 0]
            t = one_hot(t.long(), num_classes=p.size(1))
            if p.ndim == 5:
                t = rearrange(t, "b x y z c -> b c x y z")
            elif p.ndim == 4:
                t = rearrange(t, "b h w c -> b c h w")

        assert self.cfg.get("metric_activation_fn") == "sigmoid" or self.cfg.get(
            "activation_fn"
        ) == "sigmoid"
        p = p.sigmoid()

        if p.size(1) == 1 and p.ndim == t.ndim + 1:
            t = t.unsqueeze(1)

        assert p.ndim == t.ndim, (
            f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions"
        )

        if self.cfg.get("loss_params", {}).get("invert_background", False):
            t = t.clone()
            t[:, 0] = 1 - t[:, 0]

        thresholds = self.thresholds.to(p.device)
        p = torch.stack([p >= threshold for threshold in thresholds])
        t = torch.stack([t] * len(thresholds))
        if p.ndim == 6:
            reduce_pattern = "n b c x y z -> n b c"
        elif p.ndim == 5:
            reduce_pattern = "n b c h w -> n b c"
        else:
            raise ValueError(f"Expected thresholded 5D or 6D masks, got {p.shape}")

        self.dice_scores.append(_dice_from_binary_masks(p, t, reduce_pattern))

    def compute(self) -> Dict[str, torch.Tensor]:
        dice = torch.cat(self.dice_scores, dim=1)
        dice = utils.distributed_concat(dice, dim=1)
        dice_over_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        _mean_ignore_nan(dice[threshold_idx, :, class_idx])
                        for threshold_idx in range(dice.shape[0])
                    ]
                )
                for class_idx in range(dice.shape[2])
            ]
        )

        best_dice = dice_over_thresholds.amax(dim=1)
        best_thresholds = self.thresholds[dice_over_thresholds.argmax(dim=1).cpu()]
        metrics_dict = {f"dice{idx}": value for idx, value in enumerate(best_dice)}
        metrics_dict["dice_mean"] = torch.stack(list(metrics_dict.values())).mean(0)
        metrics_dict.update(
            {f"th{idx}": threshold for idx, threshold in enumerate(best_thresholds)}
        )
        return metrics_dict
