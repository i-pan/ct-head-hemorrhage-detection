"""
Commonly used losses for segmentation tasks.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from typing import Dict, List, Optional, Tuple, Union

from skp.losses import _check_shapes_equal


def _weight_func(ground_truth: torch.Tensor, weight_type: str = "square"):
    if not torch.is_floating_point(ground_truth):
        ground_truth = ground_truth.float()
    if weight_type == "square":
        return torch.reciprocal(ground_truth**2)
    elif weight_type == "simple":
        return torch.reciprocal(ground_truth)
    else:
        raise Exception(f"Invalid weight_type: {weight_type}")


def _dice_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    pred_power: float = 1.0,
    smooth: float = 1e-5,
    activation_fn: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    compute_method: str = "per_sample",
    generalized: bool = False,
    weight_type: str = "square",  # only if generalized=True
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert compute_method in {
        "per_sample",
        "per_batch",
    }, f"Invalid compute_method: {compute_method}"

    assert x.shape == y.shape, f"x.shape [{x.shape}] does not equal y.shape [{y.shape}]"
    if not torch.is_floating_point(y):
        y = y.float()

    # y should be one-hot encoded
    if activation_fn == "sigmoid":
        x = x.sigmoid()
    elif activation_fn == "softmax":
        x = x.softmax(dim=1)
    elif activation_fn is None:
        pass
    else:
        raise Exception(f"Invalid activation_fn: {activation_fn}")

    if compute_method == "per_sample":
        # compute dice score per sample, then average
        if x.ndim == 5:  # 3D
            s = "b c x y z -> b c"
        elif x.ndim == 4:  # 2D
            s = "b c h w -> b c"
        else:
            raise ValueError(f"Expected 4D or 5D prediction tensor, got {x.ndim}D")
    elif compute_method == "per_batch":
        # aggregate all pixels in batch, compute dice
        if x.ndim == 5:
            s = "b c x y z -> 1 c"
        elif x.ndim == 4:
            s = "b c h w -> 1 c"
        else:
            raise ValueError(f"Expected 4D or 5D prediction tensor, got {x.ndim}D")

    flat_sample_weight = None
    if sample_weight is not None:
        flat_sample_weight = sample_weight.to(x.device).float().flatten()
        view_shape = [flat_sample_weight.shape[0]] + [1] * (x.ndim - 1)
        broadcast_sample_weight = flat_sample_weight.view(*view_shape)
        weighted_intersection = x * y * broadcast_sample_weight
        weighted_prediction = x.pow(pred_power) * broadcast_sample_weight
        weighted_target = y * broadcast_sample_weight
    else:
        weighted_intersection = x * y
        weighted_prediction = x.pow(pred_power)
        weighted_target = y

    intersection = reduce(weighted_intersection, s, "sum")

    x = reduce(weighted_prediction, s, "sum")
    # y is 0 or 1 so raising to pred_power does nothing
    y = reduce(weighted_target, s, "sum")

    denominator = x + y

    if generalized:
        # From: https://docs.monai.io/en/stable/losses.html#generalizeddiceloss
        w = _weight_func(y, weight_type)
        infs = torch.isinf(w)
        if compute_method == "per_batch":
            w[infs] = 0.0
            w = w + infs * torch.max(w)
        elif compute_method == "per_sample":
            w[infs] = 0.0
            max_values = torch.max(w, dim=1)[0].unsqueeze(dim=1)
            w = w + infs * max_values
        intersection *= w
        denominator *= w

    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - dice_score

    if class_weights is not None:
        class_weights = class_weights.to(dice_loss.device)
        dice_loss = dice_loss * class_weights
        dice_loss = reduce(dice_loss, "b c -> b", "sum")
        dice_loss = dice_loss / class_weights.sum()

    if (
        flat_sample_weight is not None
        and compute_method == "per_sample"
        and dice_loss.shape[0] == flat_sample_weight.shape[0]
    ):
        while flat_sample_weight.ndim < dice_loss.ndim:
            flat_sample_weight = flat_sample_weight.unsqueeze(-1)
        return (
            dice_loss * flat_sample_weight
        ).sum() / flat_sample_weight.sum().clamp_min(1e-6)

    return dice_loss.mean()


def _tversky_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    smooth: float = 1e-5,
    activation_fn: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    compute_method: str = "per_sample",
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    assert compute_method in {
        "per_sample",
        "per_batch",
    }, f"Invalid compute_method: {compute_method}"
    assert x.shape == y.shape, f"x.shape [{x.shape}] does not equal y.shape [{y.shape}]"
    if not torch.is_floating_point(y):
        y = y.float()

    # y should be one-hot encoded
    if activation_fn == "sigmoid":
        x = x.sigmoid()
    elif activation_fn == "softmax":
        x = x.softmax(dim=1)
    elif activation_fn is None:
        pass
    else:
        raise Exception(f"Invalid activation_fn: {activation_fn}")

    p0 = x
    p1 = 1 - p0
    g0 = y
    g1 = 1 - g0

    if compute_method == "per_sample":
        # compute dice score per sample, then average
        if x.ndim == 5:  # 3D
            s = "b c x y z -> b c"
        elif x.ndim == 4:  # 2D
            s = "b c h w -> b c"
        else:
            raise ValueError(f"Expected 4D or 5D prediction tensor, got {x.ndim}D")
    elif compute_method == "per_batch":
        # aggregate all pixels in batch, compute dice
        if x.ndim == 5:
            s = "b c x y z -> 1 c"
        elif x.ndim == 4:
            s = "b c h w -> 1 c"
        else:
            raise ValueError(f"Expected 4D or 5D prediction tensor, got {x.ndim}D")

    tp = reduce(p0 * g0, s, "sum")
    fp = alpha * reduce(p0 * g1, s, "sum")
    fn = beta * reduce(p1 * g0, s, "sum")

    numerator = tp + smooth
    denominator = tp + fp + fn + smooth

    score = 1.0 - numerator / denominator

    if class_weights is not None:
        class_weights = class_weights.to(score.device)
        score = score * class_weights
        score = reduce(score, "b c -> b", "sum")
        score = score / class_weights.sum()

    return score.mean()


def _sigmoid_focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    From:
    https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/focal_loss.py

    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # see also https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp#L363
    loss: torch.Tensor = input - input * target - F.logsigmoid(input)

    # sigmoid(-i) if t==1; sigmoid(i) if t==0 <=>
    # 1-sigmoid(i) if t==1; sigmoid(i) if t==0 <=>
    # 1-p if t==1; p if t==0 <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss


class _SegmentationLoss(nn.Module):
    @staticmethod
    def get_inputs(out: Dict, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]
        return p, t

    def format_labels(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.ignore_background and self.invert_background:
            raise Exception(
                "ignore_background and invert_background cannot both be True"
            )
        if self.invert_background:
            if hasattr(self, "activation_fn") and self.activation_fn == "softmax":
                raise Exception(
                    "invert_background and softmax activation should not be used together"
                )
        t = t.clone()
        if self.convert_labels_to_onehot:
            num_classes = p.size(1)
            if self.ignore_background:
                num_classes += 1
            t = F.one_hot(t.long(), num_classes=num_classes).float()
            if p.ndim == 5:
                t = rearrange(t, "b x y z c -> b c x y z")
            elif p.ndim == 4:
                t = rearrange(t, "b h w c -> b c h w")
        # if only one class
        if p.size(1) == 1 and p.ndim == t.ndim + 1:
            t = t.unsqueeze(1)  # add channel dim
        if self.invert_background:
            t[:, 0] = 1 - t[:, 0]
        if self.ignore_background:
            t = t[:, 1:]
        if self.resize_ground_truth is not None:
            t = F.interpolate(
                t.float(), size=self.resize_ground_truth, mode="nearest"
            ).long()
        return t


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        if "pos_weight" in params:
            params["pos_weight"] = torch.tensor(params["pos_weight"])
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        return {"loss": super().forward(p.float(), t.float())}


class DiceLoss(_SegmentationLoss):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.convert_labels_to_onehot = params.pop("convert_labels_to_onehot", False)
        self.resize_ground_truth = params.pop("resize_ground_truth", None)
        self.invert_background = params.pop("invert_background", False)
        self.ignore_background = params.pop("ignore_background", False)
        class_weights = params.pop("class_weights", None)
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights).float(),
                persistent=False,
            )
        else:
            self.class_weights = None
        params["class_weights"] = self.class_weights
        self.activation_fn = params.get("activation_fn")
        if params.get("activation_fn", None) is None:
            print(
                "WARN: No activation function provided for Dice loss.",
                "It is recommended to provide an activation function for improved training.",
            )
        self.loss_args = params

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        loss_dict = {"loss": _dice_loss(p, t, **self.loss_args)}
        return loss_dict


class DiceBCELoss(DiceLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.dice_weight = params.pop("dice_weight", 1.0)
        self.bce_weight = params.pop("bce_weight", 1.0)
        self.alpha = params.pop("alpha", None)
        super().__init__(params)
        assert self.alpha is None or 0 < self.alpha < 1

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        dice_loss = _dice_loss(p, t, **self.loss_args)
        bce_loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        if self.alpha is not None:
            # alpha if t==1; (1-alpha) if t==0
            alpha_factor = t * self.alpha + (1 - t) * (1 - self.alpha)
            bce_loss = alpha_factor * bce_loss
        n, c = bce_loss.shape[:2]
        bce_loss = bce_loss.reshape(n, c, -1).mean(dim=2)
        if self.loss_args.get("class_weights", None) is not None:
            class_weights = self.loss_args["class_weights"].to(bce_loss.device)
            bce_loss = (bce_loss * class_weights).sum(1) / class_weights.sum()
        bce_loss = bce_loss.mean()
        loss_dict = {"dice_loss": dice_loss, "bce_loss": bce_loss}
        loss_dict["loss"] = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return loss_dict


class DiceFocalLoss(DiceLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.dice_weight = params.pop("dice_weight", 1.0)
        self.focal_weight = params.pop("focal_weight", 1.0)
        self.gamma = params.pop("gamma", 2.0)
        self.alpha = params.pop("alpha", None)
        self.scale = params.pop("scale", 1.0)
        super().__init__(params)
        assert self.alpha is None or 0 < self.alpha < 1

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        sample_weight = batch.get("sample_weight")
        dice_loss = _dice_loss(p, t, sample_weight=sample_weight, **self.loss_args)
        focal_loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        n, c = focal_loss.shape[:2]
        focal_loss = focal_loss.reshape(n, c, -1).mean(dim=2)
        if self.loss_args.get("class_weights", None) is not None:
            class_weights = self.loss_args["class_weights"].to(focal_loss.device)
            focal_loss = (focal_loss * class_weights).sum(1) / class_weights.sum()
        if sample_weight is not None:
            sample_weight = sample_weight.to(focal_loss.device).float().flatten()
            while sample_weight.ndim < focal_loss.ndim:
                sample_weight = sample_weight.unsqueeze(-1)
            focal_loss = (
                focal_loss * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            focal_loss = focal_loss.mean()
        focal_loss = focal_loss * self.scale
        loss_dict = {"dice_loss": dice_loss, "focal_loss": focal_loss}
        loss_dict["loss"] = (
            self.dice_weight * dice_loss + self.focal_weight * focal_loss
        )
        return loss_dict


class PositiveDiceNegativeFocalLoss(DiceLoss):
    """
    Hybrid loss for heavily imbalanced segmentation with many empty masks.

    Non-empty samples receive Dice + focal loss. Empty samples receive only focal
    loss, usually with a small weight, so they discourage false positives without
    dominating the segmentation objective. Set gamma=0 and alpha=None to recover
    BCE-with-logits behavior for the focal term.
    """

    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.dice_weight = params.pop("dice_weight", 1.0)
        self.focal_weight = params.pop("focal_weight", 1.0)
        self.positive_focal_weight = params.pop("positive_focal_weight", 1.0)
        self.negative_focal_weight = params.pop("negative_focal_weight", 0.05)
        self.gamma = params.pop("gamma", 2.0)
        self.alpha = params.pop("alpha", None)
        self.scale = params.pop("scale", 1.0)
        self.positive_key = params.pop("positive_key", None)
        self.target_positive_threshold = params.pop("target_positive_threshold", 0.0)
        super().__init__(params)
        assert self.alpha is None or 0 < self.alpha < 1
        if self.loss_args.get("compute_method") == "per_batch":
            print(
                "WARN: PositiveDiceNegativeFocalLoss is usually intended for",
                "compute_method='per_sample'.",
            )

    def _positive_mask(
        self, t: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.positive_key is not None and self.positive_key in batch:
            return batch[self.positive_key].to(t.device).bool().flatten()
        return t.flatten(1).sum(dim=1) > self.target_positive_threshold

    def _focal_per_sample(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        focal_loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        n, c = focal_loss.shape[:2]
        focal_loss = focal_loss.reshape(n, c, -1).mean(dim=2)
        if self.loss_args.get("class_weights", None) is not None:
            class_weights = self.loss_args["class_weights"].to(focal_loss.device)
            focal_loss = (focal_loss * class_weights).sum(1) / class_weights.sum()
        else:
            focal_loss = focal_loss.mean(dim=1)
        return focal_loss

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        positive_mask = self._positive_mask(t, batch)

        if positive_mask.any():
            dice_loss = _dice_loss(p[positive_mask], t[positive_mask], **self.loss_args)
        else:
            dice_loss = p.sum() * 0.0

        focal_per_sample = self._focal_per_sample(p, t)
        sample_weight = torch.where(
            positive_mask.to(focal_per_sample.device),
            torch.full_like(focal_per_sample, self.positive_focal_weight),
            torch.full_like(focal_per_sample, self.negative_focal_weight),
        )
        if "sample_weight" in batch:
            sample_weight = (
                sample_weight
                * batch["sample_weight"].to(focal_per_sample.device).float().flatten()
            )
        focal_loss = (
            focal_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1e-6)
        focal_loss = focal_loss * self.scale

        positive_count = positive_mask.sum().to(p.dtype)
        negative_count = positive_mask.numel() - positive_count
        positive_focal_loss = (
            focal_per_sample[positive_mask].mean()
            if positive_mask.any()
            else p.sum() * 0.0
        )
        negative_focal_loss = (
            focal_per_sample[~positive_mask].mean()
            if (~positive_mask).any()
            else p.sum() * 0.0
        )

        loss_dict = {
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "positive_focal_loss": positive_focal_loss,
            "negative_focal_loss": negative_focal_loss,
            "positive_fraction": positive_count / max(positive_mask.numel(), 1),
            "negative_count": negative_count,
        }
        loss_dict["loss"] = (
            self.dice_weight * dice_loss + self.focal_weight * focal_loss
        )
        return loss_dict


class TverskyLoss(DiceLoss):
    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        loss_dict = {"loss": _tversky_loss(p, t, **self.loss_args)}
        return loss_dict


class FocalLoss(_SegmentationLoss):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.gamma = params.get("gamma", 2.0)
        self.alpha = params.get("alpha", None)
        self.convert_labels_to_onehot = params.get("convert_labels_to_onehot", False)
        self.resize_ground_truth = params.get("resize_ground_truth", None)
        self.invert_background = params.get("invert_background", False)
        self.ignore_background = params.get("ignore_background", False)
        self.activation_fn = params.get("activation_fn")
        self.batch_size = params.get("batch_size", None)
        self.scale = params.get("scale", 1.0)
        assert not (self.invert_background and self.ignore_background)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        _check_shapes_equal(p, t)
        loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        if "sample_weight" in batch:
            loss = loss.flatten(1) * batch["sample_weight"].unsqueeze(1)
        return {"loss": self.scale * loss.mean()}


class FocalLossMemoryEfficient(FocalLoss):
    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = t.unsqueeze(1)
        # t is NOT one-hot encoded
        # t contains class indices where 0 is background
        num_classes = p.shape[1]
        if self.ignore_background:
            # ignore 0 class, need num_classes + 1 to include last class index
            target_classes = torch.arange(1, num_classes + 1, device=p.device)
        else:
            target_classes = torch.arange(num_classes, device=p.device)
        if p.ndim == 5:
            target_classes = target_classes.view(1, num_classes, 1, 1, 1)
        else:
            target_classes = target_classes.view(1, num_classes, 1, 1)
        # if self.batch_size is not None and p.shape[0] > self.batch_size:
        #     loss = []
        #     for i in range(0, p.shape[0], self.batch_size):
        #         loss.append(
        #             _sigmoid_focal_loss(
        #                 p[i : i + self.batch_size],
        #                 (t[i : i + self.batch_size] == target_classes).float(),
        #                 gamma=self.gamma,
        #                 alpha=self.alpha,
        #             ).mean()
        #         )
        #     loss = torch.stack(loss).mean()
        # else:
        t = (t == target_classes).float()
        if self.invert_background:
            # assume 1st channel is background
            t[:, 0] = 1 - t[:, 0]
        loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        loss = loss.mean()
        # loss = 0
        # for i in range(p.shape[1]):
        #     loss += _sigmoid_focal_loss(
        #         p[:, i], (t == i + 1).float(), gamma=self.gamma, alpha=self.alpha
        #     ).mean()
        # loss = loss / p.shape[1]
        return {"loss": self.scale * loss}


class DeepSupervision(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.deep_supervision_weights = params.pop("deep_supervision_weights")
        loss_name = params.pop("loss_name")
        self.loss = globals()[loss_name](params)

    @staticmethod
    def downsample_ground_truth(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            F.interpolate(
                y.unsqueeze(1).float() if y.ndim == logits.ndim - 1 else y.float(),
                size=logits.shape[2:],
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        # full resolution loss
        loss_dict["loss0"] = self.loss(out, batch)["loss"]
        # downsampled maps from earlier layers
        # auxiliary losses

        if out["aux_logits"] is not None:  # if None, then using SWI during val
            for idx, logits in enumerate(out["aux_logits"]):
                downsampled_y = self.downsample_ground_truth(logits, batch["y"])
                tmp_batch = {"y": downsampled_y}
                if "mask_present" in batch:
                    tmp_batch["mask_present"] = batch["mask_present"]
                loss_dict[f"loss{idx + 1}"] = self.loss({"logits": logits}, tmp_batch)[
                    "loss"
                ]

            loss_dict["loss"] = torch.stack(
                [
                    v * self.deep_supervision_weights[int(i.replace("loss", ""))]
                    for i, v in loss_dict.items()
                ]
            ).sum(0)
        else:
            loss_dict["loss"] = loss_dict["loss0"]

        return loss_dict


class DeepSupervisionDiceBCELoss(nn.Module):
    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing configuration parameters:
                - 'weight_dice' (float, default=1.0): Weight for Dice component.
                - 'weight_bce' (float, default=1.0): Weight for BCE component.
                - 'epsilon' (float, default=1e-6): Stability epsilon.
                - 'squared_pred' (bool, default=False): If True, use squares in Dice denominator.
                - 'to_onehot_y' (bool, default=False): If True, convert targets to one-hot.
                - 'ignore_background_channel' (bool, default=False): If True (and to_onehot_y is True),
                   assumes 0 is background, removes it, and expects preds to have C-1 channels.
                - 'class_weights' (torch.Tensor, optional): Tensor of shape (C,) for class balancing.
                - 'bce_pos_weight' (float, default=1.0): Weight for positive examples in BCE.
                - 'batch_dice' (bool, default=False): If True, calculate Dice over the whole batch (pseudovolume).
                   If False, calculate per sample and average.
                - 'ds_weights' (list, optional): List of weights for deep supervision levels.
                   Order: [original, ds1, ds2, ...].
        """
        super().__init__()
        self.w_dice = params.get("weight_dice", 1.0)
        self.w_bce = params.get("weight_bce", 1.0)
        self.eps = params.get("epsilon", 1e-6)
        self.squared_pred = params.get("squared_pred", False)
        self.to_onehot_y = params.get("to_onehot_y", False)
        self.ignore_background = params.get("ignore_background_channel", False)
        self.batch_dice = params.get("batch_dice", False)
        self.ds_weights = params.get("ds_weights", None)

        # Store class weights and BCE pos weight as buffer so they move to GPU with the module
        class_weights = params.get("class_weights", None)
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights).float(),
                persistent=False,
            )
        else:
            self.class_weights = None

        bce_pos_weight = params.get("bce_pos_weight", 1.0)
        self.register_buffer(
            "bce_pos_weight", torch.tensor(bce_pos_weight), persistent=False
        )

    def forward(self, out, batch, mode="train"):
        """
        Args:
            out (dict): Must contain 'logits'. Optional keys: 'logits_ds1', 'logits_ds2', etc.
            batch (dict): Must contain 'y' (ground truth).
        """
        target_orig = batch["y"]

        # Identify available Deep Supervision keys in order: logits, logits_ds1, logits_ds2...
        ds_keys = ["logits"] + sorted(
            [k for k in out.keys() if k.startswith("logits_ds")],
            key=lambda x: (
                int(x.replace("logits_ds", ""))
                if x.replace("logits_ds", "").isdigit()
                else 999
            ),
        )

        # Validate weights (for training)
        if self.ds_weights is not None and mode == "train":
            assert len(self.ds_weights) == len(ds_keys), (
                f"Number of ds_weights ({len(self.ds_weights)}) must match number of outputs ({len(ds_keys)})."
            )
            weights = self.ds_weights
        else:
            # Default to 1.0 for main, 0.0 for others if not specified (or 1.0 for all, usually strictly specified)
            # Here we assume equal weighting or just main if not provided.
            weights = [1.0] * len(ds_keys)

        total_loss = {"loss": 0.0}
        if self.w_dice > 0:
            total_loss["dice_loss"] = 0.0
        if self.w_bce > 0:
            total_loss["bce_loss"] = 0.0

        # Convert to one-hot first
        if self.to_onehot_y:
            target_orig = self._convert_to_onehot(target_orig, out["logits"])

        for i, key in enumerate(ds_keys):
            pred_logits = out[key]
            current_weight = weights[i]

            if current_weight == 0:
                continue

            # Determine spatial target shape from current prediction
            target_shape = pred_logits.shape[2:]  # (D, H, W) or (H, W)

            # Downsample target (Average Pooling / Interpolation area)
            # This creates continuous "soft" targets from binary masks
            if target_orig.shape[2:] != target_shape:
                # Determine mode based on dimensions
                mode = "trilinear" if len(target_shape) == 3 else "area"
                align_corners = False if mode == "trilinear" else None

                # Convert to float for interpolation if not already
                target_float = target_orig.float()

                # F.interpolate with area/trilinear behaves like Adaptive Avg Pool for downsizing
                interpolate_kwargs = {
                    "input": target_float,
                    "size": target_shape,
                    "mode": mode,
                }
                if align_corners is not None:
                    interpolate_kwargs["align_corners"] = align_corners
                target_resampled = F.interpolate(**interpolate_kwargs)
            else:
                target_resampled = target_orig

            # Calculate loss for this level
            single_scale_loss_dict = self._compute_single_scale_loss(
                pred_logits, target_resampled
            )
            for k, v in single_scale_loss_dict.items():
                total_loss[k] += current_weight * v

        # Scale by sum of weights
        for k, v in total_loss.items():
            total_loss[k] = v / sum(weights)

        return total_loss

    def _convert_to_onehot(self, target, pred):
        """
        Converts indices to one-hot.
        Handles the logic for ignoring the background channel.
        """
        # Assume target is (B, 1, ...) or (B, ...) containing indices
        if target.ndim == pred.ndim:
            target = target.squeeze(1)

        # Determine number of classes for one-hot
        if self.ignore_background:
            # If we ignore background, the pred has C channels (classes 1..C).
            # The target has indices 0..C. We map 0->Background, 1..C->Classes.
            # So we need pred.shape[1] + 1 classes for the encoding.
            n_classes = pred.shape[1] + 1
        else:
            n_classes = pred.shape[1]

        # One hot: (B, spatial...) -> (B, spatial..., C)
        target_onehot = F.one_hot(target.long(), num_classes=n_classes)

        # Permute to (B, C, spatial...)
        dims = (0, -1) + tuple(range(1, target_onehot.ndim - 1))
        target_onehot = target_onehot.permute(dims).float()

        # If ignore background, strip the first channel (index 0)
        if self.ignore_background:
            target_onehot = target_onehot[:, 1:, ...]

        return target_onehot

    def _compute_single_scale_loss(self, logits, target):
        loss_dict = {"loss": 0.0}

        # Check shapes
        if logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch after processing: Pred {logits.shape} vs Target {target.shape}. "
                "Check 'to_onehot_y' and 'ignore_background_channel' params."
            )

        # Dice Loss
        if self.w_dice > 0:
            probs = torch.sigmoid(logits)
            dice = self._get_dice(probs, target)
            loss_dict["dice_loss"] = dice
            loss_dict["loss"] += self.w_dice * dice

        # BCE Loss
        if self.w_bce > 0:
            bce = self._get_bce(logits, target)
            loss_dict["bce_loss"] = bce
            loss_dict["loss"] += self.w_bce * bce

        return loss_dict

    def _get_dice(self, probs, target):
        """
        Calculates Soft Dice Loss.
        """
        # Dimensions to sum over
        # If batch_dice is True: sum over (B, spatial...) -> ONE score per class
        # If batch_dice is False: sum over (spatial...) -> (B) scores per class, then average B

        ndim = probs.ndim
        if self.batch_dice:
            # Sum over Batch (0) and Spatial (2, ...)
            dims = (0,) + tuple(range(2, ndim))
        else:
            # Sum over Spatial only
            dims = tuple(range(2, ndim))

        # Numerator
        intersection = (probs * target).sum(dim=dims)

        # Denominator
        if self.squared_pred:
            cardinality = (probs.pow(2) + target.pow(2)).sum(dim=dims)
        else:
            cardinality = (probs + target).sum(dim=dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        dice_loss = 1.0 - dice_score

        # Apply Class Weights if provided
        if self.class_weights is not None:
            # dice_loss shape: (C,) if batch_dice else (B, C)
            # class_weights shape: (C,)
            class_weights = self.class_weights.to(dice_loss.device)
            if not self.batch_dice:
                # broadcast weights to batch
                weights = class_weights.view(1, -1)
                dice_loss = dice_loss * weights
                return (dice_loss.sum(dim=1) / class_weights.sum()).mean()
            else:
                dice_loss = dice_loss * class_weights
                return dice_loss.sum() / class_weights.sum()
        else:
            return dice_loss.mean()

    def _get_bce(self, logits, target):
        """
        Calculates Binary Cross Entropy with Logits.
        """
        # If class weights are present, we use the weight arg in BCE
        # BCE weights usually expected as (C) or (B, C, ...)

        # Note: standard BCE weights are usually for positive/negative balance.
        # Here we are implementing Class Balancing.
        # F.binary_cross_entropy_with_logits supports 'weight' (per element) and 'pos_weight' (class imbalance).
        # We will manually multiply by class weights to ensure consistent behavior with Dice logic.

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=self.bce_pos_weight
        )

        # Average over spatial dims to get (B, C)
        # We assume independent pixels, usually mean over spatial is taken.
        ndim = logits.ndim
        spatial_dims = tuple(range(2, ndim))
        bce_loss = bce_loss.mean(dim=spatial_dims)  # Now (B, C)

        if self.class_weights is not None:
            weights = self.class_weights.to(bce_loss.device).view(1, -1)
            bce_loss = bce_loss * weights
            return (bce_loss.sum(dim=1) / weights.sum()).mean()

        return bce_loss.mean()


class DeepSupervisionDiceFocalLoss(DeepSupervisionDiceBCELoss):
    """
    Uses the same arguments, but instead of BCE will calculate focal loss.
    Still uses class method _get_bce for easy class inheritance.
    Still uses bce_weight parameter for focal loss component.
    """

    def __init__(self, params):
        super().__init__(params)
        self.gamma = params.get("gamma", 2.0)
        self.alpha = params.get("alpha", 0.25)

    def _get_bce(self, logits, target):
        """
        Calculates Alpha-Balanced Focal Loss.

        Args:
            logits: (B, C, D, H, W) raw output scores.
            target: (B, C, D, H, W) binary targets.
            gamma: Focusing parameter (default 2.0). Reduces loss for easy examples.
            alpha: Balancing parameter (default 0.25).
                   Weight assigned to the positive class (1).
                   Weight assigned to negative class (0) is (1-alpha).
                   Set to None to disable alpha weighting.
        """

        # 1. Calculate raw BCE (No reduction, No pos_weight)
        # We need pixel-wise loss to apply pixel-wise alpha/focal weights.
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # 2. Calculate pt (Probability of the ground truth class)
        # Since BCE = -log(pt), we can recover pt via exp(-BCE)
        pt = torch.exp(-bce_loss)

        # 3. Calculate Focal Component: (1 - pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        # 4. Calculate Alpha Component
        # Alpha applies to Foreground (Target=1), (1-Alpha) applies to Background (Target=0)
        if self.alpha is not None:
            # Broadcast alpha to shape of targets
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # 5. Average over spatial dims to get (B, C)
        # This keeps the batch and channel dimensions separate for now
        ndim = logits.ndim
        spatial_dims = tuple(range(2, ndim))
        loss = loss.mean(dim=spatial_dims)

        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device).view(1, -1)
            loss = loss * weights
            return (loss.sum(dim=1) / weights.sum()).mean()

        # 6. Final Mean (Average over Batch and Classes)
        return loss.mean()
