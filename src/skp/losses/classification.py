"""
This module contains PyTorch losses to be used for classification (and regression).

The loss is returned as a dictionary. The primary loss must be returned as `loss`.
Other losses must be returned with the suffix `_loss`.
This is useful when the loss comprises several other losses, so all can be tracked.

Losses take as argument the input and output dictionaries. This makes things more
flexible if you have more than the standard logits and labels.
"""

import copy
import torch

from torch import nn
from torch.nn import functional as F
from typing import Dict, Optional

from skp.losses import _check_shapes_equal


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


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.seq_mode = params.pop("seq_mode", False)
        if "pos_weight" in params:
            params["pos_weight"] = torch.tensor(params["pos_weight"])
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        if self.seq_mode:
            p, t = out["logits_seq"], batch["y_seq"]
            mask = batch["mask"]
            p, t = p[mask], t[mask]
        else:
            p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        return {"loss": super().forward(p.float(), t.float())}


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, params: Dict):
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if t.ndim == 2 and t.size(1) == 1:
            t = t[:, 0]
        return {"loss": super().forward(p.float(), t.long())}


"""
LabelSmoothingCrossEntropy and SoftTargetCrossEntropy
From: https://github.com/huggingface/pytorch-image-models/blob/main/timm/loss/cross_entropy.py 
"""


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, params: Dict):
        super().__init__()
        self.smoothing = params.get("smoothing", 0.1)
        assert self.smoothing < 1.0
        self.confidence = 1.0 - self.smoothing

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        logprobs = F.log_softmax(p, dim=-1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        assert p.ndim == t.ndim, (
            f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be equal"
        )
        nll_loss = -logprobs.gather(dim=-1, index=t)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return {"loss": loss.mean()}


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, params: Dict):
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        if not torch.is_floating_point(t):
            t = t.float()
        loss = torch.sum(-t * F.log_softmax(p, dim=-1), dim=-1)
        return {"loss": loss.mean()}


class FocalLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.seq_mode = params.pop("seq_mode", False)
        self.gamma = params.get("gamma", 2.0)
        self.alpha = params.get("alpha", None)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        if self.seq_mode:
            p, t = out["logits_seq"], batch["y_seq"]
            mask = batch["mask"]
            p, t = p[mask], t[mask]
        else:
            p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        return {"loss": loss.mean()}


class L1Loss(nn.L1Loss):
    def __init__(self, params: Dict):
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        return {"loss": super().forward(p.float(), t.float())}


class SmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, params: Dict):
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        return {"loss": super().forward(p.float(), t.float())}


class L2Loss(nn.MSELoss):
    def __init__(self, params: Dict):
        super().__init__(**params)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        return {"loss": super().forward(p.float(), t.float())}


class WeightedL2Loss(nn.MSELoss):
    def __init__(self, params: Dict):
        assert "weights" in params, "weights must be specified"
        params = copy.deepcopy(params)
        weights = torch.tensor(params.pop("weights")).unsqueeze(0)
        assert weights.shape[0] == 1 and weights.ndim == 2
        super().__init__(**params)
        self.register_buffer("weights", weights.float(), persistent=False)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        loss = (
            (F.mse_loss(p.float(), t.float(), reduction="none") * self.weights)
            .sum(1)
            .mean()
        )
        return {"loss": loss}


class L1L2Loss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.l1_weight = params["l1_weight"]
        self.l2_weight = params["l2_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        p, t = p.float(), t.float()
        loss_dict = {}
        loss_dict["l1_loss"] = self.l1_func(p, t)
        loss_dict["l2_loss"] = F.mse_loss(p, t)
        loss_dict["loss"] = (
            loss_dict["l1_loss"] * self.l1_weight
            + loss_dict["l2_loss"] * self.l2_weight
        )
        return loss_dict


class WeightedL1L2Loss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        assert "weights" in params, "weights must be specified"
        weights = torch.tensor(params["weights"]).unsqueeze(0)
        assert weights.shape[0] == 1 and weights.ndim == 2
        self.register_buffer("weights", weights.float(), persistent=False)
        self.l1_weight = params["l1_weight"]
        self.l2_weight = params["l2_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        p, t = p.float(), t.float()
        loss_dict = {}
        loss_dict["l1_loss"] = (
            (self.l1_func(p.float(), t.float(), reduction="none") * self.weights)
            .sum(1)
            .mean()
        )
        loss_dict["l2_loss"] = (
            (F.mse_loss(p.float(), t.float(), reduction="none") * self.weights)
            .sum(1)
            .mean()
        )
        loss_dict["loss"] = (
            loss_dict["l1_loss"] * self.l1_weight
            + loss_dict["l2_loss"] * self.l2_weight
        )
        return loss_dict
