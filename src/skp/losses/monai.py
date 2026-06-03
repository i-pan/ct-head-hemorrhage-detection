import copy
import torch.nn as nn

try:
    from monai import losses
except ModuleNotFoundError:
    losses = None


def _require_monai() -> None:
    if losses is None:
        raise ModuleNotFoundError(
            "MONAI is required for skp.losses.monai losses. "
            "Install the optional 3D/medical-imaging dependencies before using this loss module."
        )


class TverskyLoss(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        _require_monai()
        params = copy.deepcopy(params)
        self.loss_fn = losses.TverskyLoss(**params)

    def forward(self, out: dict, batch: dict):
        p, t = out["logits"], batch["y"]
        return {"loss": self.loss_fn(p, t)}


class TverskyFocalLoss(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        _require_monai()
        params = copy.deepcopy(params)
        self.tversky_loss_fn = losses.TverskyLoss(**params.get("tversky", {}))
        self.focal_loss_fn = losses.FocalLoss(**params.get("focal", {}))
        self.tversky_weight = params.get("tversky_weight", 1.0)
        self.focal_weight = params.get("focal_weight", 1.0)

    def forward(self, out: dict, batch: dict):
        p, t = out["logits"], batch["y"]
        tversky_loss = self.tversky_loss_fn(p, t)
        focal_loss = self.focal_loss_fn(p, t)
        loss_dict = {
            "tversky_loss": tversky_loss,
            "focal_loss": focal_loss,
            "loss": self.tversky_weight * tversky_loss + self.focal_weight * focal_loss,
        }
        return loss_dict


class TverskyBCELoss(nn.Module):

    def __init__(self, params: dict):
        super().__init__()
        _require_monai()
        params = copy.deepcopy(params)
        self.tversky_loss_fn = losses.TverskyLoss(**params.get("tversky", {}))
        self.bce_loss_fn = nn.BCEWithLogitsLoss(**params.get("bce", {}))
        self.tversky_weight = params.get("tversky_weight", 1.0)
        self.bce_weight = params.get("bce_weight", 1.0)

    def forward(self, out: dict, batch: dict):
        p, t = out["logits"], batch["y"]
        tversky_loss = self.tversky_loss_fn(p, t)
        bce_loss = self.bce_loss_fn(p.float(), t.float())
        loss_dict = {
            "tversky_loss": tversky_loss,
            "bce_loss": bce_loss,
            "loss": self.tversky_weight * tversky_loss + self.bce_weight * bce_loss,
        }
        return loss_dict
