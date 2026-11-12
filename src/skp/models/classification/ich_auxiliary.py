"""Stride-16 classification with binary decoder and spatial any-logit supervision."""

from importlib import import_module
from torch import nn
from torch.nn import functional as F
from skp.models.segmentation.decoders.unet import DecoderBlock


def auxiliary_losses(
    segmentation, logit_map, target, valid=None, roi=None, positive=None, outside_weight=0.0
):
    if valid is not None:
        if not valid.any():
            zero = segmentation.float().sum() * 0 + logit_map.float().sum() * 0
            return zero, zero, zero
        segmentation, logit_map, target = (
            segmentation[valid],
            logit_map[valid],
            target[valid],
        )
        if roi is not None:
            roi = roi[valid]
        if positive is not None:
            positive = positive[valid]
    target = target.float()
    segmentation = segmentation.float()
    if roi is None:
        roi = target.new_ones(target.shape)
    roi = roi.float()
    if not 0 <= outside_weight <= 1:
        raise ValueError("outside_weight must be in [0, 1]")
    weights = roi + outside_weight * (1 - roi)
    bce = (
        F.binary_cross_entropy_with_logits(segmentation, target, reduction="none") * weights
    ).sum() / weights.sum().clamp_min(1)
    probability = segmentation.sigmoid()
    dice_roi = weights if positive is None else weights * positive[:, None, None, None].to(weights)
    dice = 1 - (2 * (probability * target * dice_roi).sum() + 1e-5) / (
        (probability * dice_roi).sum() + (target * dice_roi).sum() + 1e-5
    )
    coarse = F.adaptive_max_pool2d(
        target * (roi if outside_weight == 0 else 1), logit_map.shape[-2:]
    )
    coarse_roi = F.adaptive_max_pool2d(weights, logit_map.shape[-2:])
    spatial = (
        F.binary_cross_entropy_with_logits(logit_map.float(), coarse, reduction="none")
        * coarse_roi
    ).sum() / coarse_roi.sum().clamp_min(1)
    return bce, dice, spatial


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base = import_module(f"skp.models.{cfg.auxiliary_base_model}").Net(cfg)
        self.backbone = base.backbone
        self.linear = base.linear
        self.dropout = base.dropout
        self.pooling = base.pooling
        self.normalize = base.normalize
        self.feature_dim = base.feature_dim
        self.feature_indices = list(
            range(len(self.backbone.feature_info) - 4, len(self.backbone.feature_info))
        )
        channels = [x["num_chs"] for x in self.backbone.feature_info[-4:]]
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(self.feature_dim, channels[-2], 256),
                DecoderBlock(256, channels[-3], 128),
                DecoderBlock(128, channels[-4], 64),
                DecoderBlock(64, 0, 32),
                DecoderBlock(32, 0, 16),
            ]
        )
        self.segmentation_head = nn.Conv2d(16, 1, 1)
        self.any_index = cfg.label_columns.index("any")
        self.criterion = None

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(
        self, batch, return_loss=False, return_features=False, return_segmentation=False
    ):
        auxiliary = self.training and return_loss
        x = self.normalize(batch["x"])
        if auxiliary or return_segmentation:
            last, skips = self.backbone.forward_intermediates(
                x, indices=self.feature_indices
            )
            features = self.backbone.forward_head(last)
        else:
            features = self.backbone(x)
        assert features.shape[-2:] == (32, 32)
        dropped = F.dropout2d(features, p=self.dropout.p, training=self.training)
        logit_map = self.linear(dropped.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = {"logits": logit_map.flatten(2).max(-1).values}
        if return_features:
            out.update(features=self.pooling(features), logit_map=logit_map)
        if auxiliary or return_segmentation:
            decoded = features
            for j, block in enumerate(self.decoder):
                decoded = block(decoded, skips[-2 - j] if j < 3 else None)
            segmentation = self.segmentation_head(decoded)
            if return_segmentation:
                out["segmentation_logits"] = segmentation
        if return_loss:
            out.update(self.criterion(out, batch))
            out["classification_loss"] = out["loss"]
            if auxiliary:
                assert segmentation.shape == batch["segmentation_target"].shape
                bce, dice, spatial = auxiliary_losses(
                    segmentation,
                    logit_map[:, self.any_index : self.any_index + 1],
                    batch["segmentation_target"],
                    batch.get("segmentation_valid"),
                    batch.get("auxiliary_roi"),
                    positive=batch["y"][:, self.any_index] > 0.5,
                    outside_weight=self.cfg.get("auxiliary_outside_roi_weight", 0.0),
                )
                out.update(
                    segmentation_bce_loss=bce,
                    segmentation_dice_loss=dice,
                    spatial_loss=spatial,
                )
                out["loss"] = (
                    out["loss"]
                    + self.cfg.segmentation_loss_weight * (bce + dice)
                    + self.cfg.spatial_loss_weight * spatial
                )
        return out
