"""NaFlexViT adapter retaining the common slice classifier loss and head.

Native GAP excludes register tokens and retains pretrained final normalization.
Images remain fixed-size tensors; no NaFlex loader or variable batch sizing.
"""

import torch
from torch import nn
from timm import create_model
from skp.models.classification.net2d import Net as SliceNet


class Net(SliceNet):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        if cfg.pool != "avg":
            raise ValueError("This ablation requires average pooling")
        self.cfg = cfg
        self.backbone = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3 if cfg.pretrained else cfg.num_input_channels,
            num_classes=0,
            global_pool="avg",
        )
        if cfg.pretrained and cfg.num_input_channels != 3:
            adapt_patch_channels(self.backbone, cfg.num_input_channels)
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, cfg.num_classes)
        self.criterion = None
        self.backbone_frozen = False

    def extract_features(self, x, normalize=True):
        if x.ndim != 4 or x.shape[1] != self.cfg.num_input_channels:
            raise ValueError("Expected B x window/context channels x H x W")
        return self.backbone(self.normalize(x) if normalize else x)


def adapt_patch_channels(backbone, channels):
    """Repeat RGB weights per spatial patch position, preserving HWC ordering."""
    embed = backbone.embeds
    old = embed.proj
    if (
        not isinstance(old, nn.Linear)
        or embed.norm_input is not None
        or embed.enable_patch_interpolator
    ):
        raise ValueError("Expected the fixed-patch linear NaFlex embedding")
    if channels % 3:
        raise ValueError(
            "This ablation requires a multiple of three CT-window channels"
        )
    ph, pw = embed.patch_size
    new = nn.Linear(ph * pw * channels, old.out_features, bias=old.bias is not None)
    with torch.no_grad():
        weight = old.weight.reshape(old.out_features, ph, pw, 3)
        new.weight.copy_(
            weight.repeat(1, 1, 1, channels // 3).reshape_as(new.weight)
            * (3 / channels)
        )
        if old.bias is not None:
            new.bias.copy_(old.bias)
    embed.proj = new
    embed.in_chans = channels
    backbone.in_chans = channels
