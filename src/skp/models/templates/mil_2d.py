"""
Template for a 2D multiple-instance learning model.

Input is expected as `batch["x"]` with shape `(B, N, C, H, W)`, where `N` is
the number of instances in the bag.
"""

import torch
import torch.nn as nn

from timm import create_model
from typing import Dict

from skp.configs import Config
from skp.models.pooling import get_pool_layer


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.num_input_channels,
        )
        with torch.no_grad():
            sample = torch.randn(2, cfg.num_input_channels, cfg.image_height, cfg.image_width)
            feature_dim = get_pool_layer(cfg, dim=2)(self.backbone(sample)).shape[1]
        self.instance_pool = get_pool_layer(cfg, dim=2)
        self.attention = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(feature_dim, cfg.num_classes)
        self.criterion = None

    def forward(self, batch: Dict, return_loss: bool = False) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        b, n = x.shape[:2]
        x = x.reshape(b * n, *x.shape[2:])
        features = self.instance_pool(self.backbone(x)).reshape(b, n, -1)
        weights = self.attention(features).softmax(dim=1)
        bag_features = (features * weights).sum(dim=1)
        logits = self.linear(self.dropout(bag_features))
        out = {"logits": logits, "attention": weights.squeeze(-1)}
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
