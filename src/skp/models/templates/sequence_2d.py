"""
Template for a 2D slice/image-sequence model.

Input is expected as `batch["x"]` with shape `(B, T, C, H, W)`.
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
        self.pool = get_pool_layer(cfg, dim=2)
        self.sequence = nn.GRU(
            input_size=feature_dim,
            hidden_size=cfg.get("sequence_hidden_dim", feature_dim) or feature_dim,
            batch_first=True,
            bidirectional=cfg.get("sequence_bidirectional", True) or False,
        )
        head_dim = self.sequence.hidden_size * (2 if self.sequence.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(head_dim, cfg.num_classes)
        self.criterion = None

    def forward(self, batch: Dict, return_loss: bool = False) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        features = self.pool(self.backbone(x)).reshape(b, t, -1)
        seq_features, _ = self.sequence(features)
        logits = self.linear(self.dropout(seq_features[:, -1]))
        out = {"logits": logits, "features_seq": seq_features}
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
