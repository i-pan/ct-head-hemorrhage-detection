"""
Simple model for 2D classification (or regression)
Uses timm for backbones
"""

import torch
import torch.nn as nn

from timm import create_model
from typing import Dict

from skp.configs.base import Config
from skp.models.normalization import normalize_input
from skp.models.pooling import get_pool_layer
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        backbone_args = {
            "pretrained": self.cfg.pretrained,
            "num_classes": 0,
            "global_pool": "",
            "features_only": self.cfg.get("features_only", False),
            "in_chans": self.cfg.num_input_channels,
        }
        if self.cfg.get("backbone_img_size", False):
            # some models require specifying image size (e.g., coatnet)
            if "efficientvit" in self.cfg.backbone:
                backbone_args["img_size"] = self.cfg.image_height
            else:
                backbone_args["img_size"] = (
                    self.cfg.image_height,
                    self.cfg.image_width,
                )

        self.backbone = create_model(self.cfg.backbone, **backbone_args)

        # get feature dim by passing sample through net
        self.feature_dim = self.backbone(
            torch.randn(
                (
                    2,
                    self.cfg.num_input_channels,
                    self.cfg.image_height,
                    self.cfg.image_width,
                )
            )
        ).size(
            -1 if "xcit" in self.cfg.backbone else 1
        )  # xcit models are channels-last

        if self.cfg.get("enable_gradient_checkpointing", False):
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        self.feature_dim = self.feature_dim * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = get_pool_layer(self.cfg, dim=2)

        self.dropout = nn.Dropout(p=self.cfg.dropout)
        if self.cfg.num_classes is None or self.cfg.num_classes == 0:
            print("`num_classes` is None or 0, using nn.Identity() for final linear layer ...")
            self.linear = nn.Identity()            
        else:
            self.linear = nn.Linear(self.feature_dim, self.cfg.num_classes)

        if self.cfg.get("load_pretrained_backbone"):
            self.load_pretrained_backbone()

        if self.cfg.get("load_pretrained_model"):
            self.load_pretrained_model()

        self.criterion = None

        self.backbone_frozen = False
        if self.cfg.get("freeze_backbone", False):
            self.freeze_backbone()

    def forward(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y = batch.get("y", None)

        if return_loss:
            assert y is not None

        features = self.extract_features(x, normalize=True)

        if self.cfg.get("multisample_dropout", False):
            logits = torch.stack(
                [self.linear(self.dropout(features)) for _ in range(5)]
            ).mean(0)
        else:
            logits = self.linear(self.dropout(features))

        if self.cfg.get("model_activation_fn") == "sigmoid":
            logits = logits.sigmoid()
        elif self.cfg.get("model_activation_fn") == "softmax":
            logits = logits.softmax(dim=1)
        elif self.cfg.get("model_activation_fn") == "relu":
            logits = logits.relu()

        out = {"logits": logits}
        if return_features:
            out["features"] = features
        if return_loss:
            loss = self.criterion(out, batch)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.normalize(x) if normalize else x
        return self.pooling(self.backbone(x))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_input(x, self.cfg)

    def load_pretrained_backbone(self) -> None:
        print(
            f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ..."
        )
        weights = torch_load_weights(self.cfg.load_pretrained_backbone)
        backbone_weights = filter_weights_by_prefix(weights, "model.backbone.")
        # sometimes loading encoder from trained segmentation model as backbone
        if len(backbone_weights) == 0:
            backbone_weights = filter_weights_by_prefix(weights, "model.encoder.")
        if len(backbone_weights) == 0:
            # seg_cls model
            backbone_weights = filter_weights_by_prefix(
                weights, "model.segmenter.encoder."
            )
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_weights, strict=False
        )
        if len(missing_keys) > 0:
            print(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected keys: {unexpected_keys}")

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        weights = filter_weights_by_prefix(weights, "model.")
        self.load_state_dict(weights)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
