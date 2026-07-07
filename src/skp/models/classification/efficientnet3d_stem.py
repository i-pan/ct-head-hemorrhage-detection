"""2.5D classification model with a 3D EfficientNet stem."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from timm import create_model
from timm.layers.conv2d_same import Conv2dSame
from timm.layers.padding import pad_same

from skp.configs.base import Config
from skp.models.normalization import normalize_input
from skp.models.pooling import get_pool_layer
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


class Conv3dTo2dStem(nn.Module):
    """Replace a 2D stem with a 3D stem that collapses the slice axis."""

    def __init__(self, conv2d: nn.Conv2d, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        if conv2d.groups != 1:
            raise ValueError("Grouped 2D stem conversion is not supported.")

        self.depth = depth
        self.same_padding = isinstance(conv2d, Conv2dSame)
        self.stride_hw = conv2d.stride
        self.dilation_hw = conv2d.dilation
        padding = (0, 0, 0) if self.same_padding else (0, *conv2d.padding)
        self.conv = nn.Conv3d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=(depth, *conv2d.kernel_size),
            stride=(1, *conv2d.stride),
            padding=padding,
            dilation=(1, *conv2d.dilation),
            bias=conv2d.bias is not None,
        )
        with torch.no_grad():
            weight = conv2d.weight.unsqueeze(2).repeat(1, 1, depth, 1, 1)
            self.conv.weight.copy_(weight / depth)
            if conv2d.bias is not None:
                self.conv.bias.copy_(conv2d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape B,C,D,H,W, got {tuple(x.shape)}")
        if x.size(2) != self.depth:
            raise ValueError(f"Expected depth {self.depth}, got {x.size(2)}")
        if self.same_padding:
            x = pad_same(
                x,
                self.conv.weight.shape[-2:],
                self.stride_hw,
                self.dilation_hw,
            )
        x = self.conv(x)
        return x.squeeze(2)


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.backbone = create_model(
            self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=self.cfg.num_input_channels,
        )
        self.backbone.conv_stem = Conv3dTo2dStem(
            self.backbone.conv_stem,
            depth=self.cfg.num_slices,
        )

        if self.cfg.get("enable_gradient_checkpointing", False):
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        self.feature_dim = self.backbone.num_features
        if self.cfg.pool == "catavgmax":
            self.feature_dim *= 2
        self.pooling = get_pool_layer(self.cfg, dim=2)
        self.dropout = nn.Dropout(p=self.cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, self.cfg.num_classes)

        if self.cfg.get("load_pretrained_backbone"):
            self.load_pretrained_backbone()
        if self.cfg.get("load_pretrained_model"):
            self.load_pretrained_model()

        self.criterion = None

    def forward(
        self,
        batch: Dict,
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_features(batch["x"], normalize=True)
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
            out.update(self.criterion(out, batch))
        return out

    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.normalize(x) if normalize else x
        return self.pooling(self.backbone.forward_features(x))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_input(x, self.cfg)

    def load_pretrained_backbone(self) -> None:
        print(
            f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ..."
        )
        weights = torch_load_weights(self.cfg.load_pretrained_backbone)
        backbone_weights = filter_weights_by_prefix(weights, "model.backbone.")
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_weights,
            strict=False,
        )
        if missing_keys:
            print(f"missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"unexpected keys: {unexpected_keys}")

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        weights = filter_weights_by_prefix(weights, "model.")
        self.load_state_dict(weights)

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
