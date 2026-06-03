"""
MedNeXt wrapper with a stochastic 1x1x1 segmentation head.

The main output samples a subset of class-specific head weights per forward
pass. This is useful for very large class spaces where computing every class
logit is unnecessarily expensive.
"""

import re
import torch
import torch.nn as nn

from typing import Dict, List, Union

from skp.configs import Config
from skp.models.segmentation.mednext import create_mednext_v1
from skp.models.segmentation.mednext.stochastic_head import (
    StochasticWeightSegmentationHead,
)
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


def normalize_volume_clip_rescale(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes each spatial volume (D, H, W) in a 5D tensor (B, C, D, H, W) by:
      1. Computing the 1st and 99th percentiles of that volume.
      2. Clipping the volume values to this range.
      3. Rescaling the clipped values to the range [-1, 1].

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        eps (float): Small value to prevent division by zero for constant volumes.

    Returns:
        torch.Tensor: Normalized tensor of shape (B, C, D, H, W).
    """
    if x.ndim != 5:
        raise ValueError(
            f"Expected input to be 5D (B, C, D, H, W), got shape {x.shape}"
        )

    # Ensure input is floating point for quantile calculation
    if not x.is_floating_point():
        x = x.float()

    B, C, D, H, W = x.shape

    # 1. Compute Percentiles
    # Flatten spatial dims (D, H, W) to calculate quantiles over the volume
    try:
        x_flat = x.view(B, C, -1)
    except RuntimeError:
        x_flat = x.reshape(B, C, -1)

    # Calculate 1st and 99th percentiles along the last dimension (the flattened volume)
    # keepdim=True ensures shape is (B, C, 1)
    p1 = torch.quantile(x_flat, 0.01, dim=-1, keepdim=True)
    p99 = torch.quantile(x_flat, 0.99, dim=-1, keepdim=True)

    # Reshape percentiles to (B, C, 1, 1, 1) for broadcasting against (B, C, D, H, W)
    p1 = p1.view(B, C, 1, 1, 1)
    p99 = p99.view(B, C, 1, 1, 1)

    # 2. Clip
    # Values below p1 become p1, values above p99 become p99
    x_clipped = torch.clamp(x, min=p1, max=p99)

    # 3. Rescale to [-1, 1]
    # Formula: -1 + 2 * (x - min) / (max - min)
    # Using eps to handle cases where the volume is constant (p99 == p1)
    x_norm = -1.0 + 2.0 * (x_clipped - p1) / (p99 - p1 + eps)

    return x_norm


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mednext = getattr(create_mednext_v1, self.cfg.backbone)(
            num_input_channels=cfg.num_input_channels,
            num_classes=cfg.num_classes,
            use_conv_transpose=cfg.get("use_conv_transpose", False) or False,
            checkpoint_style="outside_block"
            if cfg.get("enable_gradient_checkpointing", False)
            else None,
            ds=cfg.get("deep_supervision", False) or False,
            ds_levels=cfg.get("ds_levels", 1) or 1,
        )

        if self.cfg.get("enable_gradient_checkpointing", False):
            print("Enabling gradient checkpointing ...")
            assert self.mednext.outside_block_checkpointing

        self.criterion = None

        if self.cfg.num_sample_classes is None:
            raise ValueError("cfg.num_sample_classes is required for stochastic MedNeXt")

        self.stochastic_head = StochasticWeightSegmentationHead(
            in_channels=self.mednext.out_0.conv_out.in_channels,
            num_total_classes=self.mednext.out_0.conv_out.out_channels,
            num_sample_classes=self.cfg.num_sample_classes,
            ignore_background_class_for_sampling=self.cfg.get(
                "ignore_background_class_for_sampling", False
            ),
        )

        self.mednext.out_0 = nn.Identity()

        if self.cfg.get("load_pretrained_weights"):
            self.load_pretrained_weights()

        if self.cfg.get("freeze_encoder", False):
            print("Freezing encoder ...")
            self.freeze_encoder()

        if self.cfg.get("freeze_decoder", False):
            print("Freezing decoder ...")
            self.freeze_decoder()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
        return_decoder_output: bool = False,
        indices: list[int] | None = None,
        seed: int | None = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x = batch["x"]
        y = batch["y"] if "y" in batch else None
        if return_loss:
            assert y is not None

        x = self.normalize(x)
        mednext_out = self.mednext(x)

        if self.cfg.get("deep_supervision", False):
            logits, class_indices = self.stochastic_head(
                mednext_out[0], y=y, indices=indices, seed=seed
            )
            out = {
                "logits": logits,
                "class_indices": class_indices,
            }
            out.update(
                {f"logits_ds{i}": mednext_out[i] for i in range(1, len(mednext_out))}
            )
        else:
            logits, class_indices = self.stochastic_head(
                mednext_out, y=y, indices=indices, seed=seed
            )
            out = {"logits": logits, "class_indices": class_indices}
        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def normalize(self, x):
        if self.cfg.normalization == "-1_1":
            mini, maxi = (
                self.cfg.normalization_params["min"],
                self.cfg.normalization_params["max"],
            )
            x = x - mini
            x = x / (maxi - mini)
            x = x - 0.5
            x = x * 2.0
        elif self.cfg.normalization == "0_1":
            mini, maxi = (
                self.cfg.normalization_params["min"],
                self.cfg.normalization_params["max"],
            )
            x = x - mini
            x = x / (maxi - mini)
        elif self.cfg.normalization == "mean_sd":
            mean, sd = (
                self.cfg.normalization_params["mean"],
                self.cfg.normalization_params["sd"],
            )
            x = (x - mean) / sd
        elif self.cfg.normalization == "per_channel_mean_sd":
            mean, sd = (
                self.cfg.normalization_params["mean"],
                self.cfg.normalization_params["sd"],
            )
            assert len(mean) == len(sd) == x.size(1)
            shape = (1, x.size(1), *([1] * (x.ndim - 2)))
            mean = x.new_tensor(mean).view(shape)
            sd = x.new_tensor(sd).view(shape)
            x = (x - mean) / sd
        elif self.cfg.normalization == "percentile_1_99":
            x = normalize_volume_clip_rescale(x)
        elif self.cfg.normalization == "clip_hu_0_1":
            assert "hu_min" in self.cfg.normalization_params
            assert "hu_max" in self.cfg.normalization_params
            # Clip based on specified HU values
            x.clamp_(
                self.cfg.normalization_params["hu_min"],
                self.cfg.normalization_params["hu_max"],
            )
            # Convert to 0, 1 range
            x = (x - self.cfg.normalization_params["hu_min"]) / (
                self.cfg.normalization_params["hu_max"]
                - self.cfg.normalization_params["hu_min"]
            )
        elif self.cfg.normalization == "none":
            x = x
        else:
            raise NotImplementedError
        return x

    def load_pretrained_weights(self) -> None:
        print(f"Loading pretrained weights from {self.cfg.load_pretrained_weights} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_weights)
        weights = filter_weights_by_prefix(weights, "model.")
        if self.cfg.get("ignore_weights"):
            for each_ignore in self.cfg.ignore_weights:
                weights = {
                    k: v for k, v in weights.items() if not k.startswith(each_ignore)
                }
        missing_keys, unexpected_keys = self.load_state_dict(
            weights,
            strict=True if self.cfg.get("load_weights_strict", False) else False,
        )
        if len(missing_keys) > 0:
            print(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected keys: {unexpected_keys}")

    def freeze_decoder(self) -> None:
        for name, param in self.mednext.named_parameters():
            # Starts with up_ or dec_
            if re.search(r"^(up_|dec_)", name):
                param.requires_grad = False

    def freeze_encoder(self) -> None:
        for name, param in self.mednext.named_parameters():
            # Starts with stem, enc_, down_, or bottleneck
            if re.search(r"^(stem|enc_|down_|bottleneck)", name):
                param.requires_grad = False

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
