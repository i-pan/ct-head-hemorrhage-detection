"""
Adapted from:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List, Optional

from skp.configs import Config


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(depthwise_conv, pointwise_conv)


class Conv2dAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        norm_layer: str = "bn",
        num_groups: int = 32,  # for GroupNorm,
        activation: str = "ReLU",
        inplace: bool = True,  # for activation
        separable: bool = False,
    ):
        if norm_layer in {"bn", "batch_norm"}:
            NormLayer = nn.BatchNorm2d
        elif norm_layer in {"gn", "group_norm"}:
            NormLayer = partial(nn.GroupNorm, num_groups=num_groups)
        else:
            raise Exception(
                f"`norm_layer` must be one of [`bn`, `batch_norm`, `gn`, `group_norm`], got `{norm_layer}`"
            )
        super().__init__()
        conv_layer = SeparableConv2d if separable else nn.Conv2d
        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = NormLayer(out_channels)
        self.act = getattr(nn, activation)(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SCSEModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        activation: str = "ReLU",
        inplace: bool = False,
    ):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            getattr(nn, activation)(inplace=inplace),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
        )
        self.sSE = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x).sigmoid() + x * self.sSE(x).sigmoid()


class Attention(nn.Module):
    def __init__(self, name: str, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: str = "bn",
        activation: str = "ReLU",
        attention_type: Optional[str] = None,
        separable: bool = False,
    ):
        super().__init__()
        self.conv1 = Conv2dAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            separable=separable,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            separable=separable,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if skip is not None:
            h, w = skip.shape[2:]
            x = F.interpolate(x, size=(h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: str = "bn",
        activation: str = "ReLU",
    ):
        conv1 = Conv2dAct(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        if self.cfg.decoder_n_blocks != len(self.cfg.decoder_out_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_out_channels` for {} blocks.".format(
                    self.cfg.decoder_n_blocks, len(self.cfg.decoder_out_channels)
                )
            )

        # reverse channels to start from head of encoder
        encoder_channels = self.cfg.encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(self.cfg.decoder_out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = self.cfg.decoder_out_channels

        if self.cfg.decoder_center_block:
            self.center = CenterBlock(
                head_channels, head_channels, norm_layer=self.cfg.decoder_norm_layer
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            norm_layer=self.cfg.decoder_norm_layer,
            attention_type=self.cfg.decoder_attention_type,
            separable=self.cfg.decoder_separable_conv,
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        output = [self.center(head)]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            output.append(decoder_block(output[-1], skip))

        return output
