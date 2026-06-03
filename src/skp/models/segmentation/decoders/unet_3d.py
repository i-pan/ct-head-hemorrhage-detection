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


class SeparableConv3d(nn.Sequential):
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
        depthwise_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(depthwise_conv, pointwise_conv)


class Conv3dAct(nn.Sequential):
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
            NormLayer = nn.BatchNorm3d
        elif norm_layer in {"gn", "group_norm"}:
            NormLayer = partial(nn.GroupNorm, num_groups=num_groups)
        else:
            raise Exception(
                f"`norm_layer` must be one of [`bn`, `batch_norm`, `gn`, `group_norm`], got `{norm_layer}`"
            )
        super().__init__()
        conv_layer = SeparableConv3d if separable else nn.Conv3d
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
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            getattr(nn, activation)(inplace=inplace),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
        )
        self.sSE = nn.Conv3d(in_channels, 1, 1)

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
        self.conv1 = Conv3dAct(
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
        self.conv2 = Conv3dAct(
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
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        size: Optional[tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if skip is not None:
            t, h, w = skip.shape[2:]
            x = F.interpolate(x.contiguous(), size=(t, h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            if size is not None and size != x.shape[2:]:
                x = F.interpolate(x.contiguous(), size=size, mode="nearest")
            else:
                # if no skip connection and size not specified,
                # upsample 2 in all dimensions
                x = F.interpolate(
                    x.contiguous(), scale_factor=(2, 2, 2), mode="nearest"
                )
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
        separable: bool = False,
    ):
        conv1 = Conv3dAct(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            separable=separable,
        )
        conv2 = Conv3dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            separable=separable,
        )
        super().__init__(conv1, conv2)


class Unet3dDecoder(nn.Module):
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
        size = None
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            if i + 1 == len(self.blocks):
                size = self.cfg.output_size or (
                    self.cfg.num_slices,
                    self.cfg.image_height,
                    self.cfg.image_width,
                )
            output.append(decoder_block(output[-1], skip, size))

        return output
