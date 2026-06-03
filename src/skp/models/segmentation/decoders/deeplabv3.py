"""
Adapted from:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/deeplabv3/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Callable, Iterable, List, Tuple, Union


class LayerNorm(nn.GroupNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-5,
        affine: bool = True,
    ):
        if isinstance(normalized_shape, int):
            num_channels = normalized_shape
        else:
            num_channels = 1
            for dim in normalized_shape:
                num_channels *= dim
        super().__init__(
            num_groups=1, num_channels=num_channels, eps=eps, affine=affine
        )


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # following ASPP-L in https://arxiv.org/pdf/1606.00915
        if self.cfg.use_psp:
            self.psp_pool_sizes = self.cfg.psp_pool_sizes or (1, 2, 3, 6)
        else:
            self.atrous_rates = self.cfg.atrous_rates or (6, 12, 18, 24)
            self.aspp_separable = self.cfg.aspp_separable or False
            self.aspp_dropout = self.cfg.aspp_dropout or 0.0

        decoder_norm_layer = self.cfg.decoder_norm_layer or "batch_norm"
        if decoder_norm_layer in {"bn", "batch_norm"}:
            norm_layer = nn.BatchNorm2d
        elif decoder_norm_layer == "layer_norm":
            norm_layer = LayerNorm
        elif decoder_norm_layer in {"gn", "group_norm"}:
            norm_layer = partial(nn.GroupNorm, 8)
        else:
            raise ValueError(f"Unsupported decoder_norm_layer: {decoder_norm_layer}")

        decoder_act_layer = self.cfg.decoder_act_layer or "relu"
        if decoder_act_layer == "relu":
            act_layer = nn.ReLU()
        elif decoder_act_layer == "gelu":
            act_layer = nn.GELU()
        elif decoder_act_layer == "silu":
            act_layer = nn.SiLU()
        else:
            raise ValueError(f"Unsupported decoder_act_layer: {decoder_act_layer}")

        if self.cfg.use_psp:
            psp_out_channels = (
                self.cfg.psp_out_channels or self.cfg.encoder_channels[-1]
            )
            self.decoder_out_channels = psp_out_channels * 2
            # just use same aspp attribute name even for PSP
            self.aspp = PSPModule(
                in_channels=self.cfg.encoder_channels[-1],
                out_channels=psp_out_channels,
                sizes=self.cfg.psp_pool_sizes,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.decoder_out_channels = self.cfg.decoder_out_channels or 256
            self.aspp = nn.Sequential(
                ASPP(
                    self.cfg.encoder_channels[-1],
                    self.decoder_out_channels,
                    self.atrous_rates,
                    separable=self.aspp_separable,
                    dropout=self.aspp_dropout,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ),
                SeparableConv2d(
                    self.decoder_out_channels,
                    self.decoder_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                make_norm(norm_layer, self.decoder_out_channels),
                act_layer,
            )

        highres_in_channels = self.cfg.encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            make_norm(norm_layer, highres_out_channels),
            act_layer,
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + self.decoder_out_channels,
                self.decoder_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_norm(norm_layer, self.decoder_out_channels),
            act_layer,
        )

    def forward(self, features):
        aspp_features = self.aspp(features[-1])
        high_res_features = self.block1(features[-4])
        aspp_features = F.interpolate(
            aspp_features, size=high_res_features.shape[2:], mode="bilinear"
        )
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return aspp_features, fused_features


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            make_norm(norm_layer, out_channels),
            act_layer,
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            make_norm(norm_layer, out_channels),
            act_layer,
        )


class ASPPPooling(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            make_norm(norm_layer, out_channels),
            act_layer,
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        separable: bool,
        dropout: float,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(norm_layer, out_channels),
                act_layer,
            )
        ]

        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv
        for rate in atrous_rates:
            modules.append(
                ASPPConvModule(
                    in_channels,
                    out_channels,
                    rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )

        modules.append(
            ASPPPooling(
                in_channels, out_channels, norm_layer=norm_layer, act_layer=act_layer
            )
        )

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(
                (len(atrous_rates) + 2) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            make_norm(norm_layer, out_channels),
            act_layer,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Conv2dAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
        separable: bool = False,
    ):
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
        self.norm = make_norm(norm_layer, out_channels)
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int,
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        if pool_size == 1:
            norm_layer = LayerNorm  # PyTorch does not support BatchNorm for 1x1 shape

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dAct(
                in_channels,
                out_channels,
                (1, 1),
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        x = self.pool(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        sizes: tuple[int, ...] = (1, 2, 3, 6),
        norm_layer: Callable = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        assert out_channels % len(sizes) == 0

        if out_channels != in_channels:
            self.shortcut = Conv2dAct(
                in_channels,
                out_channels,
                (1, 1),
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.shortcut = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels, out_channels // len(sizes), size, norm_layer, act_layer
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [self.shortcut(x)]
        x = torch.cat(xs, dim=1)
        return x


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


def make_norm(norm_layer: Callable, out_channels: int) -> nn.Module:
    if isinstance(norm_layer, partial):
        return norm_layer(num_channels=out_channels)
    return norm_layer(out_channels)
