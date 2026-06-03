"""
Adapted from:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import List, Optional

from skp.configs import Config


class LayerNorm(nn.Module):
    """LayerNorm for 3D tensors in channels-last or channels-first format.

    channels_last expects (batch, depth, height, width, channels), while
    channels_first expects (batch, channels, depth, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class GRN3d(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is channels-last
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        kernel_size: tuple[int, int, int] | int = 3,
        exp_factor: int | float = 4,
        use_res_conv: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        if dim_out is None:
            dim_out = dim
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        exp_dim = int(dim * exp_factor)
        self.pwconv1 = nn.Linear(
            dim, exp_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN3d(exp_dim)
        self.pwconv2 = nn.Linear(exp_dim, dim_out)

        if use_res_conv and dim != dim_out:
            # if specified, use conv layer for skip connection if input and output
            # dims are not the same
            self.res_conv = nn.Conv3d(dim, dim_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = rearrange(x, "n c t h w -> n t h w c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = rearrange(x, "n t h w c -> n c t h w")

        if hasattr(self, "res_conv"):
            x += self.res_conv(input)
        elif x.shape == input.shape:
            x += input

        return x


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
        attention_type: Optional[str] = None,
        use_res_conv: bool = False,
        single_block: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvNeXtBlock(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            exp_factor=4,
            use_res_conv=use_res_conv,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        if single_block:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = ConvNeXtBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                exp_factor=4,
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
            x = F.interpolate(x, size=(t, h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            if size is not None:
                x = F.interpolate(x, size=size, mode="nearest")
            else:
                # if no skip connection and size not specified,
                # upsample 2 in all dimensions
                x = F.interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_res_conv: bool = False,
    ):
        conv1 = ConvNeXtBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            exp_factor=4,
            use_res_conv=use_res_conv,
        )
        conv2 = ConvNeXtBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            exp_factor=4,
        )
        super().__init__(conv1, conv2)


class UneXt3dDecoder(nn.Module):
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
                head_channels,
                head_channels,
                use_res_conv=self.cfg.get("decoder_use_res_conv", False),
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            attention_type=self.cfg.get("decoder_attention_type"),
            use_res_conv=self.cfg.get("decoder_use_res_conv", False),
            single_block=self.cfg.get("decoder_single_block", False),
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
