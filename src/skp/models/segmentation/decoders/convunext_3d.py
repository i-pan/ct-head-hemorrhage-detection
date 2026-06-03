"""
Adapted from:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint
from typing import List

from skp.configs import Config


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
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
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        kernel_size: tuple[int, int, int] | int = 7,
        exp_factor: int | float = 4,
        drop_path: float = 0.0,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

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
        x = self.drop_path(x)

        if hasattr(self, "res_conv"):
            x += self.res_conv(input)
        elif x.shape == input.shape:
            x += input

        return x


class DecoderBlock(ConvNeXtBlock):
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        size: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            t, h, w = skip.shape[-3:]
            x = F.interpolate(x, size=(t, h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
        else:
            assert size is not None
            x = F.interpolate(x, size=size, mode="nearest")
        x = super().forward(x)
        return x


class ConvUNeXtDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        # reverse channels to start from head of encoder
        encoder_channels = self.cfg.encoder_channels[::-1]

        if self.cfg.decoder_n_blocks != len(self.cfg.decoder_out_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_out_channels` for {} blocks.".format(
                    self.cfg.decoder_n_blocks, len(self.cfg.decoder_out_channels)
                )
            )

        if self.cfg.decoder_n_blocks != len(encoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `encoder_channels` for {} blocks.".format(
                    self.cfg.decoder_n_blocks, len(encoder_channels)
                )
            )

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(self.cfg.decoder_out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = self.cfg.decoder_out_channels

        if self.cfg.decoder_center_block:
            self.center = ConvNeXtBlock(
                dim=head_channels,
                kernel_size=3,
            )
        else:
            self.center = nn.Identity()

        blocks = [
            DecoderBlock(
                in_ch + skip_ch,
                out_ch,
                kernel_size=self.cfg.decoder_block_kernel_size or (3, 7, 7),
                exp_factor=self.cfg.decoder_block_exp_factor or 4,
                use_res_conv=self.cfg.decoder_block_use_res_conv or False,
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]

        # optional additional blocks after each decoder block
        post_blocks = [
            ConvNeXtBlock(
                out_ch,
                kernel_size=self.cfg.decoder_block_kernel_size or (3, 7, 7),
                exp_factor=self.cfg.decoder_block_exp_factor or 4,
            )
            if self.cfg.decoder_post_blocks_enable
            else nn.Identity()
            for out_ch in out_channels
        ]
        self.blocks = nn.ModuleList(blocks)
        self.post_blocks = nn.ModuleList(post_blocks)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        output = [self.center(head)]
        for i, (decoder_block, post_block) in enumerate(
            zip(self.blocks, self.post_blocks)
        ):
            skip = skips[i] if i < len(skips) else None
            size = self.cfg.output_size if skip is None else None
            if self.cfg.enable_gradient_checkpointing:
                out = checkpoint(
                    decoder_block, *(output[-1], skip, size), use_reentrant=False
                )
                out = checkpoint(post_block, *(out,), use_reentrant=False)
            else:
                out = post_block(decoder_block(output[-1], skip, size))
            output.append(out)

        return output
