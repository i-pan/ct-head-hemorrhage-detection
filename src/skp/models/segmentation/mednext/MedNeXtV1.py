"""
FROM: https://github.com/MIC-DKFZ/MedNeXt/

Modifications:
- Added default option to use upsampling + normal conv instead of transposed conv
- Changed default to use GRN in bottom 2 encoder/decoder layers and bottleneck
  - Using GRN for all layers increased memory and compute significantly
- Refactored MedNext class with encoder and decoder rather than individual blocks
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    MedNeXtUpBlockV2,
    OutBlock,
)


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        exp_r: list,  # Expected length: 5 [enc0, enc1, enc2, enc3, bottleneck]
        kernel_size: int,
        block_counts: list,  # Expected length: 5
        do_res: bool = False,
        do_res_up_down: bool = False,
        norm_type="group",
        dim="3d",
        checkpoint_style=None,
    ):
        super().__init__()

        self.dim = dim
        self.checkpoint_style = checkpoint_style
        self.dummy_tensor = (
            nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            if checkpoint_style == "outside_block"
            else None
        )

        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d

        # Stem
        self.stem = conv(in_channels, n_channels, kernel_size=1)

        self.stages = nn.ModuleList()
        current_channels = n_channels

        # Build 4 Encoder Stages
        for i in range(4):
            stage = nn.ModuleDict()

            # Logic from original: grn=False for first 2 stages, True for rest
            use_grn = True if i >= 2 else False

            # 1. Blocks
            stage["blocks"] = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=current_channels,
                        out_channels=current_channels,
                        exp_r=exp_r[i],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        dim=dim,
                        grn=use_grn,
                    )
                    for _ in range(block_counts[i])
                ]
            )

            # 2. Downsampling
            stage["down"] = MedNeXtDownBlock(
                in_channels=current_channels,
                out_channels=current_channels * 2,
                exp_r=exp_r[i + 1] if i < 3 else exp_r[i],
                kernel_size=kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=use_grn,
            )

            self.stages.append(stage)
            current_channels *= 2

        # Bottleneck (Part of Encoder)
        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    exp_r=exp_r[4],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=True,
                )
                for _ in range(block_counts[4])
            ]
        )

    def iterative_checkpoint(self, sequential_block, x):
        for each_block in sequential_block:
            x = checkpoint.checkpoint(
                each_block, x, self.dummy_tensor, use_reentrant=False
            )
        return x

    def forward(self, x):
        x = self.stem(x)

        skip_connections = []

        for stage in self.stages:
            # Run Blocks
            if self.checkpoint_style == "outside_block":
                x = self.iterative_checkpoint(stage["blocks"], x)
            else:
                x = stage["blocks"](x)

            # Save Skip
            skip_connections.append(x)

            # Run Downsample
            if self.checkpoint_style == "outside_block":
                x = checkpoint.checkpoint(stage["down"], x, self.dummy_tensor)
            else:
                x = stage["down"](x)

        # Run Bottleneck
        if self.checkpoint_style == "outside_block":
            x = self.iterative_checkpoint(self.bottleneck, x)
        else:
            x = self.bottleneck(x)

        # Return: [Skip0, Skip1, Skip2, Skip3, Bottleneck]
        return skip_connections + [x]


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int,
        n_classes: int,
        exp_r: list,
        kernel_size: int,
        block_counts: list,
        deep_supervision: bool = False,
        ds_levels: int = 1,
        do_res: bool = False,
        do_res_up_down: bool = False,
        norm_type="group",
        dim="3d",
        use_conv_transpose=False,
        checkpoint_style=None,
    ):
        super().__init__()

        self.dim = dim
        self.do_ds = deep_supervision
        self.ds_levels = ds_levels
        self.checkpoint_style = checkpoint_style
        self.dummy_tensor = (
            nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            if checkpoint_style == "outside_block"
            else None
        )

        up_block_cls = MedNeXtUpBlock if use_conv_transpose else MedNeXtUpBlockV2

        self.stages = nn.ModuleList()

        # Stages 3 -> 0
        multipliers = [16, 8, 4, 2]

        for i, m in enumerate(multipliers):
            stage_idx = 3 - i  # 3, 2, 1, 0
            stage = nn.ModuleDict()

            in_c = m * base_channels
            out_c = (m // 2) * base_channels

            use_grn = True if stage_idx >= 2 else False

            # 1. Upsampling
            stage["up"] = up_block_cls(
                in_channels=in_c,
                out_channels=out_c,
                exp_r=exp_r[i],
                kernel_size=kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=use_grn,
            )

            # 2. Blocks
            stage["blocks"] = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=out_c,
                        out_channels=out_c,
                        exp_r=exp_r[i],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        dim=dim,
                        grn=use_grn,
                    )
                    for _ in range(block_counts[i])
                ]
            )

            # 3. Heads
            if stage_idx == 0:
                stage["head"] = OutBlock(
                    in_channels=out_c, n_classes=n_classes, dim=dim
                )
            elif deep_supervision and ds_levels >= stage_idx:
                stage["head"] = OutBlock(
                    in_channels=out_c, n_classes=n_classes, dim=dim
                )

            self.stages.append(stage)

        # DS Head for Bottleneck
        if deep_supervision and ds_levels == 4:
            self.ds_4 = OutBlock(
                in_channels=16 * base_channels, n_classes=n_classes, dim=dim
            )

    def iterative_checkpoint(self, sequential_block, x):
        for each_block in sequential_block:
            x = checkpoint.checkpoint(
                each_block, x, self.dummy_tensor, use_reentrant=False
            )
        return x

    def forward(self, encoder_features):
        # Pop bottleneck
        x = encoder_features.pop()

        outputs = []

        if self.do_ds and hasattr(self, "ds_4"):
            if self.checkpoint_style == "outside_block":
                out_4 = checkpoint.checkpoint(self.ds_4, x, self.dummy_tensor)
            else:
                out_4 = self.ds_4(x)
            outputs.append(out_4)

        # Iterate stages 3 -> 0
        for i, stage in enumerate(self.stages):
            # 1. Upsample
            if self.checkpoint_style == "outside_block":
                x_up = checkpoint.checkpoint(stage["up"], x, self.dummy_tensor)
            else:
                x_up = stage["up"](x)

            # 2. Skip Connection
            x_res = encoder_features.pop()

            # 3. Combine
            dec_x = x_res + x_up

            # 4. Blocks
            if self.checkpoint_style == "outside_block":
                x = self.iterative_checkpoint(stage["blocks"], dec_x)
            else:
                x = stage["blocks"](dec_x)

            # 5. Head
            if "head" in stage:
                if self.checkpoint_style == "outside_block":
                    out = checkpoint.checkpoint(stage["head"], x, self.dummy_tensor)
                else:
                    out = stage["head"](x)
                outputs.append(out)

        # Return: [Final, DS1, DS2, DS3, DS4]
        return list(reversed(outputs))


class MedNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        ds_levels: int = 1,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: bool = None,
        block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        norm_type="group",
        dim="3d",
        use_conv_transpose=False,
        grn=False,
    ):
        super().__init__()

        self.do_ds = deep_supervision
        self.ds_levels = ds_levels

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        enc_counts = block_counts[:5]
        dec_counts = block_counts[5:]
        enc_exp_r = exp_r[:5]
        dec_exp_r = exp_r[5:]

        # 1. Encoder
        self.encoder = MedNeXtEncoder(
            in_channels=in_channels,
            n_channels=n_channels,
            exp_r=enc_exp_r,
            kernel_size=enc_kernel_size,
            block_counts=enc_counts,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            checkpoint_style=checkpoint_style,
        )

        # 2. Decoder
        self.decoder = MedNeXtDecoder(
            base_channels=n_channels,
            n_classes=n_classes,
            exp_r=dec_exp_r,
            kernel_size=dec_kernel_size,
            block_counts=dec_counts,
            deep_supervision=deep_supervision,
            ds_levels=ds_levels,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            use_conv_transpose=use_conv_transpose,
            checkpoint_style=checkpoint_style,
        )

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)

        if self.do_ds:
            return outputs[: self.ds_levels + 1]
        else:
            return outputs[0]
