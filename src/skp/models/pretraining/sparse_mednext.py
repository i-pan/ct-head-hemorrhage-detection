"""
FROM: https://github.com/MIC-DKFZ/MedNeXt/

Modifications:
- Added default option to use upsampling + normal conv instead of transposed conv
- Changed default to use GRN in bottom 2 encoder/decoder layers and bottleneck
  - Using GRN for all layers increased memory and compute significantly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from skp.models.pretraining.sparse_mednext_blocks import (
    SparseMedNeXtBlock,
    SparseMedNeXtDownBlock,
    SparseMedNeXtUpBlock,
    SparseMedNeXtUpBlockV2,
    SparseOutBlock,
    SparseConv,
    combine_masks,
)


class SparseSequential(nn.Sequential):
    """
    A sequential container that passes both input and mask to each module.
    Assumes that every module in the sequence accepts (x, mask) and returns (x, mask).
    """

    def forward(self, x, mask=None):
        for module in self:
            x, mask = module(x, mask)
        return x, mask


class SparseMedNeXtEncoder(nn.Module):
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
        self.stem = SparseConv(conv(in_channels, n_channels, kernel_size=1))

        # We will store stages in a ModuleList. Each stage contains 'blocks' and 'down'
        self.stages = nn.ModuleList()

        # Configuration for the 4 Encoder levels
        # Channels: [C, 2C, 4C, 8C]
        current_channels = n_channels

        for i in range(4):
            stage = nn.ModuleDict()

            # GRN logic from original: False for first 2 stages, True for rest
            use_grn = True if i >= 2 else False

            # 1. Blocks
            stage["blocks"] = SparseSequential(
                *[
                    SparseMedNeXtBlock(
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

            # 2. Downsampling (Stride 2)
            stage["down"] = SparseMedNeXtDownBlock(
                in_channels=current_channels,
                out_channels=current_channels * 2,
                exp_r=exp_r[i + 1]
                if i < 3
                else exp_r[i],  # Look ahead for exp_r of next stage logic
                kernel_size=kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=use_grn,
            )

            self.stages.append(stage)
            current_channels *= 2

        # Bottleneck (Part of Encoder)
        self.bottleneck = SparseSequential(
            *[
                SparseMedNeXtBlock(
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

    def iterative_checkpoint(self, sequential_block, x, mask):
        for each_block in sequential_block:
            x, mask = checkpoint.checkpoint(
                each_block, x, mask, self.dummy_tensor, use_reentrant=False
            )
        return x, mask

    def forward(self, x, mask=None):
        x, mask = self.stem(x, mask)

        skip_connections = []

        for stage in self.stages:
            # Run Blocks
            if self.checkpoint_style == "outside_block":
                x, mask = self.iterative_checkpoint(stage["blocks"], x, mask)
            else:
                x, mask = stage["blocks"](x, mask)

            # Save Skip Connection (x_res)
            skip_connections.append((x, mask))

            # Run Downsample
            if self.checkpoint_style == "outside_block":
                x, mask = checkpoint.checkpoint(
                    stage["down"], x, mask, self.dummy_tensor, use_reentrant=False
                )
            else:
                x, mask = stage["down"](x, mask)

        # Run Bottleneck
        if self.checkpoint_style == "outside_block":
            x, mask = self.iterative_checkpoint(self.bottleneck, x, mask)
        else:
            x, mask = self.bottleneck(x, mask)

        # Return: [Skip0, Skip1, Skip2, Skip3, Bottleneck]
        return skip_connections + [(x, mask)]


class SparseMedNeXtDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int,  # n_channels of the encoder start
        n_classes: int,
        exp_r: list,  # Expected length: 4 [dec3, dec2, dec1, dec0]
        kernel_size: int,
        block_counts: list,  # Expected length: 4
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

        up_block_cls = (
            SparseMedNeXtUpBlock if use_conv_transpose else SparseMedNeXtUpBlockV2
        )

        self.stages = nn.ModuleList()

        # We build stages from deep (3) to shallow (0)
        # Channels input to decoder stages (coming from bottleneck/up):
        # Bottleneck is 16*C.
        # Stage 3: In 16C -> Up to 8C. Add Skip(8C). Blocks(8C).
        # Stage 2: In 8C -> Up to 4C. Add Skip(4C). Blocks(4C).
        # Stage 1: In 4C -> Up to 2C. Add Skip(2C). Blocks(2C).
        # Stage 0: In 2C -> Up to 1C. Add Skip(1C). Blocks(1C).

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
            stage["blocks"] = SparseSequential(
                *[
                    SparseMedNeXtBlock(
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

            # 3. Deep Supervision Head (Optional)
            # DS is usually applied at resolutions 1, 2, 3, 4 (Bottleneck)
            # Here we are at decoder stage output.
            # Decoder 3 (Res 1/8) -> out_3
            # Decoder 2 (Res 1/4) -> out_2
            # Decoder 1 (Res 1/2) -> out_1
            # Decoder 0 (Res 1/1) -> out_0 (Final)

            if stage_idx == 0:
                stage["head"] = SparseOutBlock(
                    in_channels=out_c, n_classes=n_classes, dim=dim
                )
            elif deep_supervision and ds_levels >= stage_idx:
                stage["head"] = SparseOutBlock(
                    in_channels=out_c, n_classes=n_classes, dim=dim
                )

            self.stages.append(stage)

        # Deep Supervision head for Bottleneck (Level 4)
        if deep_supervision and ds_levels == 4:
            self.ds_4 = SparseOutBlock(
                in_channels=16 * base_channels, n_classes=n_classes, dim=dim
            )

    def iterative_checkpoint(self, sequential_block, x, mask):
        for each_block in sequential_block:
            x, mask = checkpoint.checkpoint(
                each_block, x, mask, self.dummy_tensor, use_reentrant=False
            )
        return x, mask

    def forward(self, encoder_features):
        # encoder_features: [Skip0, Skip1, Skip2, Skip3, Bottleneck]

        # Pop bottleneck
        x, mask = encoder_features.pop()

        outputs = []

        # Handle DS for Bottleneck (if enabled)
        if self.do_ds and hasattr(self, "ds_4"):
            if self.checkpoint_style == "outside_block":
                out_4, _ = checkpoint.checkpoint(
                    self.ds_4, x, mask, self.dummy_tensor, use_reentrant=False
                )
            else:
                out_4, _ = self.ds_4(x, mask)
            outputs.append(out_4)

        # Iterate through stages (3 -> 0)
        for i, stage in enumerate(self.stages):
            # 1. Upsample
            if self.checkpoint_style == "outside_block":
                x_up, mask_up = checkpoint.checkpoint(
                    stage["up"], x, mask, self.dummy_tensor, use_reentrant=False
                )
            else:
                x_up, mask_up = stage["up"](x, mask)

            # 2. Get Skip Connection
            x_res, mask_res = encoder_features.pop()
            if x_res.shape[2:] != x_up.shape[2:]:
                x_res = F.interpolate(
                    x_res,
                    size=x_up.shape[2:],
                    mode="trilinear" if self.dim == "3d" else "bilinear",
                    align_corners=False,
                )
            fused_mask = combine_masks(mask_up, mask_res)

            # 3. Combine
            dec_x = x_res + x_up
            if fused_mask is not None:
                dec_x = dec_x * fused_mask

            # 4. Blocks
            if self.checkpoint_style == "outside_block":
                x, mask = self.iterative_checkpoint(stage["blocks"], dec_x, fused_mask)
            else:
                x, mask = stage["blocks"](dec_x, fused_mask)

            # 5. Output Head (Deep Supervision or Final)
            if "head" in stage:
                if self.checkpoint_style == "outside_block":
                    out, _ = checkpoint.checkpoint(
                        stage["head"],
                        x,
                        mask,
                        self.dummy_tensor,
                        use_reentrant=False,
                    )
                else:
                    out, _ = stage["head"](x, mask)
                outputs.append(out)

        # Outputs are collected in order: [DS4, DS3, DS2, DS1, Final] (depending on config)
        # Standard return expectation is reversed: Final, DS1, DS2... or just Final.
        # The logic in the original code returned: [Final, DS1, DS2, DS3, DS4]

        # Let's sort them to match original signature based on resolution
        # outputs currently contains [DS4(optional), DS3(opt), DS2(opt), DS1(opt), Final]

        # Reversing list gives: [Final, DS1, DS2, DS3, DS4]
        return list(reversed(outputs))


class SparseMedNeXt(nn.Module):
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
        grn=False,  # Unused argument in original signature, but kept for compatibility
    ):
        super().__init__()

        self.do_ds = deep_supervision
        self.ds_levels = ds_levels

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        # Split Configs
        enc_counts = block_counts[:5]  # 0,1,2,3, Bottleneck
        dec_counts = block_counts[5:]  # 3,2,1,0

        enc_exp_r = exp_r[:5]
        dec_exp_r = exp_r[5:]

        # 1. Encoder (includes Bottleneck)
        self.encoder = SparseMedNeXtEncoder(
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

        # 2. Decoder (Task Specific)
        self.decoder = SparseMedNeXtDecoder(
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

    def forward(self, x, mask=None):
        # 1. Encoder Pass
        # Returns list: [Skip0, Skip1, Skip2, Skip3, Bottleneck]
        enc_features = self.encoder(x, mask)

        # 2. Decoder Pass
        # Returns list: [Final, DS1, DS2, DS3, DS4]
        outputs = self.decoder(enc_features)

        # 3. Format Return
        if self.do_ds:
            # Filter based on ds_levels requested
            # output list is already [Final, DS1, DS2, DS3, DS4]
            return outputs[: self.ds_levels + 1]
        else:
            return outputs[0]
