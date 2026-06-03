import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UNetBlock3d(nn.Module):
    def __init__(self, cin, cout, norm_type="instance"):
        super().__init__()
        if norm_type == "instance":
            norm_layer = nn.InstanceNorm3d
        else:
            raise Exception(
                f"`norm_type` must be one of [`instance`], got `{norm_type}`"
            )

        self.b = nn.Sequential(
            nn.Conv3d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(cin),
            nn.GELU(),
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.b(x)


class DecoderUpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        # 1. Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )

        # 2. Convolution after upsampling
        self.conv_up = nn.Conv3d(
            cin, cin, kernel_size=3, stride=1, padding=1, bias=True
        )

        # 3. Refinement Block
        self.conv = UNetBlock3d(cin, cout)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_up(x)
        return self.conv(x)


class LightSparKDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_start_dim, out_channels=1):
        """
        Args:
            encoder_channels: List of channel counts from the encoder [stem, s1, s2, s3, bottleneck]
            decoder_start_dim: The dimension to start decoding from.
            out_channels: Number of output channels.
        """
        super().__init__()

        # Reverse encoder channels: Deep -> Shallow
        self.enc_chans_rev = list(reversed(encoder_channels))

        # Calculate decoder channel sequence
        self.dec_chans = [
            decoder_start_dim // (2**i) for i in range(len(self.enc_chans_rev))
        ]

        # Conditional Projections
        # Only create a 1x1 Conv if dimensions mismatch, otherwise Identity
        self.projs = nn.ModuleList()
        for enc_c, dec_c in zip(self.enc_chans_rev, self.dec_chans):
            if enc_c != dec_c:
                self.projs.append(nn.Conv3d(enc_c, dec_c, kernel_size=1, bias=False))
            else:
                self.projs.append(nn.Identity())

        # Decoder Blocks
        self.blocks = nn.ModuleList()
        for i in range(len(self.dec_chans) - 1):
            cin = self.dec_chans[i]
            cout = self.dec_chans[i + 1]
            self.blocks.append(DecoderUpBlock(cin, cout))

        # Final Head
        self.head = nn.Conv3d(
            self.dec_chans[-1], out_channels, kernel_size=1, bias=True
        )

    def forward(self, enc_features):
        """
        Args:
            enc_features: List of (tensor, mask) tuples or tensors from the encoder.
        """
        # Prepare features (Drop masks, Reverse order)
        feats_rev = []
        for f in reversed(enc_features):
            if isinstance(f, (tuple, list)):
                feats_rev.append(f[0])
            else:
                feats_rev.append(f)

        x = 0

        # Iterative Decoding
        for i, block in enumerate(self.blocks):
            enc_f = feats_rev[i]

            # Project (if needed) and Add
            # If Identity, this is a no-op pass-through
            enc_f_proj = self.projs[i](enc_f)
            x = x + enc_f_proj

            # Upsample and Refine
            x = block(x)

        # Final Layer Addition
        last_enc_f = feats_rev[-1]
        last_enc_f_proj = self.projs[-1](last_enc_f)
        x = x + last_enc_f_proj

        return self.head(x)


class SparKPretrainer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_channels: list,
        mask_ratio: float = 0.6,
        patch_size: int = 32,
        mirror_decoder: bool = True,
        decoder_start_dim: int = None,
        out_channels: int = 1,
    ):
        """
        Args:
            encoder: Instance of SparseMedNeXtEncoder.
            encoder_channels: List of channel counts [stem, s1, s2, s3, bottleneck].
                              e.g. [32, 64, 128, 256, 512]
            mask_ratio: Percentage of the image to mask out (0.0 to 1.0).
                        0.6 (60%) is standard for SparK/MAE.
            patch_size: The downsampling factor of the encoder (usually 32 for 5 stages).
                        The mask is generated at Input/patch_size resolution.
            mirror_decoder: If True, uses the reversed encoder_channels for the decoder.
                            If False, constructs a lighter decoder (often better for MAE).
            decoder_start_dim: If mirror_decoder is False, specifies the starting width
                               of the decoder. Defaults to bottleneck width // 2.
            out_channels: Number of output channels (1 for CT/MRI).
        """
        super().__init__()
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # --- Decoder Configuration ---
        if mirror_decoder:
            # Symmetrical Decoder (like a standard U-Net)
            # We assume the decoder starts at the bottleneck dimension
            dec_start = encoder_channels[-1]
        else:
            # Asymmetrical/Light Decoder (Standard for MAE/SparK)
            # Usually half the width or constant width
            dec_start = (
                decoder_start_dim if decoder_start_dim else encoder_channels[-1] // 2
            )

        # Initialize the LightSparKDecoder
        # Note: LightSparKDecoder expects the *encoder's* channel list to determine projections
        self.decoder = LightSparKDecoder(
            encoder_channels=encoder_channels,
            decoder_start_dim=dec_start,
            out_channels=out_channels,
        )

    def generate_sparse_mask(self, x):
        """
        Generates a random binary mask at the patch level and upsamples it.
        1 = Visible (Keep), 0 = Masked (Remove/Predict)
        """
        B, _, D, H, W = x.shape

        # 1. Calculate patch grid size. Use ceil so border voxels are covered
        # when input dimensions are not exact multiples of patch_size.
        d_p = max(1, math.ceil(D / self.patch_size))
        h_p = max(1, math.ceil(H / self.patch_size))
        w_p = max(1, math.ceil(W / self.patch_size))

        # 2. Generate an exact per-sample patch mask.
        # mask: 1 = visible input, 0 = masked target.
        num_patches = d_p * h_p * w_p
        keep_patches = round(num_patches * (1.0 - self.mask_ratio))
        keep_patches = min(num_patches, max(1, keep_patches))

        noise = torch.rand(B, num_patches, device=x.device)
        keep_indices = noise.topk(keep_patches, dim=1).indices
        mask_flat = torch.zeros(B, num_patches, device=x.device, dtype=x.dtype)
        mask_flat.scatter_(1, keep_indices, 1.0)
        mask_patch = mask_flat.view(B, 1, d_p, h_p, w_p)

        # 4. Upsample to original resolution to match SparseConv requirement
        # Nearest neighbor upsampling ensures patches remain blocky.
        mask_full = F.interpolate(mask_patch, size=(D, H, W), mode="nearest")

        return mask_full

    def forward(self, x, region_mask=None):
        """
        Args:
            x: Input image (B, C, D, H, W)
            region_mask: (Optional) Binary mask (B, 1, D, H, W) indicating valid regions
                         for LOSS calculation (e.g., foreground mask).
                         Pixels where region_mask==0 are excluded from the loss,
                         even if they were masked out by the MAE strategy.
        Returns:
            loss: scalar loss value
            pred: reconstructed image
            active_mask: the random mask used (1=visible, 0=masked)
        """
        # 1. Generate Random Mask (The "Active" Mask)
        # 1 = Input to Encoder, 0 = Hidden/Target for Decoder
        active_mask = self.generate_sparse_mask(x)

        # 2. Encoder Forward Pass (Sparse)
        # The encoder only processes regions where active_mask == 1
        # It returns a list of feature maps [skip0, ..., bottleneck]
        enc_features = self.encoder(x, active_mask)

        # 3. Decoder Forward Pass (Dense)
        # The decoder takes the sparse features, fills in the blanks, and reconstructs dense output.
        # LightSparKDecoder expects a list of features.
        pred = self.decoder(enc_features)

        # 4. Calculate Loss
        loss = self.calculate_loss(pred, x, active_mask, region_mask)

        return loss, pred, active_mask

    def calculate_loss(self, pred, target, active_mask, region_mask=None):
        """
        MSE Loss calculated ONLY on:
        1. Masked pixels (active_mask == 0)
        2. Valid regions (region_mask == 1, if provided)
        """
        # Standard MSE (raw)
        loss = (pred - target) ** 2

        # 1. Mask Strategy: We only compute loss on the pixels we removed.
        # active_mask is 1 for kept pixels, 0 for removed pixels.
        # So loss_mask should be 1 where active_mask is 0.
        loss_mask = 1.0 - active_mask

        # 2. Region Constraint (Optional)
        # If we have a foreground mask (e.g., body vs air), we don't care about
        # reconstructing the air.
        if region_mask is not None:
            # Ensure dimensions match
            if region_mask.shape != loss_mask.shape:
                region_mask = F.interpolate(
                    region_mask.float(), size=loss_mask.shape[2:], mode="nearest"
                )

            # Intersection: Must be a masked pixel AND in the valid region
            loss_mask = loss_mask * region_mask

        # 3. Compute Mean Loss over valid pixels
        # Sum of errors / Number of valid pixels
        loss_sum = (loss * loss_mask).sum()
        valid_pixels = loss_mask.expand_as(loss).sum()

        return loss_sum / valid_pixels.clamp_min(1.0)
