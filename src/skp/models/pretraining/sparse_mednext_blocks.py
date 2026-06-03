import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Helper: Mask Resizing
# -------------------------------------------------------------------------


def resize_mask(mask, size, mode="nearest"):
    """Resizes mask to match tensor spatial dimensions."""
    if mask is None:
        return None
    if mask.shape[2:] == size:
        return mask
    return F.interpolate(mask.float(), size=size, mode=mode)


def combine_masks(*masks):
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        return None

    target_size = masks[0].shape[2:]
    resized = [
        resize_mask(mask, target_size) if mask.shape[2:] != target_size else mask
        for mask in masks
    ]
    out = resized[0].float()
    for mask in resized[1:]:
        out = torch.maximum(out, mask.float())
    return out


def _as_tuple(value, ndim):
    if isinstance(value, tuple):
        return value
    return (value,) * ndim


def propagate_mask(mask, op, output_size):
    if mask is None:
        return None

    spatial_ndim = mask.ndim - 2
    mask = mask.float()

    if isinstance(op, (nn.Conv2d, nn.Conv3d)):
        stride = _as_tuple(op.stride, spatial_ndim)
        if all(each == 1 for each in stride) and mask.shape[2:] == output_size:
            out = mask
        else:
            pool = F.max_pool3d if spatial_ndim == 3 else F.max_pool2d
            kernel_size = _as_tuple(op.kernel_size, spatial_ndim)
            padding = _as_tuple(op.padding, spatial_ndim)
            dilation = _as_tuple(op.dilation, spatial_ndim)
            out = pool(
                mask,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
    elif isinstance(op, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        conv_transpose = F.conv_transpose3d if spatial_ndim == 3 else F.conv_transpose2d
        kernel_size = _as_tuple(op.kernel_size, spatial_ndim)
        weight_shape = (1, 1, *kernel_size)
        weight = mask.new_ones(weight_shape)
        out = conv_transpose(
            mask,
            weight,
            bias=None,
            stride=op.stride,
            padding=op.padding,
            output_padding=op.output_padding,
            dilation=op.dilation,
        )
        out = (out > 0).to(mask.dtype)
    elif mask.shape[2:] != output_size:
        out = resize_mask(mask, output_size)
    else:
        out = mask

    if out.shape[2:] != output_size:
        out = resize_mask(out, output_size)
    return (out > 0).to(mask.dtype)


# -------------------------------------------------------------------------
# Sparse Normalization Layers
# -------------------------------------------------------------------------
class SparseInstanceNorm(nn.Module):
    def __init__(self, num_features, dim="3d", eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x, mask=None):
        if mask is None:
            return F.instance_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=self.weight,
                bias=self.bias,
                use_input_stats=True,
                momentum=0.0,
                eps=self.eps,
            )

        if mask.shape[2:] != x.shape[2:]:
            mask = resize_mask(mask, x.shape[2:])

        # 1. Zero out background
        x = x * mask

        # 2. Compute Stats (N, C, 1, 1, 1)
        # Sum over spatial dims (2, 3, 4)
        reduce_dims = list(range(2, x.ndim))

        valid_count = mask.sum(dim=reduce_dims, keepdim=True)
        valid_count = torch.clamp(valid_count, min=1.0)

        sum_x = x.sum(dim=reduce_dims, keepdim=True)
        mean = sum_x / valid_count

        sum_x2 = (x**2).sum(dim=reduce_dims, keepdim=True)
        mean_x2 = sum_x2 / valid_count
        var = mean_x2 - mean**2
        var = torch.clamp(var, min=0.0)

        std = torch.sqrt(var + self.eps)

        # 3. Normalize
        # (x - mean) / std
        x_norm = (x - mean) / std

        # 4. Affine
        if self.affine:
            view_shape = [1, self.num_features] + [1] * (x.ndim - 2)
            x_norm = x_norm * self.weight.view(*view_shape) + self.bias.view(
                *view_shape
            )

        # 5. Re-apply mask (Bias might have added value to background)
        return x_norm * mask


class SparseGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, dim="3d", eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x, mask=None):
        if mask is None:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        if mask.shape[2:] != x.shape[2:]:
            mask = resize_mask(mask, x.shape[2:])

        # 1. Reshape for Groups: (N, G, C//G, D, H, W)
        N, C = x.shape[:2]
        x_g = x.view(N, self.num_groups, -1, *x.shape[2:])
        mask_g = mask.expand(N, C, *x.shape[2:]).reshape(
            N, self.num_groups, -1, *x.shape[2:]
        )

        # 2. Zero out background
        x_g = x_g * mask_g

        # 3. Compute Stats
        reduce_dims = list(range(2, x_g.ndim))

        valid_count = mask_g.sum(dim=reduce_dims, keepdim=True)
        valid_count = torch.clamp(valid_count, min=1.0)

        sum_x = x_g.sum(dim=reduce_dims, keepdim=True)
        mean = sum_x / valid_count

        sum_x2 = (x_g**2).sum(dim=reduce_dims, keepdim=True)
        mean_x2 = sum_x2 / valid_count
        var = mean_x2 - mean**2
        var = torch.clamp(var, min=0.0)

        std = torch.sqrt(var + self.eps)

        # 4. Normalize
        x_norm_g = (x_g - mean) / std

        # 5. Reshape Back
        x_norm = x_norm_g.view(N, C, *x.shape[2:])

        # 6. Affine
        if self.affine:
            view_shape = [1, C] + [1] * (x.ndim - 2)
            x_norm = x_norm * self.weight.view(*view_shape) + self.bias.view(
                *view_shape
            )

        return x_norm * mask


class SparseBatchNorm(nn.Module):
    def __init__(self, num_features, dim="3d", eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Use nn.Parameter for weights and biases
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Use register_buffer for non-parameter state
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x, mask=None):
        if mask is None:
            # Use standard F.batch_norm if no mask is provided
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        # Ensure mask shape matches feature shape for broadcasting
        if mask.shape[2:] != x.shape[2:]:
            mask = resize_mask(mask, x.shape[2:])

        # Apply mask before calculations
        x = x * mask

        if self.training:
            # Flatten to (C, -1) to get global stats over batch and spatial dims
            # Permute: (N, C, D, H, W) -> (C, N, D, H, W) -> (C, -1)
            x_flat = x.permute(1, 0, *range(2, x.ndim)).flatten(1)
            mask_flat = mask.expand_as(x).permute(1, 0, *range(2, x.ndim)).flatten(1)

            # Sum over valid elements
            valid_count = mask_flat.sum(dim=1)
            # Prevent division by zero if a channel has no valid elements
            valid_count = torch.clamp(valid_count, min=1.0)

            # Calculate mean and variance over valid elements only
            mean = x_flat.sum(dim=1) / valid_count
            mean_x2 = (x_flat**2).sum(dim=1) / valid_count
            var = mean_x2 - mean.pow(2)
            var = torch.clamp(var, min=0.0)  # Ensure variance is non-negative

            correction_factor = torch.where(
                valid_count > 1,
                valid_count / (valid_count - 1.0),
                torch.ones_like(valid_count),
            )
            unbiased_var = var * correction_factor

            # Update running stats using momentum
            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean + self.momentum * mean
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbiased_var
                )
        else:
            # Use running stats during evaluation
            mean = self.running_mean
            var = self.running_var

        # Apply batch normalization manually
        # Reshape stats to (1, C, 1, 1, ...) for broadcasting
        view_shape = [1, self.num_features] + [1] * (x.ndim - 2)
        mean = mean.view(*view_shape)
        var = var.view(*view_shape)
        weight = self.weight.view(*view_shape)
        bias = self.bias.view(*view_shape)

        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        x_out = x_norm * weight + bias

        # Re-apply the mask to ensure output is zero where input was zero
        return x_out * mask


class MaskedChannelLayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for NCHW/NCDHW tensors.

    This norm is already local to each spatial position, so active pixels do not
    share statistics with masked pixels. The mask is only used to keep inactive
    positions zero after the affine transform.
    """

    def __init__(self, num_channels, dim="3d", eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x, mask=None):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            view_shape = [1, self.num_channels] + [1] * (x.ndim - 2)
            x = x * self.weight.view(*view_shape) + self.bias.view(*view_shape)

        if mask is not None:
            if mask.shape[2:] != x.shape[2:]:
                mask = resize_mask(mask, x.shape[2:])
            x = x * mask
        return x


# -------------------------------------------------------------------------
# Sparse Convolution Wrapper
# -------------------------------------------------------------------------


class SparseConv(nn.Module):
    """
    Wraps a standard convolution.
    1. Multiplies input by mask.
    2. Convolves.
    3. Propagates the mask through the convolution geometry.
    4. Multiplies output by new mask.
    """

    def __init__(self, conv_op):
        super().__init__()
        self.conv = conv_op

    def forward(self, x, mask):
        # 1. Zero out invalid input
        if mask is not None:
            if mask.shape[2:] != x.shape[2:]:
                mask = resize_mask(mask, x.shape[2:])
            x = x * mask

        # 2. Perform Conv
        x = self.conv(x)

        # 3. Handle Mask Propagation (Stride/Padding)
        if mask is not None:
            mask = propagate_mask(mask, self.conv, x.shape[2:])

            # 4. Zero out invalid output (biases might have activated background)
            x = x * mask

        return x, mask


# -------------------------------------------------------------------------
# Modified MedNeXt Blocks
# -------------------------------------------------------------------------


class SparseMedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: int = True,
        norm_type: str = "group",
        n_groups: int or None = None,
        dim="3d",
        grn=False,
    ):
        super().__init__()

        self.do_res = do_res
        self.dim = dim
        self.grn_enabled = grn

        if self.dim == "2d":
            conv_cls = nn.Conv2d
        elif self.dim == "3d":
            conv_cls = nn.Conv3d

        # 1. Depthwise Conv
        # Wrapped in SparseConv to handle masking
        self.conv1 = SparseConv(
            conv_cls(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels if n_groups is None else n_groups,
            )
        )

        # 2. Normalization
        if norm_type == "group":
            self.norm = SparseGroupNorm(
                num_groups=in_channels, num_channels=in_channels, dim=dim
            )
        elif norm_type == "batch":
            self.norm = SparseBatchNorm(num_features=in_channels, dim=dim)
        elif norm_type == "instance":
            self.norm = SparseInstanceNorm(num_features=in_channels, dim=dim)
        elif norm_type == "layer":
            self.norm = MaskedChannelLayerNorm(num_channels=in_channels, dim=dim)
        else:
            raise ValueError(f"Unknown sparse norm type: {norm_type}")

        # 3. Expansion Conv (1x1)
        self.conv2 = SparseConv(
            conv_cls(
                in_channels=in_channels,
                out_channels=exp_r * in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.act = nn.GELU()

        # 4. Compression Conv (1x1)
        self.conv3 = SparseConv(
            conv_cls(
                in_channels=exp_r * in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        # 5. GRN
        if grn:
            # Parameters remain the same
            shape = (1, exp_r * in_channels) + (1,) * (3 if dim == "3d" else 2)
            self.grn_beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, x, mask=None, dummy_tensor=None):
        residual = x

        # 1. Depthwise
        x1, mask = self.conv1(x, mask)

        # 2. Norm
        x1 = self.norm(x1, mask)

        # 3. Expansion
        x1, mask = self.conv2(x1, mask)
        x1 = self.act(x1)

        # 4. GRN
        if self.grn_enabled:
            # Ensure input is masked before norm calc
            if mask is not None:
                x1 = x1 * mask

            # Global Response Norm:
            # Compute L2 norm across spatial dimensions
            if self.dim == "3d":
                spatial_dims = (-3, -2, -1)
            else:
                spatial_dims = (-2, -1)

            # Since background is 0, standard L2 norm works for "non-masked pixels"
            # as adding 0^2 doesn't change the sum.
            gx = torch.norm(x1, p=2, dim=spatial_dims, keepdim=True)

            # Normalize the responses (channel-wise)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

            # Apply affine + residual
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

            # Re-mask to be safe
            if mask is not None:
                x1 = x1 * mask

        # 5. Compression
        x1, mask = self.conv3(x1, mask)

        if self.do_res:
            x1 = residual + x1
            if mask is not None:
                x1 = x1 * mask

        return x1, mask


class SparseMedNeXtDownBlock(SparseMedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        if dim == "2d":
            conv_cls = nn.Conv2d
        elif dim == "3d":
            conv_cls = nn.Conv3d

        self.resample_do_res = do_res

        if do_res:
            self.res_conv = SparseConv(
                conv_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                )
            )

        # Overwrite conv1 with stride=2 for downsampling
        self.conv1 = SparseConv(
            conv_cls(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=in_channels,
            )
        )

    def forward(self, x, mask=None, dummy_tensor=None):
        # We need the original x and mask for the residual connection
        x_in, mask_in = x, mask

        # Run Block (Strided Conv1 happens inside here)
        x1, mask_out = super().forward(x, mask)

        if self.resample_do_res:
            # Resample residual path
            res, _ = self.res_conv(x_in, mask_in)
            x1 = x1 + res
            if mask_out is not None:
                x1 = x1 * mask_out

        return x1, mask_out


class SparseMedNeXtUpBlock(SparseMedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,  # We handle residual manually in this specific block
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.resample_do_res = do_res
        self.dim = dim

        if dim == "2d":
            conv_cls = nn.ConvTranspose2d
        elif dim == "3d":
            conv_cls = nn.ConvTranspose3d

        # 1. Residual Convolution (Upsampling)
        if do_res:
            self.res_conv = SparseConv(
                conv_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                )
            )

        # 2. Main Convolution (Upsampling)
        # We overwrite self.conv1 from the parent class to be a Transpose Conv
        self.conv1 = SparseConv(
            conv_cls(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=in_channels,
            )
        )

    def forward(self, x, mask=None, dummy_tensor=None):
        # Keep reference to input for residual connection
        x_in, mask_in = x, mask

        # 1. Run Super Block
        # (conv1 upsamples x and mask automatically via SparseConv logic)
        x1, mask_out = super().forward(x, mask)

        # 2. Apply Asymmetry Padding
        # Matches original logic: F.pad(x, (1, 0, ...))
        if self.dim == "2d":
            pad = (1, 0, 1, 0)
        elif self.dim == "3d":
            pad = (1, 0, 1, 0, 1, 0)

        x1 = F.pad(x1, pad)

        if mask_out is not None:
            # We must also pad the mask so it stays aligned with x1
            mask_out = F.pad(mask_out.float(), pad, mode="constant", value=0)

        # 3. Residual Connection
        if self.resample_do_res:
            # SparseConv handles upsampling + mask resizing for the residual path
            res, _ = self.res_conv(x_in, mask_in)

            # Apply same padding to residual
            res = F.pad(res, pad)

            x1 = x1 + res
            if mask_out is not None:
                x1 = x1 * mask_out

        return x1, mask_out


class SparseMedNeXtUpBlockV2(SparseMedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        # Initialize parent
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.resample_do_res = do_res
        self.dim = dim

        if dim == "2d":
            self.conv_cls = nn.Conv2d
            self.mode = "bilinear"
        elif dim == "3d":
            self.conv_cls = nn.Conv3d
            self.mode = "trilinear"

        if do_res:
            # 1x1 conv, stride 1 (we upsample manually before)
            self.res_conv = SparseConv(
                self.conv_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                )
            )

    def forward(self, x, mask=None, dummy_tensor=None):
        # 1. Explicit Upsampling of Data and Mask
        x_up = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)

        mask_up = None
        if mask is not None:
            # Nearest neighbor for mask to keep it binary
            mask_up = F.interpolate(mask.float(), scale_factor=2, mode="nearest")
            x_up = x_up * mask_up  # Clean edges

        # 2. Apply Sparse MedNeXtBlock
        x1, mask_out = super().forward(x_up, mask_up)

        # 3. Residual connection
        if self.resample_do_res:
            res, _ = self.res_conv(x_up, mask_up)
            x1 = x1 + res
            if mask_out is not None:
                x1 = x1 * mask_out

        return x1, mask_out


class SparseOutBlock(nn.Module):
    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == "2d":
            conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        elif dim == "3d":
            conv = nn.Conv3d(in_channels, n_classes, kernel_size=1)

        self.conv_out = SparseConv(conv)

    def forward(self, x, mask=None, dummy_tensor=None):
        out, mask = self.conv_out(x, mask)
        return out, mask
