import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from functools import partial
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import swin3d_t, swin3d_s, swin3d_b
from torchvision.models.video import (
    Swin3D_T_Weights,
    Swin3D_S_Weights,
    Swin3D_B_Weights,
)
from torchvision.models.video import mvit_v2_s
from torchvision.models.video import (
    MViT_V2_S_Weights,
)

from skp.models.utils import change_num_input_channels

"""
X3D
"""


class GroupNorm(nn.GroupNorm):
    def __init__(
        self,
        num_features: int,
        num_groups: int = 2,
        eps: float = 1e-05,
        affine: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_features,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )


class X3DEncoder(nn.Module):
    def __init__(self, cfg, include_last_block=False):
        super().__init__()
        self.cfg = cfg

        if self.cfg.backbone.endswith("_gn"):
            # format will be x3d_l_gn
            # use GroupNorm
            assert not self.cfg.pretrained, "pretrained not supported for GroupNorm"

        # Set up X3D backbone
        if self.cfg.pretrained:
            net = torch.hub.load(
                "facebookresearch/pytorchvideo",
                model=self.cfg.backbone,
                pretrained=True,
            )
        else:
            from pytorchvideo.models import hub

            kwargs_dict = {"pretrained": False}
            if self.cfg.backbone.endswith("_gn"):
                norm = GroupNorm
                kwargs_dict["norm"] = norm
                kwargs_dict["activation"] = nn.SiLU

            net = getattr(hub, self.cfg.backbone.replace("_gn", ""))(**kwargs_dict)

        self.features = net.blocks[:-1]

        if self.cfg.include_last_block:
            # only used for classification, not segmentation
            self.features.append(
                nn.Sequential(
                    net.blocks[-1].pool.pre_conv,
                    net.blocks[-1].pool.pre_norm,
                    net.blocks[-1].pool.pre_act,
                )
            )

        for idx in range(len(self.features[:5])):
            if idx == 0 and self.cfg.dim0_strides[idx] == 2:
                stem_layer = self.features[0].conv.conv_t
                stem_weights = stem_layer.state_dict()
                stem_weights["weight"] = stem_weights["weight"].repeat(1, 1, 3, 1, 1)
                in_channels, out_channels = (
                    stem_layer.in_channels,
                    stem_layer.out_channels,
                )
                self.features[0].conv.conv_t = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3, 3),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                    bias=False,
                )
                self.features[0].conv.conv_t.load_state_dict(stem_weights)
            elif idx > 0:
                self.features[idx].res_blocks[0].branch1_conv.stride = (
                    self.cfg.dim0_strides[idx],
                    2,
                    2,
                )
                self.features[idx].res_blocks[0].branch2.conv_b.stride = (
                    self.cfg.dim0_strides[idx],
                    2,
                    2,
                )

        if self.cfg.output_stride in {2, 4, 8, 16}:
            assert self.cfg.dim0_strides[-1] == 1
            self.features[-1].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-1].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

        if self.cfg.output_stride in {2, 4, 8}:
            assert self.cfg.dim0_strides[-2] == 1
            self.features[-2].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-2].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

        if self.cfg.output_stride in {2, 4}:
            assert self.cfg.dim0_strides[-3] == 1
            self.features[-3].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-3].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)
        
        if self.cfg.output_stride in {2}:
            assert self.cfg.dim0_strides[-4] == 1
            self.features[-4].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-4].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

    def forward(self, x):
        feature_maps = []
        for block in self.features:
            if self.cfg.enable_gradient_checkpointing:
                x = checkpoint(block, *(x,), use_reentrant=False)
            else:
                x = block(x)
            feature_maps.append(x)
        return feature_maps


"""
Channel-Separated Network
"""


class CSNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if self.cfg.pretrained:
            net = torch.hub.load(
                "facebookresearch/pytorchvideo", model="csn_r101", pretrained=True
            )
        else:
            from pytorchvideo.models import hub

            net = hub.csn_r101(pretrained=False)

        self.features = net.blocks[:5]

        if self.cfg.backbone == "csn_r50":
            self.features[3].res_blocks = self.features[3].res_blocks[:6]
            # [3, 4, 23, 3] -> [3, 4, 6, 3]
        elif self.cfg.backbone == "csn_r26":
            for i in [1, 2, 3, 4]:
                self.features[i].res_blocks = self.features[i].res_blocks[:2]
            # [3, 4, 23, 3] -> [2, 2, 2, 2]

        self.features[0].conv.stride = (1, 1, 1)
        self.features[1].res_blocks[0].branch1_conv.stride = (1, 2, 2)
        self.features[1].res_blocks[0].branch2.conv_b.stride = (1, 2, 2)

        if self.cfg.dim0_strides[0] == 2:
            self.features[0].pool.kernel_size = (3, 3, 3)
            self.features[0].pool.stride = (2, 2, 2)
            self.features[0].pool.padding = (1, 1, 1)

        if self.cfg.dim0_strides[1] == 2:
            self.features[1].res_blocks[0].branch1_conv.stride = (2, 2, 2)
            self.features[1].res_blocks[0].branch2.conv_b.stride = (2, 2, 2)

        if self.cfg.dim0_strides[2] == 1:
            self.features[2].res_blocks[0].branch1_conv.stride = (1, 2, 2)
            self.features[2].res_blocks[0].branch2.conv_b.stride = (1, 2, 2)

        if self.cfg.dim0_strides[3] == 1:
            self.features[3].res_blocks[0].branch1_conv.stride = (1, 2, 2)
            self.features[3].res_blocks[0].branch2.conv_b.stride = (1, 2, 2)

        if self.cfg.dim0_strides[4] == 1:
            self.features[4].res_blocks[0].branch1_conv.stride = (1, 2, 2)
            self.features[4].res_blocks[0].branch2.conv_b.stride = (1, 2, 2)

        if self.cfg.output_stride in {2, 4, 8, 16}:
            assert self.cfg.dim0_strides[-1] == 1
            self.features[-1].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-1].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

        if self.cfg.output_stride in {2, 4, 8}:
            assert self.cfg.dim0_strides[-2] == 1
            self.features[-2].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-2].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

        if self.cfg.output_stride in {2, 4}:
            assert self.cfg.dim0_strides[-3] == 1
            self.features[-3].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-3].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

        if self.cfg.output_stride in {2}:
            assert self.cfg.dim0_strides[-4] == 1
            self.features[-4].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            self.features[-4].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)

    def forward(self, x):
        feature_maps = []
        for block in self.features:
            if self.cfg.enable_gradient_checkpointing:
                x = checkpoint(block, *(x,), use_reentrant=False)
            else:
                x = block(x)
            feature_maps.append(x)
        return feature_maps


"""
3D SwinTransformer
"""


class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size,
        in_channels=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()

        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    @staticmethod
    def _compute_pad_size_3d(size_dhw, patch_size):
        pad_size = [
            (patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i]
            for i in range(3)
        ]
        return pad_size[0], pad_size[1], pad_size[2]

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = self._compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)
        x = rearrange(x, "b c t h w -> b t h w c").contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
        T, H, W, _ = x.shape[-4:]
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))
        x0 = x[..., 0::2, 0::2, 0::2, :]  # ... T/2 H/2 W/2 C
        x1 = x[..., 0::2, 1::2, 0::2, :]  # ... T/2 H/2 W/2 C
        x2 = x[..., 0::2, 0::2, 1::2, :]  # ... T/2 H/2 W/2 C
        x3 = x[..., 0::2, 1::2, 1::2, :]  # ... T/2 H/2 W/2 C
        x4 = x[..., 1::2, 0::2, 0::2, :]  # ... T/2 H/2 W/2 C
        x5 = x[..., 1::2, 1::2, 0::2, :]  # ... T/2 H/2 W/2 C
        x6 = x[..., 1::2, 0::2, 1::2, :]  # ... T/2 H/2 W/2 C
        x7 = x[..., 1::2, 1::2, 1::2, :]  # ... T/2 H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # ... T/2 H/2 W/2 8*C
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... T/2 H/2 W/2 2*C
        return x


class PatchForward3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Swin3DEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        assert self.cfg.backbone in {
            "swin3d_t",
            "swin3d_s",
            "swin3d_b",
        }, f"{self.cfg.backbone} is not a valid encoder for this model"
        if self.cfg.backbone == "swin3d_t":
            net = swin3d_t(
                weights=Swin3D_T_Weights.DEFAULT if self.cfg.pretrained else None
            )
        elif self.cfg.backbone == "swin3d_s":
            net = swin3d_s(
                weights=Swin3D_S_Weights.DEFAULT if self.cfg.pretrained else None
            )
        elif self.cfg.backbone == "swin3d_b":
            net = swin3d_b(
                weights=Swin3D_B_Weights.DEFAULT if self.cfg.pretrained else None
            )

        self.patch_embed = net.patch_embed
        self.patch_embed.proj.stride = self.cfg.patch_embed_stride or (2, 4, 4)
        assert self.patch_embed.proj.stride[0] == self.cfg.dim0_strides[0]
        if self.patch_embed.proj.stride != (2, 4, 4):
            print("Changing patch embedding layer ...")
            assert self.patch_embed.proj.stride[0] in {1, 2, 4}
            assert self.patch_embed.proj.stride[1] in {2, 4}
            assert self.patch_embed.proj.stride[2] in {2, 4}

            patch_embed_weights = self.patch_embed.state_dict()
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
            self.patch_embed = PatchEmbed3d(
                patch_size=self.cfg.patch_embed_stride,
                in_channels=self.patch_embed.proj.in_channels,
                embed_dim=self.patch_embed.proj.out_channels,
                norm_layer=norm_layer,
            )

            if self.patch_embed.proj.stride[0] == 1:
                patch_embed_weights["proj.weight"] = patch_embed_weights[
                    "proj.weight"
                ].mean(2, keepdims=True)
            elif self.patch_embed.proj.stride[0] == 4:
                patch_embed_weights["proj.weight"] = patch_embed_weights[
                    "proj.weight"
                ].repeat(1, 1, 2, 1, 1)

            if self.patch_embed.proj.stride[1] == 2:
                patch_embed_weights["proj.weight"] = reduce(
                    patch_embed_weights["proj.weight"],
                    "c1 c0 x (y1 y2) z -> c1 c0 x y1 z",
                    "mean",
                    y1=2,
                    y2=2,
                )

            if self.patch_embed.proj.stride[2] == 2:
                patch_embed_weights["proj.weight"] = reduce(
                    patch_embed_weights["proj.weight"],
                    "c1 c0 x y (z1 z2) -> c1 c0 x y z1",
                    "mean",
                    z1=2,
                    z2=2,
                )

            self.patch_embed.load_state_dict(patch_embed_weights)

        self.pos_drop = net.pos_drop

        self.features = net.features
        for idx, block_idx in enumerate([1, 3, 5]):
            if self.cfg.output_stride in {8, 16} and block_idx == 5:
                assert self.cfg.dim0_strides[-1] == 1
                dim = self.features[block_idx].reduction.in_features
                reduction_wts = self.features[block_idx].reduction.state_dict()
                reduction_wts["weight"] = reduce(
                    reduction_wts["weight"], "c1 (m c0) -> c1 c0", "mean", m=4
                )
                norm_wts = self.features[block_idx].norm.state_dict()
                norm_wts["weight"] = reduce(
                    norm_wts["weight"], "(m c0) -> c0", "mean", m=4
                )
                norm_wts["bias"] = reduce(norm_wts["bias"], "(m c0) -> c0", "mean", m=4)
                self.features[block_idx] = PatchForward3d(dim // 4)
                self.features[block_idx].reduction.load_state_dict(reduction_wts)
                self.features[block_idx].norm.load_state_dict(norm_wts)
            elif self.cfg.output_stride == 8 and block_idx == 3:
                assert self.cfg.dim0_strides[-2] == 1
                dim = self.features[block_idx].reduction.in_features
                reduction_wts = self.features[block_idx].reduction.state_dict()
                reduction_wts["weight"] = reduce(
                    reduction_wts["weight"], "c1 (m c0) -> c1 c0", "mean", m=4
                )
                norm_wts = self.features[block_idx].norm.state_dict()
                norm_wts["weight"] = reduce(
                    norm_wts["weight"], "(m c0) -> c0", "mean", m=4
                )
                norm_wts["bias"] = reduce(norm_wts["bias"], "(m c0) -> c0", "mean", m=4)
                self.features[block_idx] = PatchForward3d(dim // 4)
                self.features[block_idx].reduction.load_state_dict(reduction_wts)
                self.features[block_idx].norm.load_state_dict(norm_wts)
            elif self.cfg.dim0_strides[idx + 1] == 2:
                dim = self.features[block_idx].reduction.in_features
                reduction_wts = self.features[block_idx].reduction.state_dict()
                reduction_wts["weight"] = reduction_wts["weight"].repeat(1, 2)
                norm_wts = self.features[block_idx].norm.state_dict()
                norm_wts["weight"] = norm_wts["weight"].repeat(2)
                norm_wts["bias"] = norm_wts["bias"].repeat(2)
                self.features[block_idx] = PatchMerging3d(dim // 4)
                self.features[block_idx].reduction.load_state_dict(reduction_wts)
                self.features[block_idx].norm.load_state_dict(norm_wts)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        feature_maps = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in {0, 2, 4, 6}:
                feature_maps.append(rearrange(x, "b x y z c -> b c x y z").contiguous())
        return feature_maps


"""
3D MViT V2
"""


class MViTV2Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert self.cfg.backbone in ["mvit_v2_t", "mvit_v2_s"]
        assert len(self.cfg.dim0_strides) == 4

        net = mvit_v2_s(
            weights=MViT_V2_S_Weights.DEFAULT if self.cfg.pretrained else None
        )

        # Modify patch embedding
        patch_embed_stride = self.cfg.patch_embed_stride or (2, 4, 4)
        assert patch_embed_stride[0] == self.cfg.dim0_strides[0]
        net.conv_proj.stride = patch_embed_stride

        # Modify positional encoding attributes with specified input size
        d = self.cfg.get("num_slices", self.cfg.get("dim0"))
        h = self.cfg.get("image_height", self.cfg.get("dim1"))
        w = self.cfg.get("image_width", self.cfg.get("dim2"))
        input_size = [
            size // stride for size, stride in zip((d,) + (h, w), net.conv_proj.stride)
        ]
        net.pos_encoding.temporal_size = input_size[0]
        net.pos_encoding.spatial_size = (input_size[1], input_size[2])

        idx = 2
        for name, module in net.named_modules():
            if name.endswith("_q.pool") or name.endswith("_skip.pool"):
                if torch.prod(torch.tensor(module.stride)).item() > 1:
                    stride = module.stride
                    module.stride = (
                        self.cfg.dim0_strides[idx // 2],
                        stride[1],
                        stride[2],
                    )
                    idx += 1
            if self.cfg.increase_kv_strides is not None:
                # Default pool_k/v strides are equal to (1, X, X)
                # This increases the first dimension stride to increase_kv_strides * X
                assert 0 < self.cfg.increase_kv_strides <= 1
                if name.endswith("_k.pool") or name.endswith("_v.pool"):
                    stride = module.stride
                    if stride[1] != 1:
                        module.stride = (
                            round(self.cfg.increase_kv_strides * stride[1]),
                            stride[1],
                            stride[2],
                        )

        if self.cfg.backbone == "mvit_v2_t":
            # Tiny variant uses a 1-2-5-2 block structure vs small 1-2-11-2
            net.blocks = net.blocks[:8] + [net.blocks[-2]] + [net.blocks[-1]]

        self.conv_proj = net.conv_proj
        self.pos_encoding = net.pos_encoding
        self.features = net.blocks

        # self.norm = net.norm

    @staticmethod
    def reshape_tokens_to_feature_maps(tokens, thw):
        tokens = tokens[:, 1:]
        b, n, c = tokens.shape
        tokens = tokens.transpose(2, 1)
        return tokens.reshape((b, c) + thw).contiguous()

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)

        feature_maps = []
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for idx, block in enumerate(self.features):
            x, thw = block(x, thw)
            if self.cfg.backbone == "mvit_v2_t" and idx in {0, 2, 7, 9}:
                feature_maps.append(self.reshape_tokens_to_feature_maps(x, thw))
            elif idx in {0, 2, 13, 15}:
                feature_maps.append(self.reshape_tokens_to_feature_maps(x, thw))

        return feature_maps


def get_encoder(cfg):
    if cfg.backbone.startswith("x3d"):
        encoder = X3DEncoder(
            cfg, include_last_block=cfg.get("include_last_block", False) or False
        )
    elif cfg.backbone.startswith("csn_r"):
        encoder = CSNEncoder(cfg)
    elif cfg.backbone.startswith("swin3d"):
        encoder = Swin3DEncoder(cfg)
    elif cfg.backbone.startswith("mvit"):
        encoder = MViTV2Encoder(cfg)
    else:
        raise Exception(f"{cfg.backbone} is not a valid 3D encoder")
    change_num_input_channels(encoder, cfg.num_input_channels)

    return encoder
