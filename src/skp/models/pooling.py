import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import SelectAdaptivePool2d

from skp.configs.base import Config


class GeM(nn.Module):
    def __init__(
        self, p: int = 3, eps: float = 1e-6, dim: int = 2, flatten: bool = True
    ):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        assert dim in {2, 3}, f"dim must be one of [2, 3], not {dim}"
        self.dim = dim
        if self.dim == 2:
            self.func = F.adaptive_avg_pool2d
        elif self.dim == 3:
            self.func = F.adaptive_avg_pool3d
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assumes x.shape is (n, c, [t], h, w)
        x = self.func(x.clamp(min=self.eps).pow(self.p), output_size=1).pow(
            1.0 / self.p
        )
        return self.flatten(x)


def adaptive_avgmax_pool3d(x: torch.Tensor, output_size: int = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool3d(x: torch.Tensor, output_size: int = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool3d(x: torch.Tensor, pool_type: str, output_size: int = 1) -> torch.Tensor:
    """Selectable global pooling function with dynamic input kernel size"""
    if pool_type == "avg":
        x = F.adaptive_avg_pool3d(x, output_size)
    elif pool_type == "avgmax":
        x = adaptive_avgmax_pool3d(x, output_size)
    elif pool_type == "catavgmax":
        x = adaptive_catavgmax_pool3d(x, output_size)
    elif pool_type == "max":
        x = F.adaptive_max_pool3d(x, output_size)
    else:
        assert False, "Invalid pool type: %s" % pool_type
    return x


class FastAdaptiveAvgPool3d(nn.Module):
    def __init__(self, flatten: bool = False):
        super(FastAdaptiveAvgPool3d, self).__init__()
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean((2, 3, 4), keepdim=not self.flatten)


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: int = 1):
        super(AdaptiveAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return adaptive_avgmax_pool3d(x, self.output_size)


class AdaptiveCatAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: int = 1):
        super(AdaptiveCatAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return adaptive_catavgmax_pool3d(x, self.output_size)


class SelectAdaptivePool3d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(self, output_size: int = 1, pool_type: str = "fast", flatten: bool = False):
        super(SelectAdaptivePool3d, self).__init__()
        self.pool_type = (
            pool_type or ""
        )  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == "":
            self.pool = nn.Identity()  # pass through
        elif pool_type == "fast":
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool3d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool3d(output_size)
        elif pool_type == "avgmax":
            self.pool = AdaptiveAvgMaxPool3d(output_size)
        elif pool_type == "catavgmax":
            self.pool = AdaptiveCatAvgMaxPool3d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool3d(output_size)
        else:
            assert False, "Invalid pool type: %s" % pool_type

    def is_identity(self) -> bool:
        return not self.pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + "pool_type="
            + self.pool_type
            + ", flatten="
            + str(self.flatten)
            + ")"
        )


def get_pool_layer(cfg: Config, dim: int) -> nn.Module:
    assert cfg.pool in [
        "avg",
        "max",
        "fast",
        "avgmax",
        "catavgmax",
        "gem",
        ""
    ], f"{cfg.pool} is not a valid pooling layer"
    params = cfg.pool_params or {}
    if cfg.pool == "gem":
        return GeM(**params, dim=dim)
    else:
        if dim == 2:
            return SelectAdaptivePool2d(pool_type=cfg.pool, flatten=True)
        elif dim == 3:
            return SelectAdaptivePool3d(pool_type=cfg.pool, flatten=True)
