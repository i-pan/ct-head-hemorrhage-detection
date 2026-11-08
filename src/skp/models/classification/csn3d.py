"""CSN-R101 center-slice classification using three CT windows over axial depth.

Load the unmodified Kinetics model strictly before replacing its video head.
Keep the published backbone strides; adaptive global pooling supports depth 3.
"""

from torch import nn
from pytorchvideo.models.hub import csn_r101
from skp.models.normalization import normalize_input


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.backbone != "csn_r101" or cfg.num_input_channels != 3:
            raise ValueError("CSN-R101 requires three input window channels")
        if cfg.num_slices != 3 or cfg.flatten_depth_to_channels:
            raise ValueError("Expected three unflattened contiguous slices")
        if cfg.pool != "avg":
            raise ValueError("This ablation requires average pooling")
        self.cfg = cfg
        net = csn_r101(pretrained=cfg.pretrained)
        self.backbone = nn.Sequential(*list(net.blocks)[:5])
        self.feature_dim = 2048
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, cfg.num_classes)
        self.criterion = None
        self.backbone_frozen = False

    def set_criterion(self, loss):
        self.criterion = loss

    def extract_features(self, x, normalize=True):
        if x.ndim != 5 or x.shape[1:3] != (3, 3):
            raise ValueError(
                f"Expected B x 3 windows x 3 slices x H x W, got {tuple(x.shape)}"
            )
        if normalize:
            x = normalize_input(x, self.cfg)
        return self.pooling(self.backbone(x)).flatten(1)

    def forward(self, batch, return_loss=False, return_features=False):
        features = self.extract_features(batch["x"])
        out = {"logits": self.linear(self.dropout(features))}
        if return_features:
            out["features"] = features
        if return_loss:
            if self.criterion is None:
                raise RuntimeError("Set the training criterion first")
            loss = self.criterion(out, batch)
            out.update(loss if isinstance(loss, dict) else {"loss": loss})
        return out
