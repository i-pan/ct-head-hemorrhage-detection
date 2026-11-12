"""MaxViT spatial linear classifier followed by per-class spatial max pooling."""

import torch.nn.functional as F
from skp.models.classification.net2d import Net as BaseNet


class Net(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.backbone != "maxvit_tiny_tf_512.in1k" or cfg.pool != "avg":
            raise ValueError(
                "This ablation requires the MaxViT-Tiny average-pool baseline"
            )
        if cfg.get("multisample_dropout", False) or cfg.get("model_activation_fn"):
            raise ValueError("Expected raw logits and ordinary dropout")
        stride = cfg.get("logit_output_stride", 32)
        if stride not in (16, 32):
            raise ValueError("Expected output stride 16 or 32")
        if stride == 16:
            block = self.backbone.stages[-1].blocks[0].conv
            assert block.conv2_kxk.stride == (2, 2)
            assert block.shortcut.pool.stride == (2, 2)
            # Preserve kernels, padding, all weights, and attention partition sizes.
            # Both residual branches must have the same spatial resolution.
            block.conv2_kxk.stride = (1, 1)
            block.shortcut.pool.stride = (1, 1)

    def forward(self, batch, return_loss=False, return_features=False):
        features = self.backbone(self.normalize(batch["x"]))
        # One dropout mask per sample/channel, shared across spatial positions:
        # this is the spatial counterpart of baseline vector dropout.
        dropped = F.dropout2d(features, p=self.dropout.p, training=self.training)
        logit_map = self.linear(dropped.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = {"logits": logit_map.flatten(2).max(dim=-1).values}
        if return_features:
            out["features"] = self.pooling(features)
            out["logit_map"] = logit_map
        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss if isinstance(loss, dict) else {"loss": loss})
        return out
