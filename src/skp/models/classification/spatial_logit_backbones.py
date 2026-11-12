"""Stride-16 pretrained backbones with the MaxViT spatial logit-max head."""

from unittest.mock import patch
import torch
import torch.nn.functional as F
import timm
from timm.layers.pos_embed_rel import RelPosBias
from skp.models.classification.net2d import Net as BaseNet


class Net(BaseNet):
    def __init__(self, cfg):
        if cfg.logit_output_stride != 16:
            raise ValueError("This ablation requires output stride 16")
        coat = cfg.backbone == "coatnet_0_rw_224.sw_in1k"
        if not coat and cfg.backbone not in (
            "convnext_tiny.fb_in1k",
            "tf_efficientnetv2_m.in1k",
        ):
            raise ValueError(cfg.backbone)

        def create(name, **kwargs):
            if not coat:
                kwargs["output_stride"] = 16
            return timm.create_model(name, **kwargs)

        with patch("skp.models.classification.net2d.create_model", side_effect=create):
            super().__init__(cfg)
        if coat:
            first = self.backbone.stages[-1].blocks[0]
            assert first.shortcut.pool.stride == 2 and first.norm1.down.pool.stride == 2
            # Keep SAME spatial shape on both residual branches, preserving kernels.
            first.shortcut.pool.stride = 1
            first.shortcut.pool.padding = 0
            first.norm1.down.pool.stride = 1
            first.norm1.down.pool.padding = 0
            # Even-kernel pooling at stride one shrinks spatial size; explicit SAME pad.
            first.shortcut.pool = torch.nn.Sequential(
                torch.nn.ZeroPad2d((0, 1, 0, 1)), first.shortcut.pool
            )
            first.norm1.down.pool = torch.nn.Sequential(
                torch.nn.ZeroPad2d((0, 1, 0, 1)), first.norm1.down.pool
            )
            for block in self.backbone.stages[-1].blocks:
                old = block.attn.rel_pos
                assert old.window_size == (16, 16) and old.prefix_tokens == 0
                new = RelPosBias(
                    window_size=(32, 32),
                    num_heads=old.relative_position_bias_table.shape[1],
                )
                table = old.relative_position_bias_table.detach().T.reshape(
                    1, -1, 31, 31
                )
                resized = F.interpolate(
                    table, size=(63, 63), mode="bicubic", align_corners=False
                )
                with torch.no_grad():
                    new.relative_position_bias_table.copy_(
                        resized.reshape(-1, 63 * 63).T
                    )
                block.attn.rel_pos = new

    def forward(self, batch, return_loss=False, return_features=False):
        features = self.backbone(self.normalize(batch["x"]))
        dropped = F.dropout2d(features, p=self.dropout.p, training=self.training)
        logit_map = self.linear(dropped.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = {"logits": logit_map.flatten(2).max(dim=-1).values}
        if return_features:
            out.update(features=self.pooling(features), logit_map=logit_map)
        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss if isinstance(loss, dict) else {"loss": loss})
        return out
