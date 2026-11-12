from skp.toolbox.ich_paths import classifier
"""Strict classifier-initialized encoder swaps for the selected BHSD recipe."""

from importlib import import_module
import torch
from torch import nn
from skp.toolbox.bhsd_convnext import ROOT
from skp.models.segmentation.decoders.unet import DecoderBlock

RUNS = {
    "maxvit_tiny": (
        "rsna_ich_maxvit_logitmax_s16_series_sampling",
        "series_e050_p0.5_seed88_20261108_maxvit_logitmax_s16",
    ),
    "effnetv2m": (
        "rsna_ich_effnetv2m_logitmax_s16_series_sampling",
        "series_e050_p0.5_seed88_20261109_effnetv2m_spatial_s16",
    ),
    "coatnet0": (
        "rsna_ich_coatnet0_logitmax_s16_series_sampling",
        "series_e050_p0.5_seed88_20261109_coatnet0_spatial_s16",
    ),
}


def classifier_checkpoint(family):
    config, run = RUNS[family]
    return classifier(family, ROOT / "experiments" / config / run / "fold0/checkpoints/last.ckpt")


class Net(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        config_name, _ = RUNS[family]
        cfg = import_module(f"skp.configs.{config_name}").cfg.__deepcopy__()
        cfg.pretrained = False
        classifier = import_module(f"skp.models.{cfg.model}").Net(cfg)
        state = torch.load(
            classifier_checkpoint(family),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        assert state["epoch"] == 49 and state["global_step"] > 0
        weights = {
            k.removeprefix("model."): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
        classifier.load_state_dict(weights, strict=True)
        self.encoder = classifier.backbone
        info = self.encoder.feature_info
        assert len(info) >= 4
        self.feature_indices = list(range(len(info) - 4, len(info)))
        channels = [x["num_chs"] for x in info[-4:]]
        # Identical five-block U-Net layout/widths; only encoder-facing widths differ.
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(classifier.feature_dim, channels[-2], 256),
                DecoderBlock(256, channels[-3], 128),
                DecoderBlock(128, channels[-4], 64),
                DecoderBlock(64, 0, 32),
                DecoderBlock(32, 0, 16),
            ]
        )
        self.head = nn.Conv2d(16, 1, 1)
        self.geometry = dict(
            feature_indices=self.feature_indices,
            encoder_channels=channels,
            head_channels=classifier.feature_dim,
            decoder_channels=[256, 128, 64, 32, 16],
        )

    def forward(self, x):
        last, skips = self.encoder.forward_intermediates(
            x * 2 - 1, indices=self.feature_indices
        )
        x = self.encoder.forward_head(last)
        assert [tuple(s.shape[-2:]) for s in skips] == [
            (128, 128),
            (64, 64),
            (32, 32),
            (32, 32),
        ]
        assert tuple(x.shape[-2:]) == (32, 32)
        for j, block in enumerate(self.decoder):
            x = block(x, skips[-2 - j] if j < 3 else None)
        return self.head(x)
