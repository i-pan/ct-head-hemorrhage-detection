"""NaFlexViT-Base: fixed 512px, nine-channel, 50-epoch backbone ablation."""

from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as base_cfg

cfg = base_cfg.__deepcopy__()
cfg.backbone = "naflexvit_base_patch16_parfac_gap.e300_s576_in1k"
cfg.model = "classification.naflexvit"
