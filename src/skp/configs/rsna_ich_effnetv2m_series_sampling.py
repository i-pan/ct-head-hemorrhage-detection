"""50-epoch strict backbone ablation; inherit the full MaxViT protocol."""

from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as base_cfg

cfg = base_cfg.__deepcopy__()
cfg.backbone = "tf_efficientnetv2_m.in1k"
