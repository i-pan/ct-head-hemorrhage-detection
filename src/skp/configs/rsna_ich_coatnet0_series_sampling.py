"""50-epoch strict backbone ablation; inherit the full MaxViT protocol."""

from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as base_cfg

cfg = base_cfg.__deepcopy__()
cfg.backbone = "coatnet_0_rw_224.sw_in1k"
cfg.backbone_img_size = True  # Adapt positional bias to the same 512px input.
