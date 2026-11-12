from skp.configs.rsna_ich_maxvit_logitmax_s16_series_sampling import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.model = "classification.spatial_logit_backbones"
cfg.backbone = "convnext_tiny.fb_in1k"
