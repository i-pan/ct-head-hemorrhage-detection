from skp.configs.rsna_ich_convnext_tiny_s16_auxiliary_roi import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.auxiliary_outside_roi_weight = 0.25
