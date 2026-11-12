from skp.configs.rsna_ich_convnext_tiny_s16_auxiliary import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.auxiliary_region = "hu_teacher_union"
