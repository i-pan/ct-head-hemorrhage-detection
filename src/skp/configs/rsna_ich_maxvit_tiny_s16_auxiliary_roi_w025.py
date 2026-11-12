from skp.configs.rsna_ich_maxvit_tiny_s16_auxiliary import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.auxiliary_region = "hu_teacher_union"
cfg.auxiliary_outside_roi_weight = 0.25
