"""Spatial logit max pooling with final downsampling stride reduced to one."""

from skp.configs.rsna_ich_maxvit_logitmax_s32_series_sampling import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.logit_output_stride = 16
