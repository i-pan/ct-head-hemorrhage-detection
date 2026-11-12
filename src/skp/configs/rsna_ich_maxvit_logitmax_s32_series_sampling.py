"""50-epoch MaxViT-Tiny: spatial linear head, then max over logits."""

from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.model = "classification.maxvit_logit_pool"
cfg.logit_output_stride = 32
