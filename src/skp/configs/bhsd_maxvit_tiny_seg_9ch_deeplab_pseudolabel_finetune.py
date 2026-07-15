from copy import deepcopy

from skp.configs.bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen import (
    cfg as base_cfg,
)


cfg = deepcopy(base_cfg)
cfg.freeze_encoder = False
cfg.frozen_encoder_eval = False
cfg.parameter_groups = [{"match": "encoder", "lr_scale": 0.1}]
