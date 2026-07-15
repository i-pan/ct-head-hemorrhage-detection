from copy import deepcopy

from skp.configs.bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen import (
    cfg as base_cfg,
)


cfg = deepcopy(base_cfg)
cfg.load_pretrained_model = None
