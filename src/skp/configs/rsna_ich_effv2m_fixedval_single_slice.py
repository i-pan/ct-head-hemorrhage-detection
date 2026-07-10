from skp.configs._rsna_ich_effv2m_fixedval_common import make_cfg


cfg = make_cfg()
cfg.num_input_channels = 3
cfg.num_slices = 1
cfg.flatten_depth_to_channels = False
cfg.depth_flip_p = 0.0
