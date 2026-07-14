from skp.configs._rsna_ich_effv2m_fixedval_common import make_cfg


cfg = make_cfg()
cfg.backbone = "maxvit_tiny_tf_512.in1k"
cfg.num_input_channels = 9
cfg.num_slices = 3
cfg.flatten_depth_to_channels = True
cfg.depth_flip_p = 0.25
