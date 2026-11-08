"""CSN-R101: same 50-epoch protocol, with windows x contiguous axial depth."""

from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as base_cfg

cfg = base_cfg.__deepcopy__()
cfg.model = "classification.csn3d"
cfg.backbone = "csn_r101"
cfg.num_input_channels = 3
cfg.flatten_depth_to_channels = False
