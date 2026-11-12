import os
from skp.configs.rsna_ich_effv2m_sequence_bigru import cfg as base_cfg


cfg = base_cfg.__deepcopy__()
cfg.feature_dir = "./data/features/rsna_ich_maxvit_tiny_9ch"
cfg.feature_dim = 512

cfg.feature_dir = os.environ.get("ICH_LEGACY_FEATURES", cfg.feature_dir)
