from skp.configs.bhsd_effv2m_seg_9ch_deeplab_joint_frozen import cfg


cfg.freeze_encoder = False
cfg.parameter_groups = [{"match": "encoder", "lr_scale": 0.1}]
