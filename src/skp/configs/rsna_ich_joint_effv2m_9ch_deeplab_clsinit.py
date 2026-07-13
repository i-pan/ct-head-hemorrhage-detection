from skp.configs.rsna_ich_joint_effv2m_9ch_deeplab import cfg


cfg.load_pretrained_encoder = (
    "experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)
