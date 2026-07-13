from skp.configs.bhsd_effv2m_seg_9ch_deeplab_joint_frozen import cfg


# Controlled no-pseudolabel baseline: use the same frozen classification
# encoder and BHSD setup, but initialize the DeepLabV3+ decoder from scratch.
cfg.load_pretrained_model = None
cfg.load_pretrained_encoder = (
    "experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)
