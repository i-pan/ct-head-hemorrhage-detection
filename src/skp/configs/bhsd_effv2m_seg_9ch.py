from skp.configs._bhsd_effv2m_seg_common import make_cfg


cfg = make_cfg()
cfg.num_input_channels = 9
cfg.num_slices = 3
cfg.load_pretrained_encoder = (
    "experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)
