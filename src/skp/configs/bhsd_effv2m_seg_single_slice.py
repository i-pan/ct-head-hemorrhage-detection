from skp.configs._bhsd_effv2m_seg_common import make_cfg


cfg = make_cfg()
cfg.num_input_channels = 3
cfg.num_slices = 1
cfg.load_pretrained_encoder = (
    "experiments/rsna_ich_effv2m_fixedval_single_slice/"
    "robust-crawdad-5591/fold0/checkpoints/last.ckpt"
)
