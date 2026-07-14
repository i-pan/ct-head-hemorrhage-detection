from skp.configs._bhsd_effv2m_seg_common import make_cfg


cfg = make_cfg()
cfg.num_input_channels = 9
cfg.num_slices = 3

cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.decoder_out_channels = 256
cfg.decoder_norm_layer = "bn"
cfg.decoder_act_layer = "relu"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.aspp_separable = True
cfg.aspp_dropout = 0.1
cfg.atrous_rates = (6, 12, 18, 24)

cfg.load_pretrained_model = (
    "experiments/rsna_ich_joint_effv2m_9ch_deeplab_frozen/"
    "frozen_decoder_positive_series_20260712/fold0/checkpoints/last.ckpt"
)
cfg.load_pretrained_encoder = (
    "experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)
cfg.freeze_encoder = True
cfg.frozen_encoder_eval = True
cfg.parameter_groups = None

cfg.num_workers = 4
cfg.val_num_workers = 2
cfg.prefetch_factor = 2
cfg.val_prefetch_factor = 2
cfg.persistent_workers = False
cfg.val_persistent_workers = False
