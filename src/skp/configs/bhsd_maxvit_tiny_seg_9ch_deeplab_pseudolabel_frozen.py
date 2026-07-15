from skp.configs._bhsd_maxvit_tiny_seg_common import (
    CLASSIFIER_CHECKPOINT,
    make_cfg,
)


cfg = make_cfg()
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
    "experiments/rsna_ich_joint_maxvit_tiny_9ch_deeplab_frozen/"
    "maxvit_pseudolabel_decoder_any2_20260715/fold0/checkpoints/last.ckpt"
)
# Component loading follows full-model loading, restoring the canonical encoder.
cfg.load_pretrained_encoder = CLASSIFIER_CHECKPOINT
cfg.freeze_encoder = True
cfg.frozen_encoder_eval = True
cfg.parameter_groups = None

cfg.num_workers = 4
cfg.val_num_workers = 2
cfg.prefetch_factor = 2
cfg.val_prefetch_factor = 2
cfg.persistent_workers = False
cfg.val_persistent_workers = False
