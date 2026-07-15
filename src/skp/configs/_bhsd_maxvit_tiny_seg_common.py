from skp.configs import Config
from skp.configs._bhsd_effv2m_seg_common import make_cfg as make_base_cfg


CLASSIFIER_CHECKPOINT = (
    "experiments/rsna_ich_maxvit_tiny_fixedval_9ch/"
    "maxvit_tiny_9ch_20260714/fold0/checkpoints/last.ckpt"
)


def make_cfg() -> Config:
    cfg = make_base_cfg()
    cfg.backbone = "maxvit_tiny_tf_512.in1k"
    cfg.num_input_channels = 9
    cfg.num_slices = 3
    cfg.load_pretrained_encoder = CLASSIFIER_CHECKPOINT
    cfg.loss_params = dict(cfg.loss_params)
    cfg.loss_params["class_weights"] = [1, 1, 1, 1, 1, 2]
    cfg.val_batch_size = 32
    cfg.num_workers = 4
    cfg.val_num_workers = 2
    cfg.prefetch_factor = 2
    cfg.val_prefetch_factor = 2
    cfg.persistent_workers = False
    cfg.val_persistent_workers = False
    return cfg
