from skp.configs.rsna_ich_joint_effv2m_9ch_deeplab import cfg


CLASSIFIER_CHECKPOINT = (
    "experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)

cfg.load_pretrained_encoder = CLASSIFIER_CHECKPOINT
cfg.freeze_encoder = True
cfg.freeze_classifier = True
cfg.dataset = "rsna_ich_seg_cls_bhsd_val"
cfg.train_positive_series_only = True
cfg.bhsd_annotations_file = "./data/bhsd/bhsd_segmentation_slices.csv"
cfg.bhsd_mask_cache_size = 16

# The frozen classifier is retained for inference and validation, but only the
# segmentation decoder and head contribute trainable parameters and loss.
cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "losses": {
        "segmentation.PositiveDiceNegativeFocalLoss": {
            "params": {
                "activation_fn": "sigmoid",
                "compute_method": "per_sample",
                "dice_weight": 1.0,
                "focal_weight": 1.0,
                "positive_focal_weight": 1.0,
                "negative_focal_weight": 0.05,
                "gamma": 2.0,
                "alpha": None,
            },
            "output_key": "seg",
            "weight": 1.0,
        }
    }
}
cfg.parameter_groups = None

cfg.validate_classification_only = False
cfg.metrics = ["bhsd.SliceDiceHD95", "bhsd.VolumeDiceHD95"]
cfg.metric_activation_fn = "sigmoid"
cfg.metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg.val_metric = "volume_dice_any_best"
cfg.val_track = "max"

# Bound host RAM use under two-process DDP. Each sample contains a 9-channel
# image and a 6-channel mask, so deep worker prefetch queues are expensive.
cfg.num_workers = 4
cfg.val_num_workers = 2
cfg.prefetch_factor = 2
cfg.val_prefetch_factor = 2
cfg.persistent_workers = False
cfg.val_persistent_workers = False
