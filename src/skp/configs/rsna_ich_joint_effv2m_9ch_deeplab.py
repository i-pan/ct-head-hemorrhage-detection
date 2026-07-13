import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import (
    cls_seg_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
cls_seg_2d_defaults(cfg)

cfg.project = "rsna_ich"
cfg.task = "cls_seg"
cfg.model = "segmentation.unet_cls"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 9
cfg.num_classes = 6
cfg.cls_num_classes = 6
cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.decoder_out_channels = 256
cfg.decoder_norm_layer = "bn"
cfg.decoder_act_layer = "relu"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.aspp_separable = True
cfg.aspp_dropout = 0.1
cfg.atrous_rates = (6, 12, 18, 24)
cfg.seg_dropout = 0.0
cfg.cls_dropout = 0.2
cfg.pool = "avg"
cfg.normalization = "linear"
cfg.normalization_params = {
    "input_min": 0.0,
    "input_max": 1.0,
    "output_min": -1.0,
    "output_max": 1.0,
}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "rsna_ich_seg_cls"
cfg.data_dir = "/mnt/champaca/rsna-intracranial-hemorrhage-detection-16bit-png"
cfg.annotations_file = "./data/folds/rsna_ich_fixed_val_splits.csv"
cfg.labels_file = f"{cfg.data_dir}/slice_labels.csv"
cfg.rescale_file = f"{cfg.data_dir}/rescale_values.csv"
cfg.pseudolabel_dir = "./data/pseudolabels/rsna_9ch_last"
cfg.pseudolabel_manifest = f"{cfg.pseudolabel_dir}/manifest.csv"
cfg.require_positive_pseudolabels = True
cfg.mask_cache_size = 64
cfg.label_columns = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
cfg.ct_windows = [
    (40, 80),
    (80, 200),
    (600, 2800),
]
cfg.num_slices = 3
cfg.flatten_depth_to_channels = True

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "losses": {
        "ich.WeightedBCEWithLogitsLoss": {
            "params": {
                "class_names": cfg.label_columns,
                "class_weights": [1, 1, 1, 1, 1, 2],
            },
            "output_key": "cls",
            "weight": 1.0,
        },
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
            "weight": 0.25,
        },
    }
}

cfg.batch_size = 32
cfg.val_batch_size = 64
cfg.num_workers = 8
cfg.val_num_workers = 2
cfg.prefetch_factor = 4
cfg.val_prefetch_factor = 2
cfg.val_persistent_workers = False
cfg.num_epochs = 3
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.parameter_groups = [{"match": "segmenter.encoder", "lr_scale": 0.1}]
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.validate_classification_only = True
cfg.metrics = ["ich.SliceAUROC", "ich.SeriesAUROC"]
cfg.metric_activation_fn = "sigmoid"
cfg.series_metric_aggregations = ["max", "mean", "top3_mean"]
cfg.val_metric = "auc_any"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512
cfg.depth_flip_p = 0.25
cfg.horizontal_flip_p = 0.25
cfg.vertical_flip_p = 0.25

cfg.train_transforms = A.Compose(
    [
        A.OneOf(
            [
                A.NoOp(p=1),
                A.Affine(
                    translate_percent=(-0.04, 0.04),
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=1,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.12,
                    contrast_limit=0.18,
                    p=1,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1),
                        A.MotionBlur(blur_limit=5, p=1),
                    ],
                    p=1,
                ),
                A.GaussNoise(std_range=(0.01, 0.04), p=1),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(16, 64),
                    hole_width_range=(16, 64),
                    fill=0,
                    fill_mask=0,
                    p=1,
                ),
            ],
            p=1,
        ),
    ]
)

cfg.val_transforms = A.Compose([])
