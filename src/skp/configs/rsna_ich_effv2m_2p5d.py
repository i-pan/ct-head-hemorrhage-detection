import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import (
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)
from skp.datasets.rsna_ich_2p5d import DepthFlip


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
classification_2d_defaults(cfg)

cfg.project = "rsna_ich"
cfg.task = "classification"
cfg.model = "classification.efficientnet3d_stem"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.num_slices = 3
cfg.num_classes = 6
cfg.pool = "avg"
cfg.dropout = 0.2
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
cfg.dataset = "rsna_ich_2p5d"
cfg.data_dir = "/mnt/champaca/rsna-intracranial-hemorrhage-detection-16bit-png"
cfg.annotations_file = "./data/folds/rsna_ich_splits.csv"
cfg.labels_file = f"{cfg.data_dir}/slice_labels.csv"
cfg.rescale_file = f"{cfg.data_dir}/rescale_values.csv"
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

cfg.loss = "ich.WeightedBCEWithLogitsLoss"
cfg.loss_params = {
    "class_names": cfg.label_columns,
    "class_weights": [1, 1, 1, 1, 1, 5],
}

cfg.batch_size = 32
cfg.val_batch_size = 64
cfg.num_workers = 8
cfg.val_num_workers = 8
cfg.num_epochs = 5
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.metrics = ["ich.SliceAUROC", "ich.SeriesAUROC"]
cfg.metric_activation_fn = "sigmoid"
cfg.series_metric_aggregations = ["max", "mean", "top3_mean"]
cfg.val_metric = "auc_any"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

cfg.train_transforms = A.Compose(
    [
        DepthFlip(depth=cfg.num_slices, channels_per_slice=len(cfg.ct_windows), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.04, 0.04),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.75,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.18,
            p=0.35,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MotionBlur(blur_limit=5, p=1),
            ],
            p=0.15,
        ),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.25),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0,
            p=0.25,
        ),
    ]
)

cfg.val_transforms = A.Compose([])
