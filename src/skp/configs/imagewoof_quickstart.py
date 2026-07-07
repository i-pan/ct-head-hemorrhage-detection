import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import (
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
classification_2d_defaults(cfg)

# Task/model
cfg.project = "imagewoof_quickstart"
cfg.task = "classification"
cfg.model = "classification.net2d"
cfg.backbone = "resnet18"
cfg.pretrained = False  # Set True if you want timm ImageNet weights.
cfg.num_input_channels = 3
cfg.num_classes = 10
cfg.pool = "avg"
cfg.dropout = 0.0
cfg.normalization = "linear"
cfg.normalization_params = {
    "input_min": 0,
    "input_max": 255,
    "output_min": 0,
    "output_max": 1,
}
cfg.backbone_img_size = False

# Data
cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "simple2d"
cfg.data_dir = "./data/imagewoof2-160"
cfg.annotations_file = "./data/imagewoof2-160/annotations.csv"
cfg.inputs = "image"
cfg.targets = ["target"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR

# Optimization
cfg.loss = "classification.CrossEntropyLoss"
cfg.loss_params = {}
cfg.batch_size = 64
cfg.val_batch_size = cfg.batch_size
cfg.num_epochs = 1
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

# Metrics/checkpointing
cfg.metrics = ["classification.Accuracy"]
cfg.metric_activation_fn = "softmax"
cfg.val_metric = "accuracy"
cfg.val_track = "max"
cfg.mlflow_system_monitor = False

# Images/transforms
cfg.image_height = 160
cfg.image_width = 160
cfg.train_transforms = A.Compose(
    [
        A.Resize(height=cfg.image_height, width=cfg.image_width, p=1),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
    ]
)
cfg.val_transforms = A.Compose(
    [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]
)

# Smaller worker count is friendlier for laptops and CI smoke tests.
cfg.num_workers = 2
