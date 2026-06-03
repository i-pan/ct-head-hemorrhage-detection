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

cfg.task = "classification"
cfg.model = "classification.net2d"
cfg.backbone = "resnet18"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.num_classes = 1
cfg.pool = "avg"
cfg.dropout = 0.0
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "simple2d"
cfg.data_dir = "./data/images"
cfg.annotations_file = "./data/train.csv"
cfg.inputs = "image"
cfg.targets = ["target"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.val_batch_size = cfg.batch_size
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.metrics = ["classification.AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 224
cfg.image_width = 224

cfg.train_transforms = A.Compose(
    [
        A.Resize(height=cfg.image_height, width=cfg.image_width, p=1),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
    ]
)

cfg.val_transforms = A.Compose(
    [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]
)
