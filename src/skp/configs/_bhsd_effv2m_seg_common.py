import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import (
    dataloader_defaults,
    runtime_defaults,
    segmentation_2d_defaults,
)


def make_cfg() -> Config:
    cfg = Config()
    runtime_defaults(cfg)
    dataloader_defaults(cfg)
    segmentation_2d_defaults(cfg)

    cfg.project = "bhsd_ich_seg"
    cfg.task = "segmentation_2d"
    cfg.model = "segmentation.base"
    cfg.backbone = "tf_efficientnetv2_m"
    cfg.pretrained = False
    cfg.num_classes = 6
    cfg.activation_fn = "sigmoid"
    cfg.normalization = "linear"
    cfg.normalization_params = {
        "input_min": 0.0,
        "input_max": 1.0,
        "output_min": -1.0,
        "output_max": 1.0,
    }
    cfg.backbone_img_size = False

    cfg.decoder_type = "UnetDecoder"
    cfg.decoder_n_blocks = 5
    cfg.decoder_out_channels = [256, 128, 64, 32, 16]
    cfg.decoder_norm_layer = "bn"
    cfg.decoder_center_block = False
    cfg.decoder_attention_type = None
    cfg.decoder_separable_conv = False
    cfg.seg_dropout = 0.0

    cfg.fold = 0
    cfg.split_column = "fold"
    cfg.dataset = "bhsd_seg"
    cfg.annotations_file = "./data/bhsd/bhsd_segmentation_slices.csv"
    cfg.label_columns = [
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "any",
    ]
    cfg.mask_cache_size = 16

    cfg.loss = "segmentation.DiceFocalLoss"
    cfg.loss_params = {
        "activation_fn": "sigmoid",
        "compute_method": "per_sample",
        "dice_weight": 1.0,
        "focal_weight": 1.0,
        "gamma": 2.0,
    }

    cfg.batch_size = 32
    cfg.val_batch_size = 64
    cfg.num_workers = 8
    cfg.val_num_workers = 4
    cfg.prefetch_factor = 4
    cfg.val_prefetch_factor = 2
    cfg.num_epochs = 10
    cfg.optimizer = "AdamW"
    cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
    cfg.parameter_groups = [{"match": "encoder", "lr_scale": 0.1}]
    cfg.scheduler = "LinearWarmupCosineAnnealingLR"
    cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
    cfg.scheduler_interval = "step"

    cfg.metrics = ["bhsd.SliceDiceHD95", "bhsd.VolumeDiceHD95"]
    cfg.metric_activation_fn = "sigmoid"
    cfg.metric_threshold = 0.5
    cfg.metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cfg.val_metric = "volume_dice_any_best"
    cfg.val_track = "max"

    cfg.image_height = 512
    cfg.image_width = 512

    cfg.train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
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
                        fill_mask=None,
                        p=1,
                    ),
                ],
                p=1,
            ),
        ]
    )
    cfg.val_transforms = A.Compose([])
    return cfg
