import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import cls_seg_2d_defaults, dataloader_defaults, runtime_defaults


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
cls_seg_2d_defaults(cfg)

cfg.task = "cls_seg"
cfg.model = "segmentation.unet_cls"
cfg.backbone = "resnet18"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.num_classes = 1
cfg.decoder_type = "UnetDecoder"
cfg.decoder_out_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.seg_dropout = 0.0
cfg.cls_dropout = 0.0
cfg.pool = "avg"
cfg.activation_fn = "sigmoid"
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "simple2d_seg_cls"
cfg.data_dir = "./data/images"
cfg.seg_data_dir = "./data/masks"
cfg.annotations_file = "./data/train.csv"
cfg.inputs = "image"
cfg.masks = "mask"
cfg.targets = ["target"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.rescale_mask = 255.0

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "losses": {
        "classification.BCEWithLogitsLoss": {
            "params": {},
            "output_key": "cls",
            "weight": 1.0,
        },
        "segmentation.DiceLoss": {
            "params": {
                "convert_labels_to_onehot": False,
                "pred_power": 2.0,
                "ignore_background": False,
                "activation_fn": cfg.activation_fn,
                "compute_method": "per_batch",
            },
            "output_key": "seg",
            "weight": 1.0,
        },
    }
}

cfg.batch_size = 16
cfg.val_batch_size = cfg.batch_size
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.metrics = ["segmentation.MultilabelDiceScore", "classification.AUROC"]
cfg.metric_activation_fn = cfg.activation_fn
cfg.metric_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 256
cfg.image_width = 256

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
