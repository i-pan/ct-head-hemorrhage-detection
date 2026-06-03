from skp.configs import Config
from skp.configs.defaults import (
    dataloader_defaults,
    runtime_defaults,
    segmentation_3d_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
segmentation_3d_defaults(cfg)

cfg.task = "segmentation_3d"
cfg.model = "segmentation.base_3d"
cfg.decoder_type = "DeepLabV3PlusDecoder3d"
cfg.backbone = "x3d_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.num_classes = 1
cfg.seg_dropout = 0.0
cfg.dim0_strides = [2, 2, 2, 1, 1]
cfg.output_stride = 8
cfg.decoder_out_channels = 256
cfg.atrous_rates = (3, 6, 10, 14)
cfg.aspp_separable = True
cfg.aspp_dropout = 0.0
cfg.activation_fn = "sigmoid"
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "simple2dc_seg"
cfg.data_dir = "./data/volumes"
cfg.seg_data_dir = "./data/masks"
cfg.annotations_file = "./data/train.csv"
cfg.inputs = "volume"
cfg.targets = "mask"
cfg.cv2_load_flag = None
cfg.data_format = "cthw"
cfg.rescale_mask = 255.0

cfg.loss = "segmentation.DiceBCELoss"
cfg.loss_params = {
    "compute_method": "per_batch",
    "ignore_background": False,
    "convert_labels_to_onehot": False,
    "activation_fn": cfg.activation_fn,
}

cfg.batch_size = 2
cfg.val_batch_size = 1
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.metrics = ["segmentation.MultilabelDiceScore"]
cfg.metric_activation_fn = cfg.activation_fn
cfg.metric_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.dim0 = 64
cfg.dim1 = 128
cfg.dim2 = 128
cfg.num_slices = cfg.dim0
cfg.image_height = cfg.dim1
cfg.image_width = cfg.dim2

cfg.train_transforms = None
cfg.val_transforms = None
