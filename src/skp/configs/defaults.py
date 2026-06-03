from skp.configs import Config


CONFIG_FIELD_DOCS = {
    "runtime": {
        "save_dir": "Root directory for experiment outputs before config/run/fold subdirectories are appended.",
        "project": "MLflow experiment name.",
        "mlflow_tracking_uri": "Optional MLflow tracking URI. If None, MLflow uses its environment/default behavior.",
        "debug": "Boolean flag set by --debug; intended for configs/datasets to reduce work during quick checks.",
        "float32_matmul_precision": "Value passed to torch.set_float32_matmul_precision.",
        "save_top_k": "Number of best checkpoints to keep according to val_metric.",
        "save_weights_only": "If True, checkpoint only model weights instead of full training state.",
        "early_stopping": "Enable Lightning early stopping on val_metric.",
        "mlflow_system_monitor": "Use MLflow system metrics as the default hardware/process monitor.",
        "log_gpu_stats": "Opt-in Lightning GPU stats logging; usually redundant with MLflow system metrics.",
        "find_unused_parameters": "DDPStrategy find_unused_parameters flag.",
        "accumulate_grad_batches": "Number of batches to accumulate before each optimizer step.",
        "progressive_resizing": "Optional staged-training dictionary handled by skp.runners.run_progressive_resizing.",
        "hyperparameter_sweep": "Optional built-in sweep dictionary handled by skp.runners.run_hyperparameter_sweep.",
        "ema": "EMA callback settings. ema['on'] controls whether the callback is enabled.",
    },
    "dataloader": {
        "num_workers": "Training DataLoader worker count.",
        "pin_memory": "DataLoader pin_memory flag.",
        "persistent_workers": "Keeps workers alive across epochs when num_workers > 0.",
        "skip_failed_data": "Dataset-level flag for retrying after failed sample loads.",
        "double_cv": "Optional nested-CV outer fold identifier.",
        "split_column": "Annotation column used to select train/val/test rows. Set explicitly in every experiment config.",
        "train_split_values": "Fixed-split values in split_column used for training rows. Leave None for k-fold mode.",
        "val_split_values": "Fixed-split values in split_column used for validation rows. Leave None for k-fold mode.",
        "test_split_values": "Fixed-split values in split_column used for test rows. Leave None for k-fold mode.",
        "inference_transforms": "Optional transform pipeline used only by inference-mode datasets.",
        "wds": "Use WebDataset loaders instead of standard torch DataLoader.",
        "val_num_workers": "Optional validation worker override.",
        "num_val_batches_per_gpu": "Validation WebDataset epoch length per GPU.",
        "sampler": "Optional sampler class name from skp.tasks.samplers for training.",
        "num_iterations_per_epoch": "Optional fixed number of train iterations per epoch.",
    },
    "classification_2d": {
        "features_only": "Build backbone feature extractor without final classification head when supported.",
        "freeze_backbone": "Freeze backbone parameters.",
        "load_pretrained_backbone": "Path to backbone weights to load before training.",
        "load_pretrained_model": "Path to full model weights to load before training.",
        "enable_gradient_checkpointing": "Enable model-supported activation checkpointing.",
        "multisample_dropout": "Use model-supported multisample dropout.",
        "model_activation_fn": "Optional model output activation for wrappers that apply one.",
        "pool_params": "Optional pooling layer parameter dictionary.",
        "vars": "Optional auxiliary variable names consumed by some custom models.",
        "sampling_weight_col": "Annotation column used by weighted dataset sampling.",
        "group_index_col": "Annotation column identifying grouped samples.",
        "mix_proba": "Probability of applying MixUp during a training batch.",
        "mixup": "Beta distribution alpha for MixUp.",
    },
    "segmentation_2d": {
        "deep_supervision": "Enable auxiliary segmentation heads when supported.",
        "decoder_type": "Required decoder class name from skp.models.segmentation.decoders; set explicitly in each experiment config.",
        "decoder_out_channels": "Decoder channel specification; meaning depends on decoder type.",
        "decoder_attention_type": "Optional decoder attention mode.",
        "freeze_encoder": "Freeze encoder parameters.",
        "freeze_decoder": "Freeze decoder parameters.",
        "output_size": "Optional segmentation output resize target.",
        "label_dtype": "Optional dtype conversion for segmentation labels.",
        "use_sliding_window_inference": "Use MONAI sliding-window inference during validation.",
        "sliding_window_overlap": "Fractional overlap for sliding-window inference.",
    },
    "segmentation_3d": {
        "features_only": "Use 3D backbone feature maps for decoder models.",
        "depth": "Optional input depth for models that need it at initialization.",
        "output_stride": "Target encoder output stride for supported backbones/decoders.",
        "dim0_strides": "3D encoder stride schedule along the slice/depth axis.",
        "decoder_norm_layer": "3D decoder normalization layer name.",
        "num_sample_classes": "Number of classes sampled by stochastic segmentation heads.",
        "ignore_background_class_for_sampling": "Exclude class 0 when stochastic heads sample present classes.",
        "mixup_weight": "Relative probability of MONAI MixUp when both MixUp and CutMix are enabled.",
        "cutmix_weight": "Relative probability of MONAI CutMix when both MixUp and CutMix are enabled.",
        "log_grad_norm": "Opt-in logging of total gradient norm before optimizer steps.",
    },
}


def runtime_defaults(cfg: Config) -> Config:
    """Apply project/runtime defaults shared by all experiment templates."""
    cfg.save_dir = "./experiments"
    cfg.project = "skp"
    cfg.mlflow_tracking_uri = None
    cfg.debug = False
    cfg.float32_matmul_precision = "high"
    cfg.save_top_k = 1
    cfg.save_weights_only = False
    cfg.early_stopping = False
    cfg.early_stopping_patience = 5
    cfg.early_stopping_min_delta = 0.0
    cfg.early_stopping_verbose = False
    cfg.mlflow_system_monitor = True
    cfg.log_gpu_stats = False
    cfg.gpu_stats_log_step_interval = 5
    cfg.metric_activation_fn = None
    cfg.metric_include_classes = None
    cfg.metric_thresholds = None
    cfg.metric_labels_to_onehot = False
    cfg.metric_ignore_class0 = False
    cfg.metric_invert_background = False
    cfg.binary_cls_threshold = 0.5
    cfg.metric_seq_mode = False
    cfg.find_unused_parameters = False
    cfg.accumulate_grad_batches = 1
    cfg.progressive_resizing = None
    cfg.hyperparameter_sweep = None
    cfg.ema = {
        "on": False,
        "decay": 0.9999,
        "update_after_step": 0,
        "update_every_n_steps": 1,
        "use_warmup": False,
        "warmup_gamma": 1.0,
        "warmup_power": 2 / 3,
        "switch_ema": False,
    }
    return cfg


def dataloader_defaults(cfg: Config) -> Config:
    """Apply defaults for standard and WebDataset dataloader construction."""
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.persistent_workers = True
    cfg.skip_failed_data = False
    cfg.double_cv = None
    cfg.split_column = None
    cfg.train_split_values = None
    cfg.val_split_values = None
    cfg.test_split_values = None
    cfg.inference_transforms = None
    cfg.wds = False
    cfg.val_num_workers = None
    cfg.num_val_batches_per_gpu = None
    cfg.sampler = None
    cfg.num_iterations_per_epoch = None
    return cfg


def classification_2d_defaults(cfg: Config) -> Config:
    """Apply defaults for 2D image classification experiments."""
    cfg.features_only = False
    cfg.freeze_backbone = False
    cfg.load_pretrained_backbone = None
    cfg.load_pretrained_model = None
    cfg.enable_gradient_checkpointing = False
    cfg.multisample_dropout = False
    cfg.model_activation_fn = None
    cfg.pool_params = None
    cfg.vars = None
    cfg.sampling_weight_col = None
    cfg.group_index_col = None
    cfg.mix_proba = None
    cfg.mixup = None
    return cfg


def segmentation_2d_defaults(cfg: Config) -> Config:
    """Apply defaults for 2D semantic segmentation experiments."""
    cfg.deep_supervision = False
    cfg.deep_supervision_num_levels = None
    cfg.decoder_type = None
    cfg.decoder_out_channels = None
    cfg.decoder_separable_conv = False
    cfg.decoder_attention_type = None
    cfg.decoder_center_block = False
    cfg.use_psp = False
    cfg.psp_pool_sizes = None
    cfg.psp_out_channels = None
    cfg.atrous_rates = None
    cfg.aspp_separable = False
    cfg.aspp_dropout = 0.0
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.load_pretrained_encoder = None
    cfg.load_pretrained_decoder = None
    cfg.load_pretrained_model = None
    cfg.load_pretrained_segmentation_heads = False
    cfg.load_segmentation_head_classes = None
    cfg.enable_gradient_checkpointing = False
    cfg.output_size = None
    cfg.seg_dropout = 0.0
    cfg.label_dtype = None
    cfg.mixup = None
    cfg.use_sliding_window_inference = False
    cfg.sliding_window_overlap = 0.25
    return cfg


def cls_seg_2d_defaults(cfg: Config) -> Config:
    """Apply defaults for joint 2D classification and segmentation experiments."""
    segmentation_2d_defaults(cfg)
    cfg.cls_num_classes = None
    cfg.load_pretrained_segmenter = None
    cfg.assume_mask_empty_if_not_present = False
    cfg.pool_params = None
    return cfg


def segmentation_3d_defaults(cfg: Config) -> Config:
    """Apply defaults for 3D semantic segmentation experiments."""
    cfg.features_only = True
    cfg.deep_supervision = False
    cfg.deep_supervision_num_levels = None
    cfg.ds_levels = 1
    cfg.depth = None
    cfg.output_size = None
    cfg.output_stride = 32
    cfg.dim0_strides = [2, 2, 2, 2, 2]
    cfg.block_kernel_size = None
    cfg.include_last_block = False
    cfg.decoder_type = None
    cfg.decoder_out_channels = None
    cfg.decoder_norm_layer = "batch_norm"
    cfg.decoder_act_layer = "relu"
    cfg.decoder_attention_type = None
    cfg.decoder_center_block = False
    cfg.decoder_separable_conv = False
    cfg.decoder_use_res_conv = False
    cfg.decoder_single_block = False
    cfg.decoder_block_kernel_size = None
    cfg.decoder_block_exp_factor = None
    cfg.decoder_block_use_res_conv = False
    cfg.decoder_post_blocks_enable = False
    cfg.use_psp = False
    cfg.psp_pool_sizes = None
    cfg.psp_out_channels = None
    cfg.atrous_rates = None
    cfg.aspp_separable = False
    cfg.aspp_dropout = 0.0
    cfg.seg_dropout = 0.0
    cfg.use_conv_transpose = False
    cfg.load_pretrained_encoder = None
    cfg.load_pretrained_decoder = None
    cfg.load_pretrained_model = None
    cfg.load_pretrained_weights = None
    cfg.ignore_weights = None
    cfg.load_weights_strict = False
    cfg.load_segmentation_head_classes = None
    cfg.num_sample_classes = None
    cfg.ignore_background_class_for_sampling = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.enable_gradient_checkpointing = False
    cfg.backbone_img_size = False
    cfg.label_dtype = None
    cfg.mix_proba = None
    cfg.mixup = None
    cfg.cutmix = None
    cfg.mixup_weight = 0.5
    cfg.cutmix_weight = 0.5
    cfg.use_sliding_window_inference = False
    cfg.sliding_window_overlap = 0.25
    cfg.log_grad_norm = False
    return cfg
