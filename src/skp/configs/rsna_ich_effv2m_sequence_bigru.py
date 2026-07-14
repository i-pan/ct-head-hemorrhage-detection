from skp.configs import Config
from skp.configs.defaults import dataloader_defaults, runtime_defaults


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)

cfg.project = "rsna_ich"
cfg.task = "classification"
cfg.model = "classification.ich_sequence"
cfg.dataset = "ich_sequence_features"
cfg.feature_dir = "./data/features/rsna_ich_effv2m_9ch"
cfg.feature_dim = 1280
cfg.num_classes = 6
cfg.label_columns = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]

cfg.max_sequence_length = 64
cfg.sequence_projection_dim = 512
cfg.sequence_hidden_dim = 256
cfg.sequence_num_layers = 2
cfg.sequence_dropout = 0.2
cfg.sequence_architecture = "gru"
cfg.transformer_num_heads = 8
cfg.transformer_feedforward_dim = 1024
cfg.attention_dim = 128
cfg.dropout = 0.2
cfg.feature_noise_std = 0.02
cfg.feature_dropout = 0.05
cfg.slice_feature_dropout = 0.02
cfg.sequence_reverse_p = 0.0

cfg.loss = "ich_sequence.SequenceClassificationLoss"
cfg.loss_params = {
    "class_names": cfg.label_columns,
    "class_weights": [1, 1, 1, 1, 1, 2],
    "slice_weight": 1.0,
    "series_weight": 0.5,
    "mil_weight": 0.25,
}

cfg.batch_size = 128
cfg.val_batch_size = 256
cfg.num_workers = 4
cfg.val_num_workers = 2
cfg.prefetch_factor = 4
cfg.val_prefetch_factor = 2
cfg.num_epochs = 20
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

cfg.metrics = [
    "ich_sequence.ContextualSliceAUROC",
    "ich_sequence.ContextualSeriesAUROC",
    "ich_sequence.MILSeriesAUROC",
    "ich_sequence.BaselineSliceAUROC",
    "ich_sequence.BaselineMaxSeriesAUROC",
]
cfg.val_metric = "slice_auc_any"
cfg.val_track = "max"
