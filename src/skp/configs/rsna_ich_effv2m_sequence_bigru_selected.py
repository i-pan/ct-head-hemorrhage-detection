from skp.configs.rsna_ich_effv2m_sequence_bigru import cfg as base_cfg


cfg = base_cfg.__deepcopy__()


# Selected by the seeded validation sweep. Slice classification remains primary,
# so the series and MIL objectives are intentionally auxiliary.
cfg.loss_params = dict(cfg.loss_params)
cfg.loss_params.update(
    {
        "slice_weight": 1.0,
        "series_weight": 0.25,
        "mil_weight": 0.1,
    }
)

cfg.selected_sequence_checkpoint = (
    "./experiments/rsna_ich_effv2m_sequence_bigru/"
    "seqsweep_01_sliceheavy_20260713/fixed_split/checkpoints/best.ckpt"
)
cfg.source_classifier_checkpoint = (
    "./experiments/rsna_ich_effv2m_fixedval_9ch/"
    "definite-crayfish-7226/fold0/checkpoints/last.ckpt"
)
cfg.seed = 88
cfg.training_devices = 2
cfg.training_strategy = "ddp"
cfg.training_precision = "bf16-mixed"
cfg.blend_space = "logit"
cfg.series_baseline_aggregation = "max_probability"
cfg.slice_blend_alphas = [0.0, 0.75, 0.75, 1.0, 0.75, 1.0]
cfg.series_blend_alphas = [0.0, 0.5, 0.5, 1.0, 0.5, 0.5]
cfg.mil_blend_alphas = [0.0, 1.0, 0.5, 0.75, 1.0, 0.5]
