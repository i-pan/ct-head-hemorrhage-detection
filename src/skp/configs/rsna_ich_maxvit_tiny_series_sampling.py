"""MaxViT slice classification: one center slice per series per epoch.

Positive/negative bins cycle independently without replacement. Context remains
three axial slices × three CT windows; labels belong to the sampled center slice.
Override num_epochs (50/75/100) and series_positive_slice_probability on the CLI.
The step schedule is derived from the series sampler, never the full slice table.
"""

from pathlib import Path
from skp.configs.rsna_ich_maxvit_tiny_fixedval_9ch import cfg as base_cfg

cfg = base_cfg.__deepcopy__()
cfg.sampler = "SeriesSliceSampler"
cfg.series_positive_slice_probability = 0.5
cfg.train_drop_last = False
cfg.num_epochs = 50
cfg.val_metric = "series_auc_any_max"
cfg.val_track = "max"
cfg.scheduler_params = dict(base_cfg.scheduler_params)
cfg.num_workers = 6
cfg.prefetch_factor = 2
cfg.seed = 88
cfg.mlflow_tracking_uri = f"sqlite:///{Path(__file__).resolve().parents[3] / 'experiments/series_sampling_mlflow.db'}"
