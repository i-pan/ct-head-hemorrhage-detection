"""Real manifest/soft-mask checks, run only after complete pseudo-label export."""

import json
import copy
import pickle
import numpy as np
import torch
from skp.configs.rsna_ich_convnext_tiny_s16_auxiliary import cfg
from skp.datasets.rsna_ich_pseudolabel import Dataset
from skp.datasets.rsna_ich_2p5d import Dataset as BaseDataset
from skp.tasks.samplers import configure_series_sampling_schedule
import pandas as pd


def main():
    torch.set_num_threads(4)
    from skp.configs.rsna_ich_convnext_tiny_logitmax_s16_series_sampling import cfg as saved
    for key in [
        "batch_size",
        "val_batch_size",
        "accumulate_grad_batches",
        "num_workers",
        "prefetch_factor",
        "seed",
        "num_epochs",
        "optimizer",
        "optimizer_params",
        "scheduler",
        "scheduler_params",
        "pretrained",
        "series_positive_slice_probability",
        "label_columns",
        "ct_windows",
        "dropout",
        "normalization",
        "normalization_params",
        "loss",
        "loss_params",
        "val_metric",
    ]:
        assert cfg.get(key) == saved.get(key), key
    train = Dataset(cfg, "train")
    baseline = BaseDataset(cfg, "train")
    pd.testing.assert_frame_equal(train.df, baseline.df)
    cfg.world_size = 2
    cfg.args = {"strategy": "ddp"}
    configure_series_sampling_schedule(cfg, train)
    assert cfg.scheduler_total_steps == cfg.optimizer_steps_per_epoch * cfg.num_epochs
    validation = Dataset(cfg, "val")
    assert set(train.df.patient_id).isdisjoint(validation.df.patient_id)
    negative = np.flatnonzero(train.df["any"].to_numpy() == 0)
    positive = np.flatnonzero(train.df["any"].to_numpy() > 0)
    for index in [*positive[:8], *negative[:8]]:
        batch = train[int(index)]
        assert batch["x"].shape == (9, 512, 512)
        assert batch["segmentation_target"].shape == (1, 512, 512)
        target = batch["segmentation_target"]
        assert torch.isfinite(target).all() and target.min() >= 0 and target.max() <= 1
        if batch["y"][-1] == 0:
            assert target.sum() == 0
            assert batch["auxiliary_roi"].min() == 1
    missing = np.flatnonzero(
        (train.df["any"].to_numpy() > 0) & (train.pseudo_indices < 0)
    )
    assert len(missing) == 0
    supplemental = np.flatnonzero(train.pseudo_indices >= train.primary_count)
    assert len(supplemental) == len(train.df) - len(train.df[train.pseudo_indices < train.primary_count])
    for index in supplemental[:4]:
        batch = train[int(index)]
        assert batch["y"][-1] == 1 and batch["segmentation_target"].max() > 0
    # Mask additions must not change the original image augmentation path.
    baseline.transforms = copy.deepcopy(train.transforms)
    for i, index in enumerate([*positive[:4], *negative[:4]]):
        train.transforms.set_random_seed(88 + i)
        baseline.transforms.set_random_seed(88 + i)
        np.random.seed(88 + i)
        aux = train[int(index)]
        np.random.seed(88 + i)
        original = baseline[int(index)]
        torch.testing.assert_close(aux["x"], original["x"], rtol=0, atol=0)
        torch.testing.assert_close(aux["y"], original["y"], rtol=0, atol=0)
    train.use_roi = True
    for index in [*positive[:4], *negative[:4], *supplemental[:2]]:
        batch = train[int(index)]
        assert batch["auxiliary_roi"].shape == (1, 512, 512)
        assert ((batch["auxiliary_roi"] == 0) | (batch["auxiliary_roi"] == 1)).all()
    val = validation[0]
    assert "segmentation_target" not in val
    print(
        json.dumps(
            dict(
                train_slices=len(train),
                positive_slices=len(positive),
                classification_only_positive_slices=len(missing),
                series=train.df.series_uid.nunique(),
                validation_slices=len(validation),
                checks="full positive coverage,negative empty masks,aligned transforms,held-out patient isolation passed",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
