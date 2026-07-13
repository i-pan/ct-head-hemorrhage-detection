from pathlib import Path

import numpy as np
import pandas as pd
import torch

from skp.configs import Config
from skp.datasets.ich_sequence_features import Dataset
from skp.losses.ich_sequence import SequenceClassificationLoss
from skp.models.classification.ich_sequence import Net


def make_feature_store(root: Path) -> None:
    features = np.arange(8 * 4, dtype=np.float16).reshape(8, 4)
    logits = np.arange(8 * 2, dtype=np.float16).reshape(8, 2)
    labels = np.zeros((8, 2), dtype=np.uint8)
    labels[2, 1] = 1
    labels[7, 0] = 1
    series = pd.DataFrame(
        {
            "series_index": [0, 1],
            "series_uid": ["p/s/a", "p/s/b"],
            "patient_id": ["p", "p"],
            "study_id": ["s", "s"],
            "series_id": ["a", "b"],
            "offset": [0, 3],
            "length": [3, 5],
        }
    )
    for split in ["train", "val"]:
        np.save(root / f"{split}_features.npy", features)
        np.save(root / f"{split}_base_logits.npy", logits)
        np.save(root / f"{split}_labels.npy", labels)
        series.to_csv(root / f"{split}_series.csv", index=False)


def dataset_cfg(root: Path) -> Config:
    cfg = Config()
    cfg.feature_dir = str(root)
    cfg.max_sequence_length = 4
    cfg.sequence_reverse_p = 0.0
    return cfg


def model_cfg() -> Config:
    cfg = Config()
    cfg.feature_dim = 4
    cfg.sequence_projection_dim = 6
    cfg.sequence_hidden_dim = 3
    cfg.sequence_num_layers = 2
    cfg.sequence_dropout = 0.2
    cfg.attention_dim = 4
    cfg.num_classes = 2
    cfg.dropout = 0.1
    cfg.feature_noise_std = 0.0
    cfg.feature_dropout = 0.0
    cfg.slice_feature_dropout = 0.0
    return cfg


def test_dataset_pads_and_resamples_series(tmp_path):
    make_feature_store(tmp_path)
    dataset = Dataset(dataset_cfg(tmp_path), "train")

    short = dataset[0]
    assert short["length"].item() == 3
    assert short["valid_mask"].tolist() == [True, True, True, False]
    assert short["series_y"].tolist() == [0.0, 1.0]

    long = dataset[1]
    assert long["length"].item() == 4
    assert long["original_length"].item() == 5
    assert long["source_index"].tolist() == [0, 1, 3, 4]
    assert long["series_y"].tolist() == [1.0, 0.0]


def test_restore_predictions_uses_nearest_sample():
    sampled = np.array([0, 2, 5])
    predictions = np.array([[0.0], [2.0], [5.0]])
    restored = Dataset.restore_predictions(predictions, sampled, original_length=6)
    assert restored[:, 0].tolist() == [0.0, 0.0, 2.0, 2.0, 5.0, 5.0]


def test_model_masks_padding_and_loss_is_finite(tmp_path):
    make_feature_store(tmp_path)
    dataset = Dataset(dataset_cfg(tmp_path), "val")
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }
    model = Net(model_cfg())
    criterion = SequenceClassificationLoss(
        {
            "class_names": ["subtype", "any"],
            "class_weights": [1, 2],
            "slice_weight": 1.0,
            "series_weight": 0.5,
            "mil_weight": 0.25,
        }
    )
    model.set_criterion(criterion)
    model.eval()
    out = model(batch, return_loss=True)

    assert out["slice_logits"].shape == (2, 4, 2)
    assert out["series_logits"].shape == (2, 2)
    assert out["mil_logits"].shape == (2, 2)
    assert torch.isfinite(out["loss"])
    assert torch.all(out["sequence_attention"][0, 3] == 0)
    assert torch.all(out["mil_attention"][0, 3] == 0)
    torch.testing.assert_close(
        out["sequence_attention"].sum(dim=1), torch.ones(2, 2)
    )
