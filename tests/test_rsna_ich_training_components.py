import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import cv2

from skp.configs import Config
from skp.datasets.rsna_ich_2p5d import Dataset, DepthFlip, _slice_sort_key
from skp.losses.ich import WeightedBCEWithLogitsLoss
from skp.metrics.ich import SeriesAUROC, SliceAUROC
from skp.models.normalization import normalize_input
from skp.models.classification.efficientnet3d_stem import Conv3dTo2dStem, Net
from skp.models.segmentation.unet_cls import Net as JointSegClsNet


LABEL_COLUMNS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


def _write_png(path, value, shape=(4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, value, dtype=np.uint16)
    assert cv2.imwrite(str(path), image)


def _make_rsna_fixture(tmp_path):
    data_dir = tmp_path / "rsna"
    rows = []
    label_rows = []
    rescale_rows = []
    series_specs = [
        ("p0", "study0", "series0", "train", 0, -100),
        ("p1", "study1", "series1", "train", 1, -100),
        ("p2", "study2", "series2", "test", None, -100),
        ("p3", "study3", "series3", "excluded_bhsd", None, -100),
    ]
    for patient, study, series, split, fold, intercept in series_specs:
        rescale_rows.append(
            {
                "patient_id": patient,
                "study_id": study,
                "series_id": series,
                "plane": "axial",
                "rescale_slope": 1.0,
                "rescale_intercept": intercept,
            }
        )
        for idx, value in enumerate([100, 140, 180], start=1):
            filename = f"IM{idx:04d}_000.png"
            rel_path = f"images/{patient}/{study}/{series}/{filename}"
            _write_png(data_dir / rel_path, value)
            rows.append(
                {
                    "patient_id": patient,
                    "study_id": study,
                    "series_id": series,
                    "slice_path": rel_path,
                    "slice_id": filename.removesuffix(".png"),
                    "excluded_bhsd": split == "excluded_bhsd",
                    "bhsd_match_level": "study_id" if split == "excluded_bhsd" else "",
                    "split": split,
                    "fold": fold,
                    "initial_test_study_sample": split == "test",
                }
            )
            labels = {name: 0 for name in LABEL_COLUMNS}
            labels["any"] = int(idx == 2)
            labels["subdural"] = int(idx == 2)
            label_rows.append(
                {
                    "patient_ID": patient,
                    "study_ID": study,
                    "series_ID": series,
                    "filename": filename,
                    **labels,
                    "epidural_subdural": 0,
                    "any_hemorrhage": labels["any"],
                }
            )

    split_csv = tmp_path / "splits.csv"
    labels_csv = tmp_path / "slice_labels.csv"
    rescale_csv = tmp_path / "rescale_values.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    pd.DataFrame(label_rows).to_csv(labels_csv, index=False)
    pd.DataFrame(rescale_rows).to_csv(rescale_csv, index=False)
    return data_dir, split_csv, labels_csv, rescale_csv


def _dataset_cfg(tmp_path):
    data_dir, split_csv, labels_csv, rescale_csv = _make_rsna_fixture(tmp_path)
    return Config(
        data_dir=str(data_dir),
        annotations_file=str(split_csv),
        labels_file=str(labels_csv),
        rescale_file=str(rescale_csv),
        label_columns=LABEL_COLUMNS,
        ct_windows=[(40, 80), (80, 200), (600, 2800)],
        image_height=4,
        image_width=4,
        num_slices=3,
        fold=0,
        train_transforms=None,
        val_transforms=None,
        inference_transforms=None,
    )


def test_rsna_ich_dataset_filters_modes_and_builds_2p5d_input(tmp_path):
    cfg = _dataset_cfg(tmp_path)

    train = Dataset(cfg, "train")
    val = Dataset(cfg, "val")
    test = Dataset(cfg, "test")

    assert len(train) == 3
    assert len(val) == 3
    assert len(test) == 3
    assert train.df["patient_id"].unique().tolist() == ["p1"]
    assert val.df["patient_id"].unique().tolist() == ["p0"]
    assert "p3" not in set(Dataset(cfg, "inference").df["patient_id"])

    sample = val[0]
    assert sample["x"].shape == (3, 3, 4, 4)
    assert sample["y"].tolist() == [0, 0, 0, 0, 0, 0]
    assert torch.all(sample["x"][:, 0] == 0)
    assert sample["group_index"].item() == 0

    brain_window_middle = sample["x"][0, 1]
    assert torch.allclose(brain_window_middle, torch.zeros(4, 4))
    next_sample = val[1]
    assert torch.allclose(next_sample["x"][0, 1], torch.full((4, 4), 0.5))


def test_rsna_ich_dataset_can_train_on_positive_series_only(tmp_path):
    cfg = _dataset_cfg(tmp_path)
    splits = pd.read_csv(cfg.annotations_file)
    labels = pd.read_csv(cfg.labels_file)

    splits.loc[splits["patient_id"] == "p2", ["split", "fold"]] = ["train", 1]
    labels.loc[labels["patient_ID"] == "p2", LABEL_COLUMNS] = 0
    splits.to_csv(cfg.annotations_file, index=False)
    labels.to_csv(cfg.labels_file, index=False)

    cfg.train_positive_series_only = True
    train = Dataset(cfg, "train")

    assert train.df["patient_id"].unique().tolist() == ["p1"]
    assert len(train) == 3


def test_slice_sort_key_uses_natural_numeric_ordering():
    filenames = ["slice10.png", "slice2.png", "slice1.png", "IM0003_000.png"]

    assert sorted(filenames, key=_slice_sort_key) == [
        "slice1.png",
        "slice2.png",
        "IM0003_000.png",
        "slice10.png",
    ]


def test_depth_flip_reverses_slice_axis_without_mixing_window_channels():
    x = np.array(
        [
            [[0, 1, 10, 11, 20, 21]],
        ],
        dtype=np.float32,
    )
    transform = DepthFlip(depth=3, channels_per_slice=2, p=1.0)

    out = transform(image=x)["image"]

    assert out.tolist() == [[[20, 21, 10, 11, 0, 1]]]


def test_rsna_ich_dataset_applies_depth_flip_after_transforms(tmp_path):
    cfg = _dataset_cfg(tmp_path)
    cfg.depth_flip_p = 1.0
    cfg.horizontal_flip_p = 0.0
    cfg.vertical_flip_p = 0.0

    sample = Dataset(cfg, "train")[0]

    assert torch.all(sample["x"][:, 0] != 0)
    assert torch.all(sample["x"][:, 2] == 0)


def test_rsna_ich_dataset_can_emit_single_windowed_slice(tmp_path):
    cfg = _dataset_cfg(tmp_path)
    cfg.num_slices = 1

    sample = Dataset(cfg, "train")[1]

    assert sample["x"].shape == (3, 4, 4)
    assert torch.allclose(sample["x"][0], torch.full((4, 4), 0.5))


def test_rsna_ich_dataset_can_flatten_2p5d_slices_to_channels(tmp_path):
    cfg = _dataset_cfg(tmp_path)
    cfg.num_slices = 3
    cfg.flatten_depth_to_channels = True

    sample = Dataset(cfg, "train")[1]

    assert sample["x"].shape == (9, 4, 4)
    assert torch.allclose(sample["x"][0], torch.zeros(4, 4))
    assert torch.allclose(sample["x"][3], torch.full((4, 4), 0.5))
    assert torch.allclose(sample["x"][6], torch.ones(4, 4))


def test_linear_normalization_names_input_and_output_bounds_explicitly():
    x = torch.tensor([0.0, 0.5, 1.0])

    assert torch.allclose(
        normalize_input(
            x,
            Config(
                normalization="linear",
                normalization_params={
                    "input_min": 0.0,
                    "input_max": 1.0,
                    "output_min": -1.0,
                    "output_max": 1.0,
                },
            ),
        ),
        torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_legacy_minmax_normalization_aliases_still_work():
    x = torch.tensor([0.0, 0.5, 1.0])

    assert torch.allclose(
        normalize_input(
            x,
            Config(
                normalization="-1_1",
                normalization_params={"min": 0.0, "max": 1.0},
            ),
        ),
        torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_weighted_bce_loss_normalizes_by_class_weights():
    loss_fn = WeightedBCEWithLogitsLoss(
        {"class_names": LABEL_COLUMNS, "class_weights": [1, 1, 1, 1, 1, 5]}
    )
    out = {"logits": torch.zeros(2, 6)}
    batch = {"y": torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]).float()}

    losses = loss_fn(out, batch)
    expected = F.binary_cross_entropy_with_logits(
        out["logits"],
        batch["y"],
        reduction="none",
    ).mean(dim=0)
    weighted = (expected * torch.tensor([1, 1, 1, 1, 1, 5])).sum() / 10

    assert losses["loss"] == pytest.approx(weighted)
    assert losses["any_loss"] == pytest.approx(expected[-1])


def test_slice_and_series_auc_metrics():
    cfg = Config(
        label_columns=LABEL_COLUMNS,
        metric_activation_fn="sigmoid",
        series_metric_aggregations=["max", "mean", "top3_mean"],
    )
    logits = torch.tensor(
        [
            [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ]
    )
    targets = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    ).float()
    batch = {
        "y": targets,
        "group_index": torch.tensor([0, 0, 1, 1]),
    }
    out = {"logits": logits}

    slice_auc = SliceAUROC(cfg)
    series_auc = SeriesAUROC(cfg)
    slice_auc.update(out, batch)
    series_auc.update(out, batch)

    assert slice_auc.compute()["auc_any"] == pytest.approx(torch.tensor(1.0))
    series_metrics = series_auc.compute()
    assert series_metrics["series_auc_any_max"] == pytest.approx(torch.tensor(1.0))
    assert series_metrics["series_auc_any_top3_mean"] == pytest.approx(
        torch.tensor(1.0)
    )


def test_3d_stem_adapts_conv2d_weights_and_collapses_depth():
    conv2d = torch.nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        conv2d.weight.fill_(3.0)
    stem = Conv3dTo2dStem(conv2d, depth=3)

    assert stem.conv.weight.shape == (2, 3, 3, 3, 3)
    assert torch.allclose(stem.conv.weight, torch.ones_like(stem.conv.weight))
    out = stem(torch.ones(1, 3, 3, 8, 8))
    assert out.shape == (1, 2, 8, 8)


def test_efficientnet_3d_stem_model_forward_shape():
    cfg = Config(
        backbone="tf_efficientnetv2_b0",
        pretrained=False,
        num_input_channels=3,
        num_slices=3,
        image_height=64,
        image_width=64,
        num_classes=6,
        pool="avg",
        pool_params=None,
        dropout=0.2,
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
        },
        enable_gradient_checkpointing=False,
        model_activation_fn=None,
        load_pretrained_backbone=None,
        load_pretrained_model=None,
    )
    model = Net(cfg)
    out = model({"x": torch.rand(2, 3, 3, 64, 64)})

    assert out["logits"].shape == (2, 6)


def test_joint_seg_cls_model_can_skip_segmentation_output_for_validation():
    cfg = Config(
        backbone="tf_efficientnetv2_b0",
        pretrained=False,
        num_input_channels=9,
        image_height=64,
        image_width=64,
        num_classes=6,
        cls_num_classes=6,
        decoder_type="DeepLabV3PlusDecoder",
        decoder_out_channels=64,
        decoder_norm_layer="bn",
        decoder_act_layer="relu",
        decoder_attention_type=None,
        decoder_center_block=False,
        use_psp=False,
        aspp_separable=True,
        aspp_dropout=0.0,
        atrous_rates=(6, 12, 18),
        seg_dropout=0.0,
        cls_dropout=0.2,
        pool="avg",
        pool_params=None,
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
        },
        backbone_img_size=False,
        enable_gradient_checkpointing=False,
        deep_supervision=False,
        output_size=None,
        load_pretrained_encoder=None,
        load_pretrained_decoder=None,
        load_pretrained_model=None,
        load_pretrained_segmenter=None,
        load_pretrained_classifier_head=None,
        freeze_classifier=False,
        freeze_encoder=False,
        freeze_decoder=False,
    )
    model = JointSegClsNet(cfg)
    batch = {"seg": {"x": torch.rand(2, 9, 64, 64)}}

    out = model(batch, return_loss=False, return_seg=False)

    assert "seg" not in out
    assert out["cls"]["logits"].shape == (2, 6)


def test_joint_seg_cls_model_can_freeze_classifier_and_encoder():
    cfg = Config(
        backbone="tf_efficientnetv2_b0",
        pretrained=False,
        num_input_channels=9,
        image_height=64,
        image_width=64,
        num_classes=6,
        cls_num_classes=6,
        decoder_type="DeepLabV3PlusDecoder",
        decoder_out_channels=64,
        decoder_norm_layer="bn",
        decoder_act_layer="relu",
        decoder_attention_type=None,
        decoder_center_block=False,
        use_psp=False,
        aspp_separable=True,
        aspp_dropout=0.0,
        atrous_rates=(6, 12, 18),
        seg_dropout=0.0,
        cls_dropout=0.2,
        pool="avg",
        pool_params=None,
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
        },
        backbone_img_size=False,
        enable_gradient_checkpointing=False,
        deep_supervision=False,
        output_size=None,
        load_pretrained_encoder=None,
        load_pretrained_decoder=None,
        load_pretrained_model=None,
        load_pretrained_segmenter=None,
        load_pretrained_classifier_head=None,
        freeze_classifier=True,
        freeze_encoder=True,
        freeze_decoder=False,
    )
    model = JointSegClsNet(cfg)

    assert not any(p.requires_grad for p in model.segmenter.encoder.parameters())
    assert not any(p.requires_grad for p in model.classifier.parameters())
    assert all(p.requires_grad for p in model.segmenter.decoder.parameters())
    assert all(p.requires_grad for p in model.segmenter.segmentation_head.parameters())
