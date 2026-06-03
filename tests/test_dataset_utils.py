import pandas as pd
import pytest

from skp.configs import Config
from skp.datasets._utils import (
    filter_by_mode,
    get_transforms,
    read_annotations,
    validate_mode,
)


def test_validate_mode_accepts_test_alias():
    validate_mode("test")
    with pytest.raises(ValueError, match="mode"):
        validate_mode("bad")


def test_read_annotations_infers_csv_json_and_jsonl(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "fold": [0, 1]})

    csv_path = tmp_path / "ann.csv"
    json_path = tmp_path / "ann.json"
    jsonl_path = tmp_path / "ann.jsonl"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")
    df.to_json(jsonl_path, orient="records", lines=True)

    assert read_annotations(Config(annotations_file=csv_path)).shape == (2, 2)
    assert read_annotations(Config(annotations_file=json_path)).shape == (2, 2)
    assert read_annotations(Config(annotations_file=jsonl_path)).shape == (2, 2)


def test_filter_by_mode_treats_val_test_consistently():
    df = pd.DataFrame({"fold": [0, 1, 0, 1]})
    cfg = Config(fold=1, split_column="fold")

    assert filter_by_mode(df, cfg, "train")["fold"].tolist() == [0, 0]
    assert filter_by_mode(df, cfg, "val")["fold"].tolist() == [1, 1]
    assert filter_by_mode(df, cfg, "test")["fold"].tolist() == [1, 1]
    assert filter_by_mode(df, cfg, "inference").shape[0] == 4


def test_filter_by_mode_supports_fixed_splits():
    df = pd.DataFrame(
        {
            "split": ["train", "train", "valid", "val", "test"],
            "x": [0, 1, 2, 3, 4],
        }
    )
    cfg = Config(
        split_column="split",
        train_split_values="train",
        val_split_values=["val", "valid"],
        test_split_values="test",
    )

    assert filter_by_mode(df, cfg, "train")["x"].tolist() == [0, 1]
    assert filter_by_mode(df, cfg, "val")["x"].tolist() == [2, 3]
    assert filter_by_mode(df, cfg, "test")["x"].tolist() == [4]


def test_filter_by_mode_requires_explicit_split_column():
    df = pd.DataFrame({"fold": [0, 1, 0, 1]})
    cfg = Config(fold=1)

    with pytest.raises(ValueError, match="split_column"):
        filter_by_mode(df, cfg, "train")


def test_filter_by_mode_rejects_missing_split_column():
    df = pd.DataFrame({"fold": [0, 1, 0, 1]})
    cfg = Config(fold=1, split_column="split")

    with pytest.raises(ValueError, match="not present"):
        filter_by_mode(df, cfg, "train")


def test_filter_by_mode_rejects_incomplete_fixed_splits():
    df = pd.DataFrame({"split": ["train", "val", "test"]})
    cfg = Config(
        split_column="split",
        train_split_values="train",
        val_split_values="val",
    )

    with pytest.raises(ValueError, match="train, val, and test"):
        filter_by_mode(df, cfg, "train")


def test_filter_by_mode_rejects_empty_selected_split():
    df = pd.DataFrame({"split": ["train", "val", "test"]})
    cfg = Config(
        split_column="split",
        train_split_values="train",
        val_split_values="valid",
        test_split_values="test",
    )

    with pytest.raises(ValueError, match="No rows selected"):
        filter_by_mode(df, cfg, "val")


def test_get_transforms_prefers_inference_transforms():
    cfg = Config(
        train_transforms="train",
        val_transforms="val",
        inference_transforms="infer",
    )

    assert get_transforms(cfg, "train") == "train"
    assert get_transforms(cfg, "val") == "val"
    assert get_transforms(cfg, "test") == "val"
    assert get_transforms(cfg, "inference") == "infer"
