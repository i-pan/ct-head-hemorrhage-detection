from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from torch.utils.data import default_collate


VALID_MODES = {"train", "val", "test", "inference"}
FIXED_SPLIT_VALUE_KEYS = {
    "train": "train_split_values",
    "val": "val_split_values",
    "test": "test_split_values",
}


def validate_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        modes = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"mode must be one of {modes}, got {mode}")


def read_annotations(cfg, **kwargs) -> pd.DataFrame:
    path = Path(cfg.annotations_file)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz", ".zip"}:
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else ""

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix == ".txt":
        return pd.read_csv(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        kwargs.setdefault("lines", True)
        return pd.read_json(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path, **kwargs)

    supported = ".csv, .tsv, .tab, .txt, .json, .jsonl, .ndjson, .parquet, .feather, .pkl, .pickle"
    raise ValueError(
        f"Could not infer annotation file type from {path}. Supported extensions: {supported}"
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _get_split_column(df: pd.DataFrame, cfg) -> str:
    split_column = cfg.get("split_column")
    if not split_column:
        raise ValueError(
            "cfg.split_column must be set for train/val/test dataset modes. "
            "Use cfg.split_column = 'fold' for k-fold experiments or a fixed "
            "split column such as cfg.split_column = 'split'."
        )
    if split_column not in df.columns:
        raise ValueError(
            f"cfg.split_column={split_column!r} is not present in annotations. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    return split_column


def _has_fixed_split_values(cfg) -> bool:
    return any(cfg.get(key) is not None for key in FIXED_SPLIT_VALUE_KEYS.values())


def _validate_fixed_split_values(cfg) -> dict[str, list[Any]]:
    split_values = {
        mode: _as_list(cfg.get(key)) for mode, key in FIXED_SPLIT_VALUE_KEYS.items()
    }
    missing_modes = [mode for mode, values in split_values.items() if not values]
    if missing_modes:
        missing = ", ".join(missing_modes)
        raise ValueError(
            "Fixed split mode requires non-empty split values for train, val, "
            f"and test. Missing values for: {missing}."
        )

    seen: dict[Any, str] = {}
    overlaps: list[str] = []
    for mode, values in split_values.items():
        for value in values:
            if value in seen:
                overlaps.append(f"{value!r} appears in both {seen[value]} and {mode}")
            seen[value] = mode
    if overlaps:
        raise ValueError(
            "Fixed split values must be mutually exclusive: " + "; ".join(overlaps)
        )

    return split_values


def _filter_or_raise(
    df: pd.DataFrame, mask: pd.Series, *, mode: str, split_column: str
) -> pd.DataFrame:
    filtered = df[mask]
    if filtered.empty:
        available = sorted(df[split_column].dropna().unique().tolist())
        raise ValueError(
            f"No rows selected for mode={mode!r} using split_column={split_column!r}. "
            f"Available split values: {available}"
        )
    return filtered


def filter_by_mode(df: pd.DataFrame, cfg, mode: str) -> pd.DataFrame:
    validate_mode(mode)
    if mode == "inference":
        return df

    split_column = _get_split_column(df, cfg)

    if cfg.get("double_cv") is not None:
        if "outer" not in df.columns:
            raise ValueError("double_cv requires an 'outer' column in annotations.")
        inner_column = f"inner{cfg.double_cv}"
        if inner_column not in df.columns:
            raise ValueError(f"double_cv requires an {inner_column!r} column.")
        df = df.copy()
        df = df[df.outer != cfg.double_cv]
        df[split_column] = df[inner_column]

    if _has_fixed_split_values(cfg):
        split_values = _validate_fixed_split_values(cfg)
        values = split_values[mode]
        mask = df[split_column].isin(values)
        return _filter_or_raise(
            df, mask, mode=mode, split_column=split_column
        )

    fold = cfg.get("fold")
    if fold is None:
        raise ValueError(
            "K-fold split mode requires cfg.fold. For fixed train/val/test "
            "splits, set cfg.train_split_values, cfg.val_split_values, and "
            "cfg.test_split_values instead."
        )
    if mode == "train":
        mask = df[split_column] != fold
    else:
        mask = df[split_column] == fold
    return _filter_or_raise(df, mask, mode=mode, split_column=split_column)


def get_transforms(cfg, mode: str):
    validate_mode(mode)
    if mode == "train":
        return cfg.train_transforms
    if mode == "inference":
        return cfg.get("inference_transforms") or cfg.val_transforms
    return cfg.val_transforms


def get_collate_fn(
    mode: str,
    train_collate_fn: Callable = default_collate,
    eval_collate_fn: Callable = default_collate,
) -> Callable:
    validate_mode(mode)
    return train_collate_fn if mode == "train" else eval_collate_fn
