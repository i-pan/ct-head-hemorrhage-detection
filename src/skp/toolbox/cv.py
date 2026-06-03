"""Cross-validation helpers for grouped datasets."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def create_double_cv(
    df: pd.DataFrame,
    id_column: str,
    num_inner: int,
    num_outer: int,
    stratified: Optional[str] = None,
    seed: int = 88,
) -> pd.DataFrame:
    """Create nested grouped cross-validation columns.

    The returned dataframe contains ``outer``, one ``inner{outer_fold}`` column
    per outer fold, and a convenience ``fold`` column equal to ``outer``.
    """
    np.random.seed(seed)
    df = df.reset_index(drop=True).copy()
    df["outer"] = -1
    kfold_class = GroupKFold if stratified is None else StratifiedGroupKFold
    split_target = id_column if stratified is None else stratified

    outer_kfold = kfold_class(n_splits=num_outer)
    outer_split = outer_kfold.split(
        X=df[id_column], y=df[split_target], groups=df[id_column]
    )
    for outer_fold, (_, outer_valid) in enumerate(outer_split):
        df.loc[outer_valid, "outer"] = outer_fold
        inner_col = f"inner{outer_fold}"
        df[inner_col] = -1
        inner_df = df[df.outer != outer_fold].reset_index(drop=True)
        inner_kfold = kfold_class(n_splits=num_inner)
        inner_split = inner_kfold.split(
            X=inner_df[id_column],
            y=inner_df[split_target],
            groups=inner_df[id_column],
        )
        for inner_fold, (_, inner_valid) in enumerate(inner_split):
            inner_valid_ids = inner_df.loc[inner_valid, id_column].tolist()
            df.loc[df[id_column].isin(inner_valid_ids), inner_col] = inner_fold

    _validate_double_cv(df, id_column)
    df["fold"] = df["outer"]
    return df


def _validate_double_cv(df: pd.DataFrame, id_column: str) -> None:
    for outer_fold in df.outer.unique():
        train_df = df.loc[df.outer != outer_fold]
        valid_df = df.loc[df.outer == outer_fold]
        overlap = set(train_df[id_column]) & set(valid_df[id_column])
        if overlap:
            raise ValueError(f"Outer fold {outer_fold} has group leakage: {overlap}")

        inner_col = f"inner{outer_fold}"
        for inner_fold in df[inner_col].unique():
            inner_train = train_df[train_df[inner_col] != inner_fold]
            inner_valid = train_df[train_df[inner_col] == inner_fold]
            overlap = set(inner_train[id_column]) & set(inner_valid[id_column])
            if overlap:
                raise ValueError(
                    f"Inner fold {inner_fold} has group leakage: {overlap}"
                )
        if valid_df[inner_col].unique().tolist() != [-1]:
            raise ValueError(f"Outer validation rows must be -1 in {inner_col}.")
