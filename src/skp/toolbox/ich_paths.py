"""Portable ICH recipe paths. Historical defaults preserve existing runs."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def path(key, default):
    return Path(os.environ.get(key, str(default))).expanduser().resolve()


def registry():
    name = os.environ.get("ICH_CHECKPOINT_REGISTRY")
    return json.loads(Path(name).read_text()) if name else {}


def classifier(family, default):
    return Path(registry().get("classifiers", {}).get(family, str(default))).resolve()


def teacher_dir(family, fold, default):
    roots = registry().get("teachers", {})
    return Path(roots[family]) / f"fold{fold}" if family in roots else default


def assert_teacher_isolation(manifest, splits):
    heldout = set(
        splits.loc[splits.split.isin(["val", "test"]), "patient_id"].astype(str)
    )
    overlap = set(manifest.patient_id.astype(str)) & heldout
    if overlap:
        raise ValueError(
            f"Teacher cohort overlaps {len(overlap)} held-out patients; regenerate patient-safe splits or remove these patients before teacher training."
        )
