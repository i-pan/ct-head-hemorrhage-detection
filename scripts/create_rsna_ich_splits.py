#!/usr/bin/env python
"""Create patient-level RSNA ICH train folds and holdout split.

This script reads only file paths and directory identifiers. It does not open
image pixels. The expected RSNA image layout is:

    images/<patient_id>/<study_id>/<series_id>/<slice>.png

BHSD samples are excluded from train/test by matching shared identifiers found
in the BHSD path tree. The most specific available match is used per BHSD path:
slice, series, study, then patient.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RSNA_ROOT = Path("/mnt/champaca/rsna-intracranial-hemorrhage-detection-16bit-png")
DEFAULT_BHSD_ROOT = Path("/mnt/stor/datasets/BHSD")
DEFAULT_OUTPUT = Path("data/folds/rsna_ich_splits.csv")
DEFAULT_SUMMARY = Path("data/folds/rsna_ich_splits.summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsna-root", type=Path, default=DEFAULT_RSNA_ROOT)
    parser.add_argument("--bhsd-root", type=Path, default=DEFAULT_BHSD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--test-studies", type=int, default=1000)
    parser.add_argument(
        "--val-studies",
        type=int,
        default=0,
        help=(
            "If >0, sample this many eligible non-test studies and assign all "
            "studies from their patients to a fixed validation split."
        ),
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument(
        "--drop-excluded",
        action="store_true",
        help="Do not include BHSD-excluded rows in the output CSV.",
    )
    return parser.parse_args()


def scan_rsna_images(rsna_root: Path) -> pd.DataFrame:
    images_root = rsna_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"RSNA images directory not found: {images_root}")

    rows = []
    for path in sorted(images_root.glob("*/*/*/*.png")):
        rel = path.relative_to(rsna_root)
        parts = rel.parts
        if len(parts) != 5 or parts[0] != "images":
            continue
        _, patient_id, study_id, series_id, filename = parts
        rows.append(
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "series_id": series_id,
                "slice_path": rel.as_posix(),
                "slice_id": Path(filename).stem,
            }
        )

    if not rows:
        raise ValueError(f"No PNG slices found under {images_root}")
    return pd.DataFrame(rows)


def _path_tokens(path: Path) -> set[str]:
    tokens = set(path.parts)
    tokens.add(path.name)
    tokens.add(path.stem)
    for part in path.parts:
        stem = Path(part).stem
        for prefix in ["mask_", "seg_", "label_", "labels_"]:
            if stem.startswith(prefix):
                tokens.add(stem[len(prefix) :])
        for suffix in ["_mask", "_seg", "_segmentation", "_label", "_labels"]:
            if stem.endswith(suffix):
                tokens.add(stem[: -len(suffix)])
    return {token for token in tokens if token}


def collect_bhsd_matches(df: pd.DataFrame, bhsd_root: Path) -> dict[str, set[str]]:
    matches = {
        "patient_id": set(),
        "study_id": set(),
        "series_id": set(),
        "slice_id": set(),
    }
    if not bhsd_root.exists():
        raise FileNotFoundError(f"BHSD directory not found: {bhsd_root}")

    known = {
        column: set(df[column].astype(str).unique())
        for column in ["patient_id", "study_id", "series_id", "slice_id"]
    }

    patient_candidates = set()
    for csv_path in sorted(bhsd_root.glob("*.csv")):
        manifest = pd.read_csv(csv_path, dtype=str)
        if "SeriesID" in manifest.columns:
            values = set(manifest["SeriesID"].dropna().astype(str))
            matches["study_id"].update(values & known["study_id"])
            matches["series_id"].update(values & known["series_id"])
        if "PatientID" in manifest.columns:
            values = set(manifest["PatientID"].dropna().astype(str))
            patient_candidates.update(values & known["patient_id"])

        for column in ["image", "label", "slice_id"]:
            if column not in manifest.columns:
                continue
            values = set()
            for value in manifest[column].dropna().astype(str):
                path = Path(value)
                values.add(path.name)
                values.add(path.stem)
                if path.suffix == ".gz":
                    values.add(Path(path.stem).stem)
            matches["slice_id"].update(values & known["slice_id"])

    if not (matches["slice_id"] or matches["series_id"] or matches["study_id"]):
        matches["patient_id"].update(patient_candidates)

    for path in bhsd_root.rglob("*"):
        tokens = _path_tokens(path.relative_to(bhsd_root))
        path_matches = {
            column: values & tokens for column, values in known.items()
        }
        # Prefer excluding at the most specific available level for this path.
        for column in ["slice_id", "series_id", "study_id", "patient_id"]:
            if path_matches[column]:
                matches[column].update(path_matches[column])
                break

    return matches


def mark_bhsd_exclusions(
    df: pd.DataFrame, matches: dict[str, set[str]]
) -> pd.DataFrame:
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    level = pd.Series("", index=df.index, dtype=object)

    for column in ["slice_id", "series_id", "study_id", "patient_id"]:
        values = matches.get(column, set())
        if not values:
            continue
        current = df[column].astype(str).isin(values)
        newly_matched = current & ~mask
        level.loc[newly_matched] = column
        mask = mask | current

    df["excluded_bhsd"] = mask
    df["bhsd_match_level"] = level
    return df


def sample_patients_for_study_target(
    study_to_patient: pd.DataFrame,
    *,
    target_studies: int,
    rng: np.random.Generator,
) -> set[str]:
    """Sample patients so the final patient-level split is near target_studies."""
    patient_study_counts = (
        study_to_patient.groupby("patient_id")["study_id"].nunique().to_dict()
    )
    patients = np.array(sorted(patient_study_counts))
    rng.shuffle(patients)

    selected: set[str] = set()
    total_studies = 0
    for patient_id in patients:
        count = int(patient_study_counts[patient_id])
        if selected and total_studies >= target_studies:
            break
        if selected and abs(target_studies - total_studies) <= abs(
            target_studies - (total_studies + count)
        ):
            break
        selected.add(str(patient_id))
        total_studies += count
    return selected


def assign_splits(
    df: pd.DataFrame,
    *,
    test_studies: int,
    num_folds: int,
    seed: int,
    val_studies: int = 0,
) -> pd.DataFrame:
    if test_studies <= 0:
        raise ValueError("test_studies must be positive.")
    if num_folds <= 1:
        raise ValueError("num_folds must be greater than 1.")

    df = df.copy()
    df["split"] = "excluded_bhsd"
    df["fold"] = pd.NA

    eligible = df.loc[~df["excluded_bhsd"]].copy()
    study_to_patient = eligible[["study_id", "patient_id"]].drop_duplicates()
    if len(study_to_patient) < test_studies:
        raise ValueError(
            f"Requested {test_studies} test studies, but only "
            f"{len(study_to_patient)} eligible studies are available."
        )

    rng = np.random.default_rng(seed)
    sampled_studies = rng.choice(
        np.array(sorted(study_to_patient["study_id"].unique())),
        size=test_studies,
        replace=False,
    )
    test_patients = set(
        study_to_patient.loc[
            study_to_patient["study_id"].isin(sampled_studies), "patient_id"
        ]
    )

    eligible_mask = ~df["excluded_bhsd"]
    test_mask = eligible_mask & df["patient_id"].isin(test_patients)
    train_val_mask = eligible_mask & ~df["patient_id"].isin(test_patients)
    df.loc[test_mask, "split"] = "test"
    df.loc[train_val_mask, "split"] = "train"

    if val_studies > 0:
        train_val_studies = (
            df.loc[train_val_mask, ["study_id", "patient_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(train_val_studies) < val_studies:
            raise ValueError(
                f"Requested {val_studies} validation studies, but only "
                f"{len(train_val_studies)} eligible non-test studies are available."
            )
        val_patients = sample_patients_for_study_target(
            train_val_studies,
            target_studies=val_studies,
            rng=rng,
        )
        val_mask = train_val_mask & df["patient_id"].isin(val_patients)
        df.loc[val_mask, "split"] = "val"
        df["initial_val_study_sample"] = val_mask
    else:
        val_mask = pd.Series(False, index=df.index)
        df["initial_val_study_sample"] = False

    train_mask = train_val_mask & ~val_mask
    train_patients = np.array(sorted(df.loc[train_mask, "patient_id"].unique()))
    rng.shuffle(train_patients)
    patient_to_fold = {
        patient_id: idx % num_folds for idx, patient_id in enumerate(train_patients)
    }
    df.loc[train_mask, "fold"] = df.loc[train_mask, "patient_id"].map(patient_to_fold)
    df["initial_test_study_sample"] = df["study_id"].isin(sampled_studies)
    return df


def validate_split(df: pd.DataFrame, num_folds: int) -> None:
    eligible = df.loc[~df["excluded_bhsd"]]
    train = eligible.loc[eligible["split"] == "train"]
    val = eligible.loc[eligible["split"] == "val"]
    test = eligible.loc[eligible["split"] == "test"]

    split_patients = [
        ("train", set(train["patient_id"])),
        ("val", set(val["patient_id"])),
        ("test", set(test["patient_id"])),
    ]
    for (split_a, patients_a), (split_b, patients_b) in combinations(split_patients, 2):
        overlap = patients_a & patients_b
        if overlap:
            raise ValueError(
                f"Patient leakage between {split_a} and {split_b}: "
                f"{sorted(overlap)[:5]}"
            )

    if set(train["fold"].dropna().astype(int).unique()) != set(range(num_folds)):
        raise ValueError("Training folds do not cover the expected fold IDs.")

    fold_patients = defaultdict(set)
    for fold, fold_df in train.groupby("fold"):
        fold_patients[int(fold)].update(fold_df["patient_id"].unique())
    for fold_a, patients_a in fold_patients.items():
        for fold_b, patients_b in fold_patients.items():
            if fold_a >= fold_b:
                continue
            overlap = patients_a & patients_b
            if overlap:
                raise ValueError(
                    f"Patient leakage between folds {fold_a} and {fold_b}: "
                    f"{sorted(overlap)[:5]}"
                )


def summarize(
    df: pd.DataFrame,
    *,
    test_studies: int,
    val_studies: int,
    seed: int,
) -> dict:
    def counts(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "slices": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "studies": int(frame["study_id"].nunique()),
            "series": int(frame["series_id"].nunique()),
        }

    summary = {
        "seed": seed,
        "requested_initial_test_studies": test_studies,
        "requested_initial_val_studies": val_studies,
        "all": counts(df),
        "excluded_bhsd": counts(df.loc[df["excluded_bhsd"]]),
        "train": counts(df.loc[df["split"] == "train"]),
        "val": counts(df.loc[df["split"] == "val"]),
        "test": counts(df.loc[df["split"] == "test"]),
        "initial_test_study_sample_count": int(
            df.loc[df["initial_test_study_sample"], "study_id"].nunique()
        ),
        "initial_val_study_sample_count": int(
            df.loc[df["initial_val_study_sample"], "study_id"].nunique()
        ),
        "folds": {},
    }
    for fold, fold_df in df.loc[df["split"] == "train"].groupby("fold"):
        summary["folds"][str(int(fold))] = counts(fold_df)
    return summary


def main() -> None:
    args = parse_args()
    df = scan_rsna_images(args.rsna_root)
    matches = collect_bhsd_matches(df, args.bhsd_root)
    df = mark_bhsd_exclusions(df, matches)
    df = assign_splits(
        df,
        test_studies=args.test_studies,
        val_studies=args.val_studies,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    validate_split(df, args.num_folds)

    summary = summarize(
        df,
        test_studies=args.test_studies,
        val_studies=args.val_studies,
        seed=args.seed,
    )
    summary["bhsd_matches"] = {key: len(value) for key, value in matches.items()}

    output_df = df
    if args.drop_excluded:
        output_df = output_df.loc[~output_df["excluded_bhsd"]].copy()
    output_df = output_df[
        [
            "patient_id",
            "study_id",
            "series_id",
            "slice_path",
            "split",
            "fold",
            "excluded_bhsd",
        ]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote splits to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
