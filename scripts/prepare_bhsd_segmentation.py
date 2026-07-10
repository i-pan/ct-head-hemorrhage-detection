#!/usr/bin/env python
"""Prepare BHSD masks and folds for ICH segmentation experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BHSD_ROOT = Path("/mnt/stor/datasets/BHSD")
OUTPUT_DIR = Path("data/bhsd")
EXAMPLES_DIR = Path("diagnostics/bhsd_orientation_examples")
MANIFEST_NAME = "bhsd_segmentation_slices.csv"
SUMMARY_NAME = "bhsd_segmentation_summary.json"

LABEL_COLUMNS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
MASK_VALUE_TO_CLASS = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}
CLASS_COLORS = {
    1: (255, 80, 80),
    2: (80, 220, 120),
    3: (80, 170, 255),
    4: (255, 220, 80),
    5: (220, 120, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhsd-root", type=Path, default=BHSD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--examples-dir", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--num-examples", type=int, default=15)
    parser.add_argument(
        "--skip-mask-volumes",
        action="store_true",
        help="Only write the manifest/examples; do not save local .npy mask volumes.",
    )
    return parser.parse_args()


def require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nibabel is required to read BHSD NIfTI files. Run with "
            "`uv run --with nibabel scripts/prepare_bhsd_segmentation.py` "
            "or install the medical dependency group."
        ) from e
    return nib


def reorient_nifti_data(x: np.ndarray) -> np.ndarray:
    """Match the orientation used by the BHSD PNG exports.

    The transform is verified against the released per-slice PNG labels before
    any manifest is written.
    """
    x = x[:, :, ::-1]
    x = np.rot90(x, axes=(0, 1))
    x = x[:, ::-1]
    return np.ascontiguousarray(x)


def parse_bhsd_slice_name(name: str) -> tuple[str, int]:
    match = re.match(r"(.+)_([0-9]+)_(?:image|label)\.png$", name)
    if match is None:
        raise ValueError(f"Could not parse BHSD slice filename: {name}")
    return match.group(1), int(match.group(2))


def parse_image_2dc(value: str) -> list[str]:
    value = str(value).strip()
    if value.startswith("["):
        return [part.strip(" '\"") for part in value.strip("[]").split(",")]
    return [part.strip() for part in value.split(",")]


def verify_orientation(
    df: pd.DataFrame,
    *,
    bhsd_root: Path,
    nib,
) -> dict[str, int | float]:
    label_root = bhsd_root / "label_192" / "labels"
    png_roots = [bhsd_root / "png_3ch", bhsd_root / "png"]
    checked = 0
    exact = 0
    min_iou = 1.0
    cache: dict[str, np.ndarray] = {}

    positives = df.loc[df["label"] != "empty", ["label"]].drop_duplicates()
    for label_name in positives["label"]:
        series_uid, slice_index = parse_bhsd_slice_name(label_name)
        if series_uid not in cache:
            label_volume = nib.load(str(label_root / f"{series_uid}.nii.gz")).get_fdata()
            cache[series_uid] = reorient_nifti_data(label_volume).astype(np.uint8)

        png_path = next(
            (root / label_name for root in png_roots if (root / label_name).exists()),
            None,
        )
        if png_path is None:
            raise FileNotFoundError(f"Missing BHSD PNG label for {label_name}")
        png_mask = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if png_mask is None:
            raise FileNotFoundError(f"Could not read PNG label: {png_path}")

        nifti_mask = cache[series_uid][..., slice_index]
        checked += 1
        if np.array_equal(nifti_mask, png_mask):
            exact += 1
        intersection = np.logical_and(nifti_mask > 0, png_mask > 0).sum()
        union = np.logical_or(nifti_mask > 0, png_mask > 0).sum()
        iou = float(intersection / union) if union else 1.0
        min_iou = min(min_iou, iou)

    if exact != checked:
        raise RuntimeError(
            "BHSD NIfTI orientation verification failed: "
            f"{exact}/{checked} labels matched exactly, min IoU={min_iou:.6f}."
        )
    return {"checked_positive_labels": checked, "exact_matches": exact, "min_iou": min_iou}


def assign_patient_folds(
    df: pd.DataFrame, *, num_folds: int, seed: int
) -> dict[str, int]:
    patient_stats = (
        df.groupby("patient_id")
        .agg(num_slices=("image", "size"), positive_slices=("has_mask", "sum"))
        .reset_index()
    )
    patient_stats = patient_stats.sample(frac=1, random_state=seed)
    patient_stats = patient_stats.sort_values(
        ["num_slices", "positive_slices"], ascending=False, kind="mergesort"
    )

    fold_totals = [
        {"num_slices": 0, "positive_slices": 0, "patients": 0}
        for _ in range(num_folds)
    ]
    assignment: dict[str, int] = {}
    for row in patient_stats.itertuples(index=False):
        fold = min(
            range(num_folds),
            key=lambda idx: (
                fold_totals[idx]["num_slices"],
                fold_totals[idx]["positive_slices"],
                fold_totals[idx]["patients"],
            ),
        )
        assignment[str(row.patient_id)] = fold
        fold_totals[fold]["num_slices"] += int(row.num_slices)
        fold_totals[fold]["positive_slices"] += int(row.positive_slices)
        fold_totals[fold]["patients"] += 1
    return assignment


def build_manifest(
    *,
    bhsd_root: Path,
    output_dir: Path,
    num_folds: int,
    seed: int,
    nib,
) -> pd.DataFrame:
    csv_path = bhsd_root / "train_all_slices_png_3ch_kfold.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"BHSD slice manifest not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    df = df.rename(columns={"PatientID": "patient_id", "SeriesID": "study_id"})
    df["series_uid"] = df["image"].map(lambda name: parse_bhsd_slice_name(name)[0])
    df["slice_index"] = df["image"].map(lambda name: parse_bhsd_slice_name(name)[1])
    df["has_mask"] = df["label"] != "empty"
    df["image_path"] = df["image"].map(
        lambda name: str((bhsd_root / "png_3ch" / name).resolve())
    )
    neighbors = df["image_2dc"].map(parse_image_2dc)
    df["prev_image_path"] = neighbors.map(
        lambda names: str((bhsd_root / "png_3ch" / names[0]).resolve())
    )
    df["next_image_path"] = neighbors.map(
        lambda names: str((bhsd_root / "png_3ch" / names[-1]).resolve())
    )
    df["mask_volume_path"] = df["series_uid"].map(
        lambda uid: str((output_dir / "mask_volumes" / f"{uid}.npy").resolve())
    )
    spacing = {}
    for series_uid in sorted(df["series_uid"].unique()):
        image_path = bhsd_root / "label_192" / "images" / f"{series_uid}.nii.gz"
        image = nib.load(str(image_path))
        voxel_sizes = nib.affines.voxel_sizes(image.affine)[:3]
        spacing[series_uid] = {
            "row_spacing_mm": float(voxel_sizes[1]),
            "col_spacing_mm": float(voxel_sizes[0]),
            "slice_spacing_mm": float(voxel_sizes[2]),
        }
    for column in ["row_spacing_mm", "col_spacing_mm", "slice_spacing_mm"]:
        df[column] = df["series_uid"].map(
            {series_uid: values[column] for series_uid, values in spacing.items()}
        )
    patient_to_fold = assign_patient_folds(df, num_folds=num_folds, seed=seed)
    df["fold"] = df["patient_id"].map(patient_to_fold).astype(int)
    keep_columns = [
        "patient_id",
        "study_id",
        "series_uid",
        "slice_index",
        "fold",
        "has_mask",
        "image_path",
        "prev_image_path",
        "next_image_path",
        "mask_volume_path",
        "row_spacing_mm",
        "col_spacing_mm",
        "slice_spacing_mm",
        "image",
        "label",
    ]
    return df[keep_columns].sort_values(
        ["patient_id", "study_id", "slice_index"]
    ).reset_index(drop=True)


def write_mask_volumes(df: pd.DataFrame, *, bhsd_root: Path, nib) -> dict[str, dict]:
    output_paths = {
        row.series_uid: Path(row.mask_volume_path)
        for row in df[["series_uid", "mask_volume_path"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    summary = {}
    for series_uid, output_path in sorted(output_paths.items()):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        label_path = bhsd_root / "label_192" / "labels" / f"{series_uid}.nii.gz"
        mask = nib.load(str(label_path)).get_fdata()
        mask = reorient_nifti_data(mask).astype(np.uint8)
        if sorted(np.unique(mask).tolist())[-1] > 5:
            raise ValueError(f"Unexpected mask value in {label_path}")
        np.save(output_path, mask)
        summary[series_uid] = {
            "shape": list(mask.shape),
            "values": [int(v) for v in np.unique(mask).tolist()],
            "positive_voxels": int((mask > 0).sum()),
        }
    return summary


def mask_to_multilabel(mask: np.ndarray) -> np.ndarray:
    channels = [(mask == value).astype(np.uint8) for value in range(1, 6)]
    channels.append((mask > 0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def create_overlay(image: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    panels = []
    for channel_index, channel_name in enumerate(["brain", "subdural", "bone"]):
        gray = image[..., channel_index]
        panel = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        for value, color in CLASS_COLORS.items():
            class_mask = mask == value
            if not class_mask.any():
                continue
            color_arr = np.asarray(color, dtype=np.float32)
            panel[class_mask] = 0.55 * panel[class_mask] + 0.45 * color_arr
        panel = panel.clip(0, 255).astype(np.uint8)
        cv2.putText(
            panel,
            channel_name,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    canvas = np.concatenate(panels, axis=1)
    cv2.putText(
        canvas,
        title,
        (12, canvas.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def write_examples(
    df: pd.DataFrame,
    *,
    examples_dir: Path,
    num_examples: int,
) -> list[str]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    used = set()
    for value in range(1, 6):
        class_rows = []
        for row in df.loc[df["has_mask"]].itertuples(index=False):
            mask = np.load(row.mask_volume_path, mmap_mode="r")[..., int(row.slice_index)]
            if (mask == value).any():
                class_rows.append(row)
        if class_rows:
            row = class_rows[len(class_rows) // 2]
            selected.append(row)
            used.add((row.series_uid, int(row.slice_index)))

    for row in df.loc[df["has_mask"]].itertuples(index=False):
        key = (row.series_uid, int(row.slice_index))
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= num_examples:
            break

    paths = []
    for row in selected[:num_examples]:
        image = cv2.imread(row.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {row.image_path}")
        mask = np.load(row.mask_volume_path, mmap_mode="r")[..., int(row.slice_index)]
        present = [
            MASK_VALUE_TO_CLASS[value]
            for value in sorted(int(v) for v in np.unique(mask).tolist())
            if value in MASK_VALUE_TO_CLASS
        ]
        title = f"{row.series_uid} slice {int(row.slice_index):03d}: {', '.join(present)}"
        overlay = create_overlay(image, mask, title)
        output_path = examples_dir / f"{row.series_uid}_{int(row.slice_index):03d}.png"
        cv2.imwrite(str(output_path), overlay)
        paths.append(str(output_path))
    return paths


def summarize(df: pd.DataFrame, mask_summary: dict, orientation: dict) -> dict:
    folds = {}
    for fold, fold_df in df.groupby("fold"):
        folds[str(int(fold))] = {
            "patients": int(fold_df["patient_id"].nunique()),
            "studies": int(fold_df["study_id"].nunique()),
            "slices": int(len(fold_df)),
            "positive_slices": int(fold_df["has_mask"].sum()),
        }
    value_voxels = defaultdict(int)
    for item in mask_summary.values():
        # Filled below from saved volumes only when volumes are written.
        for value in item.get("values", []):
            value_voxels[str(value)] += 0
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "studies": int(df["study_id"].nunique()),
        "series": int(df["series_uid"].nunique()),
        "positive_slices": int(df["has_mask"].sum()),
        "folds": folds,
        "label_columns": LABEL_COLUMNS,
        "mask_value_to_class": MASK_VALUE_TO_CLASS,
        "orientation_verification": orientation,
    }


def main() -> None:
    args = parse_args()
    nib = require_nibabel()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        bhsd_root=args.bhsd_root,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        nib=nib,
    )
    orientation = verify_orientation(manifest, bhsd_root=args.bhsd_root, nib=nib)

    mask_summary = {}
    if not args.skip_mask_volumes:
        mask_summary = write_mask_volumes(manifest, bhsd_root=args.bhsd_root, nib=nib)

    manifest_path = args.output_dir / MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)

    examples = []
    if not args.skip_mask_volumes:
        examples = write_examples(
            manifest, examples_dir=args.examples_dir, num_examples=args.num_examples
        )

    summary = summarize(manifest, mask_summary, orientation)
    summary["examples"] = examples
    summary_path = args.output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    if examples:
        print(f"Wrote {len(examples)} orientation examples to {args.examples_dir}")


if __name__ == "__main__":
    main()
