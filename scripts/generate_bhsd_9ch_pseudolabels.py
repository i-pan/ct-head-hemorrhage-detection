#!/usr/bin/env python
"""Generate 9-channel BHSD OOF predictions and RSNA segmentation pseudolabels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from skp.datasets import bhsd_seg, rsna_ich_2p5d
from skp.metrics.bhsd import _hd95
from skp.models.segmentation.base import Net
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


LABEL_COLUMNS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
DEFAULT_THRESHOLDS = [round(value / 10, 1) for value in range(1, 10)]
DEFAULT_CHECKPOINT_ROOT = Path("experiments/bhsd_effv2m_seg_9ch/pleased-hare-6156")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    oof = subparsers.add_parser("oof-threshold")
    oof.add_argument("--bhsd-config", default="bhsd_effv2m_seg_9ch")
    oof.add_argument("--output-dir", default="data/pseudolabels/bhsd_9ch_oof_last")
    oof.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    oof.add_argument("--batch-size", type=int, default=32)
    oof.add_argument("--num-workers", type=int, default=4)
    oof.add_argument("--device", default="cuda:0")
    oof.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))

    rsna = subparsers.add_parser("pseudolabel-rsna")
    rsna.add_argument("--bhsd-config", default="bhsd_effv2m_seg_9ch")
    rsna.add_argument("--rsna-config", default="rsna_ich_effv2m_fixedval_9ch")
    rsna.add_argument("--output-dir", default="data/pseudolabels/rsna_9ch_last")
    rsna.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    rsna.add_argument("--threshold-file", default="")
    rsna.add_argument("--threshold", type=float, default=None)
    rsna.add_argument("--temperature", type=float, default=1.0)
    rsna.add_argument("--splits", default="train,val")
    rsna.add_argument("--batch-size", type=int, default=32)
    rsna.add_argument("--num-workers", type=int, default=6)
    rsna.add_argument("--rank", type=int, default=0)
    rsna.add_argument("--world-size", type=int, default=1)
    rsna.add_argument("--device", default="cuda:0")
    rsna.add_argument("--png-compression", type=int, default=3)
    rsna.add_argument("--max-series", type=int, default=None)
    rsna.add_argument("--overwrite", action="store_true")

    merge = subparsers.add_parser("merge-rsna-manifests")
    merge.add_argument("--output-dir", default="data/pseudolabels/rsna_9ch_last")

    return parser.parse_args()


def _set_inference_defaults(cfg):
    cfg = deepcopy(cfg)
    cfg.pretrained = False
    cfg.load_pretrained_encoder = None
    cfg.load_pretrained_decoder = None
    cfg.load_pretrained_model = None
    cfg.train_transforms = None
    cfg.val_transforms = None
    cfg.inference_transforms = None
    cfg.num_workers = 0
    cfg.val_num_workers = 0
    return cfg


def load_config(config_name: str):
    return import_module(f"skp.configs.{config_name}").cfg


def checkpoint_paths(checkpoint_root: str | Path) -> list[Path]:
    root = Path(checkpoint_root)
    paths = [root / f"fold{fold}" / "checkpoints" / "last.ckpt" for fold in range(5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    return paths


def load_model(checkpoint_path: str | Path, device: torch.device, base_cfg) -> Net:
    cfg = _set_inference_defaults(base_cfg)
    model = Net(cfg)
    weights = torch_load_weights(str(checkpoint_path))
    weights = filter_weights_by_prefix(weights, "model.")
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model


def worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": dataset.collate_fn,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        params["persistent_workers"] = True
        params["prefetch_factor"] = 2
    return DataLoader(dataset, **params)


def predict_model(model: Net, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        return model({"x": x})["logits"].sigmoid().float().cpu()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return math.nan
    intersection = int(np.logical_and(pred, target).sum())
    return (2.0 * intersection) / denom


def _series_key(row) -> str:
    if "series_uid" in row:
        return str(row.series_uid)
    return f"{row.patient_id}/{row.study_id}/{row.series_id}"


def _safe_relpath(series_uid: str) -> Path:
    return Path(*series_uid.split("/")).with_suffix(".npz")


def _slice_sort_key(filename: str) -> str:
    stem = Path(filename).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return f"~|{stem}"
    return f"{int(numbers[0]):020d}|{stem}"


def _logit(x: float) -> float:
    eps = 1e-6
    x = min(max(float(x), eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def thresholded_sigmoid_soft_mask(
    probability: np.ndarray,
    *,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    hard = probability >= threshold
    if not hard.any():
        return np.zeros(probability.shape, dtype=np.uint8)
    eps = 1e-6
    p = np.clip(probability.astype(np.float32), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    centered = 1.0 / (1.0 + np.exp(-((logits - _logit(threshold)) / temperature)))
    centered *= hard
    return np.round(np.clip(centered, 0.0, 1.0) * 255.0).astype(np.uint8)


def subtype_masks_from_any(
    any_mask: np.ndarray,
    subtype_probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    masks = np.zeros((6, *any_mask.shape), dtype=np.uint8)
    masks[5] = any_mask
    allowed = np.flatnonzero(labels[:5] >= 0.5)
    if any_mask.max() == 0 or allowed.size == 0:
        return masks
    if allowed.size == 1:
        masks[int(allowed[0])] = any_mask
        return masks

    allowed_probs = subtype_probs[allowed].astype(np.float32)
    denom = allowed_probs.sum(axis=0, keepdims=True)
    weights = np.divide(
        allowed_probs,
        denom,
        out=np.full_like(allowed_probs, 1.0 / allowed.size),
        where=denom > 1e-6,
    )
    soft_any = any_mask.astype(np.float32)
    masks[allowed] = np.round(weights * soft_any[None]).astype(np.uint8)
    return masks


def run_oof_threshold(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    device = torch.device(args.device)
    ckpts = checkpoint_paths(args.checkpoint_root)
    bhsd_9ch_cfg = load_config(args.bhsd_config)

    full_df = pd.read_csv(bhsd_9ch_cfg.annotations_file)
    full_df["global_index"] = np.arange(len(full_df))
    key_to_global = {
        (row.series_uid, int(row.slice_index)): int(row.global_index)
        for row in full_df.itertuples(index=False)
    }
    metadata_path = out_dir / "oof_metadata.csv"
    full_df.to_csv(metadata_path, index=False)

    shape = (len(full_df), bhsd_9ch_cfg.image_height, bhsd_9ch_cfg.image_width)
    prob_path = out_dir / "oof_any_prob_uint8.npy"
    target_path = out_dir / "oof_any_target_uint8.npy"
    prob_mm = np.lib.format.open_memmap(
        prob_path, mode="w+", dtype=np.uint8, shape=shape
    )
    target_mm = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.uint8, shape=shape
    )

    for fold, ckpt in enumerate(ckpts):
        cfg = _set_inference_defaults(bhsd_9ch_cfg)
        cfg.fold = fold
        dataset = bhsd_seg.Dataset(cfg, "val")
        loader = make_loader(dataset, args.batch_size, args.num_workers)
        model = load_model(ckpt, device, bhsd_9ch_cfg)
        print(f"OOF fold {fold}: N={len(dataset)} checkpoint={ckpt}", flush=True)
        for batch_idx, batch in enumerate(loader):
            probs = predict_model(model, batch["x"], device)[:, 5].numpy()
            targets = (batch["y"][:, 5].numpy() >= 0.5).astype(np.uint8)
            local_indices = batch["index"].numpy()
            global_indices = []
            for local_idx in local_indices:
                row = dataset.df.iloc[int(local_idx)]
                global_indices.append(
                    key_to_global[(str(row.series_uid), int(row.slice_index))]
                )
            global_indices = np.asarray(global_indices, dtype=np.int64)
            prob_mm[global_indices] = np.round(probs * 255.0).astype(np.uint8)
            target_mm[global_indices] = targets
            if batch_idx % 20 == 0:
                print(f"  fold {fold} batch {batch_idx}/{len(loader)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prob_mm.flush()
    target_mm.flush()

    prob = np.load(prob_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r").astype(bool)
    volume_rows = []
    for threshold in thresholds:
        cutoff = int(round(threshold * 255.0))
        slice_scores = []
        slice_hd95_scores = []
        volume_scores = []
        volume_hd95_scores = []
        for _, group in full_df.groupby("series_uid", sort=False):
            idx = group["global_index"].to_numpy(dtype=np.int64)
            p = prob[idx] >= cutoff
            t = target[idx]
            row_spacing = float(group.iloc[0]["row_spacing_mm"])
            col_spacing = float(group.iloc[0]["col_spacing_mm"])
            slice_spacing = float(group.iloc[0]["slice_spacing_mm"])
            for slice_idx in range(len(idx)):
                score = _binary_dice(p[slice_idx], t[slice_idx])
                if not math.isnan(score):
                    slice_scores.append(score)
                hd95 = _hd95(
                    p[slice_idx],
                    t[slice_idx],
                    (row_spacing, col_spacing),
                )
                if not math.isnan(hd95):
                    slice_hd95_scores.append(hd95)
            volume_score = _binary_dice(p, t)
            if not math.isnan(volume_score):
                volume_scores.append(volume_score)
            volume_hd95 = _hd95(
                p,
                t,
                (slice_spacing, row_spacing, col_spacing),
            )
            if not math.isnan(volume_hd95):
                volume_hd95_scores.append(volume_hd95)
        volume_rows.append(
            {
                "threshold": threshold,
                "slice_dice_any": float(np.mean(slice_scores)),
                "slice_hd95_any": float(np.mean(slice_hd95_scores)),
                "volume_dice_any": float(np.mean(volume_scores)),
                "volume_hd95_any": float(np.mean(volume_hd95_scores)),
                "n_slice_scores": len(slice_scores),
                "n_volume_scores": len(volume_scores),
            }
        )

    sweep_df = pd.DataFrame(volume_rows)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    best_dice = sweep_df.loc[sweep_df["volume_dice_any"].idxmax()].to_dict()
    best_hd95 = sweep_df.loc[sweep_df["volume_hd95_any"].idxmin()].to_dict()
    summary = {
        "bhsd_config": args.bhsd_config,
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_paths": [str(path) for path in ckpts],
        "metadata_path": str(metadata_path),
        "probability_path": str(prob_path),
        "target_path": str(target_path),
        "thresholds": thresholds,
        "selected_threshold": float(best_dice["threshold"]),
        "selected_volume_dice_any": float(best_dice["volume_dice_any"]),
        "selected_slice_dice_any": float(best_dice["slice_dice_any"]),
        "selected_hd95_threshold": float(best_hd95["threshold"]),
        "selected_volume_hd95_any": float(best_hd95["volume_hd95_any"]),
        "selected_slice_hd95_any": float(best_hd95["slice_hd95_any"]),
    }
    with open(out_dir / "threshold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def load_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    threshold_file = Path(args.threshold_file)
    if not threshold_file.exists():
        raise FileNotFoundError(
            "--threshold is required when --threshold-file does not exist: "
            f"{threshold_file}"
        )
    with open(threshold_file) as f:
        summary = json.load(f)
    return float(summary["selected_threshold"])


def make_rsna_dataset(args: argparse.Namespace, rsna_9ch_cfg):
    cfg = _set_inference_defaults(rsna_9ch_cfg)
    cfg.dataset = "rsna_ich_2p5d"
    cfg.batch_size = args.batch_size
    cfg.val_batch_size = args.batch_size
    dataset = rsna_ich_2p5d.Dataset(cfg, "inference")
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    df = dataset.df
    df = df.loc[df["split"].isin(splits)].copy()
    df = df.loc[df["any"] >= 0.5].copy()
    all_series = sorted(df["series_uid"].unique())
    assigned = set(all_series[args.rank :: args.world_size])
    if args.max_series is not None:
        assigned = set(sorted(assigned)[: args.max_series])
    df = df.loc[df["series_uid"].isin(assigned)].copy()
    df = df.sort_values(["series_uid", "slice_sort_key", "filename"]).reset_index(
        drop=True
    )
    dataset.df = df
    dataset.series_index = {
        series_uid: idx
        for idx, series_uid in enumerate(sorted(df["series_uid"].unique()))
    }
    return dataset, splits, len(all_series), len(assigned)


def write_series_npz(
    series_uid: str,
    records: list[dict],
    out_dir: Path,
) -> str:
    if not records:
        return ""
    rel_path = _safe_relpath(series_uid)
    path = out_dir / "masks_by_series" / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.stack([record["mask"] for record in records], axis=0)
    np.savez_compressed(
        path,
        masks=masks,
        slice_paths=np.asarray([record["slice_path"] for record in records]),
        filenames=np.asarray([record["filename"] for record in records]),
        slice_sort_keys=np.asarray([record["slice_sort_key"] for record in records]),
        labels=np.stack([record["labels"] for record in records], axis=0).astype(
            np.uint8
        ),
        any_probability_mean=np.asarray(
            [record["any_probability_mean"] for record in records],
            dtype=np.float16,
        ),
        any_probability_max=np.asarray(
            [record["any_probability_max"] for record in records],
            dtype=np.float16,
        ),
    )
    return str(path.relative_to(out_dir))


def run_rsna_pseudolabel(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rank_logs").mkdir(exist_ok=True)
    threshold = load_threshold(args)
    device = torch.device(args.device)
    ckpts = checkpoint_paths(args.checkpoint_root)
    bhsd_9ch_cfg = load_config(args.bhsd_config)
    rsna_9ch_cfg = load_config(args.rsna_config)
    dataset, splits, total_series, assigned_series = make_rsna_dataset(
        args, rsna_9ch_cfg
    )
    loader = make_loader(dataset, args.batch_size, args.num_workers)
    print(
        f"RSNA rank {args.rank}/{args.world_size}: N={len(dataset)} positive slices, "
        f"series={assigned_series}/{total_series}, threshold={threshold}",
        flush=True,
    )

    models = [load_model(path, device, bhsd_9ch_cfg) for path in ckpts]
    manifest_path = out_dir / f"manifest_rank{args.rank}.csv"
    fieldnames = [
        "patient_id",
        "study_id",
        "series_id",
        "series_uid",
        "slice_path",
        "filename",
        "split",
        *LABEL_COLUMNS,
        "mask_path",
        "mask_index",
        "mask_nonzero_pixels",
        "any_probability_mean",
        "any_probability_max",
        "threshold",
        "temperature",
    ]

    mode = "w" if args.overwrite or not manifest_path.exists() else "a"
    processed_series = set()
    if mode == "a":
        existing = pd.read_csv(manifest_path)
        processed_series = set(existing["series_uid"].dropna().unique())

    current_series = None
    pending_records: list[dict] = []
    pending_manifest_rows: list[dict] = []
    total_saved = 0

    def flush_series(writer) -> None:
        nonlocal current_series, pending_records, pending_manifest_rows, total_saved
        if current_series is None:
            return
        mask_path = write_series_npz(current_series, pending_records, out_dir)
        index_by_slice_path = {
            record["slice_path"]: idx for idx, record in enumerate(pending_records)
        }
        for row in pending_manifest_rows:
            if row["slice_path"] in index_by_slice_path:
                row["mask_path"] = mask_path
                row["mask_index"] = index_by_slice_path[row["slice_path"]]
            writer.writerow(row)
        total_saved += len(pending_records)
        current_series = None
        pending_records = []
        pending_manifest_rows = []

    with open(manifest_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for batch_idx, batch in enumerate(loader):
            local_indices = batch["index"].numpy()
            rows = dataset.df.iloc[local_indices].reset_index(drop=True)
            if rows["series_uid"].isin(processed_series).all():
                continue

            probs_sum = None
            for model in models:
                probs = predict_model(model, batch["x"], device)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs = (probs_sum / len(models)).numpy()
            labels = batch["y"].numpy()

            for sample_idx, row in rows.iterrows():
                series_uid = str(row.series_uid)
                if series_uid in processed_series:
                    continue
                if current_series is None:
                    current_series = series_uid
                if series_uid != current_series:
                    flush_series(writer)
                    processed_series.add(current_series)
                    current_series = series_uid

                any_prob = probs[sample_idx, 5]
                any_mask = thresholded_sigmoid_soft_mask(
                    any_prob,
                    threshold=threshold,
                    temperature=args.temperature,
                )
                label_values = labels[sample_idx].astype(np.uint8)
                mask_nonzero = int(np.count_nonzero(any_mask))
                manifest_row = {
                    "patient_id": row.patient_id,
                    "study_id": row.study_id,
                    "series_id": row.series_id,
                    "series_uid": series_uid,
                    "slice_path": row.slice_path,
                    "filename": row.filename,
                    "split": row.split,
                    **{
                        label: int(label_values[idx])
                        for idx, label in enumerate(LABEL_COLUMNS)
                    },
                    "mask_path": "",
                    "mask_index": "",
                    "mask_nonzero_pixels": mask_nonzero,
                    "any_probability_mean": float(any_prob.mean()),
                    "any_probability_max": float(any_prob.max()),
                    "threshold": threshold,
                    "temperature": args.temperature,
                }
                pending_manifest_rows.append(manifest_row)
                if mask_nonzero > 0:
                    masks = subtype_masks_from_any(
                        any_mask,
                        probs[sample_idx, :5],
                        labels[sample_idx],
                    )
                    pending_records.append(
                        {
                            "mask": masks,
                            "slice_path": row.slice_path,
                            "filename": row.filename,
                            "slice_sort_key": row.slice_sort_key,
                            "labels": label_values,
                            "any_probability_mean": float(any_prob.mean()),
                            "any_probability_max": float(any_prob.max()),
                        }
                    )
            if batch_idx % 20 == 0:
                print(
                    f"rank {args.rank}: batch {batch_idx}/{len(loader)} "
                    f"saved_masks={total_saved}",
                    flush=True,
                )
        flush_series(writer)

    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "bhsd_config": args.bhsd_config,
        "rsna_config": args.rsna_config,
        "splits": splits,
        "n_positive_slices": len(dataset),
        "n_assigned_series": assigned_series,
        "threshold": threshold,
        "temperature": args.temperature,
        "checkpoint_paths": [str(path) for path in ckpts],
        "manifest_path": str(manifest_path),
        "saved_masks": total_saved,
    }
    with open(out_dir / "rank_logs" / f"summary_rank{args.rank}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def merge_rsna_manifests(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    manifest_paths = sorted(out_dir.glob("manifest_rank*.csv"))
    if not manifest_paths:
        raise FileNotFoundError(f"No rank manifests found in {out_dir}")
    dfs = [pd.read_csv(path) for path in manifest_paths]
    manifest = pd.concat(dfs, ignore_index=True)
    manifest = manifest.sort_values(["series_uid", "slice_path"]).reset_index(drop=True)
    output_path = out_dir / "manifest.csv"
    manifest.to_csv(output_path, index=False)
    summaries = []
    for path in sorted((out_dir / "rank_logs").glob("summary_rank*.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    summary = {
        "manifest_path": str(output_path),
        "rank_manifests": [str(path) for path in manifest_paths],
        "n_rows": int(len(manifest)),
        "n_masks": int(manifest["mask_path"].fillna("").ne("").sum()),
        "n_series": int(manifest["series_uid"].nunique()),
        "rank_summaries": summaries,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "oof-threshold":
        run_oof_threshold(args)
    elif args.command == "pseudolabel-rsna":
        run_rsna_pseudolabel(args)
    elif args.command == "merge-rsna-manifests":
        merge_rsna_manifests(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
