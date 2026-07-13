#!/usr/bin/env python
"""Extract frozen RSNA ICH classifier features into contiguous NumPy arrays."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="rsna_ich_effv2m_fixedval_9ch"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_extraction_config(config_name: str):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.pretrained = False
    cfg.train_transforms = None
    cfg.val_transforms = None
    cfg.inference_transforms = None
    cfg.depth_flip_p = 0.0
    cfg.horizontal_flip_p = 0.0
    cfg.vertical_flip_p = 0.0
    return cfg


def load_classifier(cfg, checkpoint: str, device: torch.device):
    model_module = import_module(f"skp.models.{cfg.model}")
    model = model_module.Net(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    model.eval().to(device)
    return model


def worker_extract(
    rank: int,
    world_size: int,
    device_id: int,
    config_name: str,
    checkpoint: str,
    split: str,
    temp_dir: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> None:
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)
    cfg = load_extraction_config(config_name)
    dataset_cls = import_module(f"skp.datasets.{cfg.dataset}").Dataset
    dataset = dataset_cls(cfg, split)
    n_rows = min(len(dataset), max_samples) if max_samples else len(dataset)
    indices = np.arange(rank, n_rows, world_size, dtype=np.int64)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    model = load_classifier(cfg, checkpoint, device)

    features_path = Path(temp_dir) / f"{split}.features.rank{rank}.npy"
    logits_path = Path(temp_dir) / f"{split}.logits.rank{rank}.npy"
    indices_path = Path(temp_dir) / f"{split}.indices.rank{rank}.npy"
    features = open_memmap(
        features_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(indices), model.feature_dim),
    )
    logits = open_memmap(
        logits_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(indices), cfg.num_classes),
    )
    np.save(indices_path, indices)

    cursor = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model({"x": x}, return_features=True)
            n = x.shape[0]
            features[cursor : cursor + n] = out["features"].float().cpu().numpy()
            logits[cursor : cursor + n] = out["logits"].float().cpu().numpy()
            cursor += n
            if rank == 0 and (batch_idx + 1) % 100 == 0:
                print(
                    f"{split}: rank {rank} extracted {cursor}/{len(indices)}",
                    flush=True,
                )
    features.flush()
    logits.flush()


def merge_parts(
    output_dir: Path,
    temp_dir: Path,
    split: str,
    world_size: int,
    n_rows: int,
    feature_dim: int,
    num_classes: int,
) -> None:
    final_features = open_memmap(
        output_dir / f"{split}_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, feature_dim),
    )
    final_logits = open_memmap(
        output_dir / f"{split}_base_logits.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, num_classes),
    )
    for rank in range(world_size):
        indices = np.load(temp_dir / f"{split}.indices.rank{rank}.npy")
        features = np.load(
            temp_dir / f"{split}.features.rank{rank}.npy", mmap_mode="r"
        )
        logits = np.load(
            temp_dir / f"{split}.logits.rank{rank}.npy", mmap_mode="r"
        )
        for start in range(0, len(indices), 8192):
            stop = min(start + 8192, len(indices))
            final_features[indices[start:stop]] = features[start:stop]
            final_logits[indices[start:stop]] = logits[start:stop]
    final_features.flush()
    final_logits.flush()


def write_metadata(
    output_dir: Path, split: str, dataset, n_rows: int | None = None
) -> None:
    df = dataset.df.iloc[:n_rows].reset_index(drop=True).copy()
    labels = df[dataset.label_columns].to_numpy(dtype=np.uint8)
    np.save(output_dir / f"{split}_labels.npy", labels)

    slice_columns = [
        "patient_id",
        "study_id",
        "series_id",
        "series_uid",
        "filename",
        "slice_path",
        "slice_sort_key",
    ]
    slice_df = df[slice_columns].copy()
    slice_df.insert(0, "row_index", np.arange(len(slice_df), dtype=np.int64))
    slice_df.to_csv(output_dir / f"{split}_slices.csv", index=False)

    grouped = df.groupby("series_uid", sort=False, observed=True)
    series_df = grouped.agg(
        patient_id=("patient_id", "first"),
        study_id=("study_id", "first"),
        series_id=("series_id", "first"),
        offset=("series_uid", lambda x: int(x.index[0])),
        length=("series_uid", "size"),
    ).reset_index()
    series_df.insert(0, "series_index", np.arange(len(series_df), dtype=np.int64))
    series_df.to_csv(output_dir / f"{split}_series.csv", index=False)


def extract_split(args, split: str, devices: list[int], output_dir: Path) -> dict:
    cfg = load_extraction_config(args.config)
    dataset_cls = import_module(f"skp.datasets.{cfg.dataset}").Dataset
    dataset = dataset_cls(cfg, split)
    temp_dir = output_dir / ".parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    processes = []
    for rank, device_id in enumerate(devices):
        process = ctx.Process(
            target=worker_extract,
            args=(
                rank,
                len(devices),
                device_id,
                args.config,
                args.checkpoint,
                split,
                str(temp_dir),
                args.batch_size,
                args.num_workers,
                args.max_samples,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Feature extraction worker {process.pid} failed with "
                f"exit code {process.exitcode}."
            )

    sample_features = np.load(
        temp_dir / f"{split}.features.rank0.npy", mmap_mode="r"
    )
    feature_dim = sample_features.shape[1]
    n_rows = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    merge_parts(
        output_dir,
        temp_dir,
        split,
        len(devices),
        n_rows,
        feature_dim,
        cfg.num_classes,
    )
    write_metadata(output_dir, split, dataset, n_rows)
    for path in temp_dir.glob(f"{split}.*.rank*.npy"):
        path.unlink()
    return {
        "rows": n_rows,
        "series": int(dataset.df.iloc[:n_rows]["series_uid"].nunique()),
        "feature_dim": int(feature_dim),
        "max_sequence_length": int(
            dataset.df.iloc[:n_rows].groupby("series_uid").size().max()
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction.")
    devices = [int(value) for value in args.devices.split(",")]
    unavailable = [device for device in devices if device >= torch.cuda.device_count()]
    if unavailable:
        raise ValueError(f"CUDA devices are not visible: {unavailable}")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in args.splits:
        print(f"Extracting {split} on CUDA devices {devices} ...", flush=True)
        summary[split] = extract_split(args, split, devices, output_dir)
    parts_dir = output_dir / ".parts"
    if parts_dir.exists() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()

    manifest = {
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_mtime_ns": os.stat(args.checkpoint).st_mtime_ns,
        "feature_dtype": "float16",
        "base_logits_dtype": "float16",
        "label_dtype": "uint8",
        "splits": summary,
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
