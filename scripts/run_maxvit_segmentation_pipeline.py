#!/usr/bin/env python
"""Run the restartable MaxViT BHSD-to-RSNA pseudolabel pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TEACHER_CONFIG = "bhsd_maxvit_tiny_seg_9ch"
TEACHER_RUN_ID = "maxvit_teachers_any2_20260715"
TEACHER_ROOT = Path("experiments") / TEACHER_CONFIG / TEACHER_RUN_ID
OOF_DIR = Path("data/pseudolabels/bhsd_maxvit_9ch_oof_any2")
PSEUDOLABEL_DIR = Path("data/pseudolabels/rsna_maxvit_9ch_any2")

PRETRAIN_CONFIG = "rsna_ich_joint_maxvit_tiny_9ch_deeplab_frozen"
PRETRAIN_RUN_ID = "maxvit_pseudolabel_decoder_any2_20260715"
PRETRAIN_ROOT = Path("experiments") / PRETRAIN_CONFIG / PRETRAIN_RUN_ID

FINAL_CONFIG = "bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen"
FINAL_RUN_ID = "maxvit_final_pseudolabel_any2_20260715"
DIRECT_CONFIG = "bhsd_maxvit_tiny_seg_9ch_deeplab_direct_frozen"
DIRECT_RUN_ID = "maxvit_direct_frozen_any2_20260715"
FINETUNE_CONFIG = "bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_finetune"
FINETUNE_RUN_ID = "maxvit_pseudolabel_finetune_any2_20260715"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--skip-controls", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def checkpoint_set_complete(root: Path) -> bool:
    return all(
        (root / f"fold{fold}/checkpoints/last.ckpt").exists() for fold in range(5)
    )


def train_cv(config: str, run_id: str, devices: int, precision: str) -> None:
    root = Path("experiments") / config / run_id
    if checkpoint_set_complete(root):
        print(f"Skipping completed CV run: {root}", flush=True)
        return
    run(
        [
            sys.executable,
            "-m",
            "skp.train",
            config,
            "--run_id",
            run_id,
            "--kfold",
            "0,1,2,3,4",
            "--devices",
            str(devices),
            "--strategy",
            "ddp",
            "--precision",
            precision,
            "--keep_pretrained_weights_path",
        ]
    )


def train_fixed(config: str, run_id: str, devices: int, precision: str) -> None:
    checkpoint = Path("experiments") / config / run_id / "fold0/checkpoints/last.ckpt"
    if checkpoint.exists():
        print(f"Skipping completed fixed run: {checkpoint}", flush=True)
        return
    run(
        [
            sys.executable,
            "-m",
            "skp.train",
            config,
            "--run_id",
            run_id,
            "--devices",
            str(devices),
            "--strategy",
            "ddp",
            "--precision",
            precision,
            "--keep_pretrained_weights_path",
        ]
    )


def generate_oof() -> None:
    summary = OOF_DIR / "threshold_summary.json"
    if summary.exists():
        print(f"Skipping completed OOF threshold sweep: {summary}", flush=True)
        return
    run(
        [
            sys.executable,
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "oof-threshold",
            "--bhsd-config",
            TEACHER_CONFIG,
            "--checkpoint-root",
            str(TEACHER_ROOT),
            "--output-dir",
            str(OOF_DIR),
            "--device",
            "cuda:0",
            "--batch-size",
            "32",
            "--num-workers",
            "4",
        ]
    )


def generate_pseudolabels() -> None:
    merged_manifest = PSEUDOLABEL_DIR / "manifest.csv"
    rank_summaries = [
        PSEUDOLABEL_DIR / f"rank_logs/summary_rank{rank}.json" for rank in range(2)
    ]
    if merged_manifest.exists() and all(path.exists() for path in rank_summaries):
        print(f"Skipping completed pseudolabel generation: {merged_manifest}")
        return

    if PSEUDOLABEL_DIR.exists():
        shutil.rmtree(PSEUDOLABEL_DIR)
    PSEUDOLABEL_DIR.mkdir(parents=True)
    logs = Path("logs/maxvit_pseudolabel_generation")
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    for rank in range(2):
        log_path = logs / f"rank{rank}.log"
        log_file = log_path.open("w")
        command = [
            sys.executable,
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "pseudolabel-rsna",
            "--bhsd-config",
            TEACHER_CONFIG,
            "--rsna-config",
            "rsna_ich_maxvit_tiny_fixedval_9ch",
            "--checkpoint-root",
            str(TEACHER_ROOT),
            "--threshold-file",
            str(OOF_DIR / "threshold_summary.json"),
            "--output-dir",
            str(PSEUDOLABEL_DIR),
            "--rank",
            str(rank),
            "--world-size",
            "2",
            "--device",
            f"cuda:{rank}",
            "--batch-size",
            "32",
            "--num-workers",
            "6",
            "--temperature",
            "1.0",
            "--overwrite",
        ]
        print("\n$ " + " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        processes.append((process, log_file, log_path))

    failures = []
    for process, log_file, log_path in processes:
        status = process.wait()
        log_file.close()
        if status != 0:
            failures.append(f"{log_path} (exit {status})")
    if failures:
        raise RuntimeError("Pseudolabel workers failed: " + ", ".join(failures))

    run(
        [
            sys.executable,
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "merge-rsna-manifests",
            "--output-dir",
            str(PSEUDOLABEL_DIR),
        ]
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    train_cv(TEACHER_CONFIG, TEACHER_RUN_ID, args.devices, args.precision)
    generate_oof()
    generate_pseudolabels()
    train_fixed(PRETRAIN_CONFIG, PRETRAIN_RUN_ID, args.devices, args.precision)

    # This is the deployable five-decoder ensemble.
    train_cv(FINAL_CONFIG, FINAL_RUN_ID, args.devices, args.precision)
    if not args.skip_controls:
        train_cv(DIRECT_CONFIG, DIRECT_RUN_ID, args.devices, args.precision)
        train_cv(FINETUNE_CONFIG, FINETUNE_RUN_ID, args.devices, args.precision)
    print("MaxViT segmentation pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
