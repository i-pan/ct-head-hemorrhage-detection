#!/usr/bin/env python
"""Queue independently initialized 50/75/100-epoch MaxViT series-sampling runs.

Uses two GPUs per run, sequentially. Stops on any training/export failure.
Example: .venv/bin/python scripts/run_ich_series_sampling_experiments.py
Change the positive-bin probability with --positive-probability 0.75.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "rsna_ich_maxvit_tiny_series_sampling"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 75, 100])
    parser.add_argument("--positive-probability", type=float, default=0.5)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--tag", default="20261107")
    parser.add_argument("--export-only", type=Path)
    args = parser.parse_args()
    if args.export_only:
        from analyze_ich_backbone_ensemble import export_predictions

        inference_args = argparse.Namespace(
            device="cuda:0", batch_size=64, num_workers=4
        )
        export_predictions(
            args.config,
            args.export_only / "fold0/checkpoints/best.ckpt",
            args.export_only / "validation_predictions.npz",
            inference_args,
        )
        return
    if not 0 <= args.positive_probability <= 1:
        parser.error("positive-probability must be in [0,1]")
    if any(e <= 0 or e % args.validation_every for e in args.epochs):
        parser.error(
            "Epoch budgets must be positive multiples of validation-every so every run has a final validation."
        )
    out = ROOT / "logs" / f"series_sampling_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        CUDA_VISIBLE_DEVICES="0,1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    records = []
    for epochs in args.epochs:
        name = f"series_e{epochs:03d}_p{args.positive_probability:g}_seed88_{args.tag}"
        run_dir = ROOT / "experiments" / args.config / name
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {run_dir}")
        command = [
            sys.executable,
            "-m",
            "skp.train",
            args.config,
            "--run_id",
            name,
            "--accelerator",
            "cuda",
            "--devices",
            "2",
            "--strategy",
            "ddp",
            "--precision",
            "bf16-mixed",
            "--check_val_every_n_epoch",
            str(args.validation_every),
            "--num_epochs",
            str(epochs),
            "--series_positive_slice_probability",
            str(args.positive_probability),
        ]
        records.append(
            dict(
                epochs=epochs,
                run_id=name,
                run_dir=str(run_dir),
                command=command,
                positive_bin_probability=args.positive_probability,
                status="queued",
            )
        )

    def save():
        tmp = out / "queue.tmp.json"
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(out / "queue.json")

    save()
    for record in records:
        record.update(
            status="training", started_utc=datetime.now(timezone.utc).isoformat()
        )
        save()
        print("TRAIN", record["run_id"], flush=True)
        try:
            with (out / f"{record['run_id']}.log").open("w") as log:
                subprocess.run(
                    record["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            record["status"] = "exporting_validation"
            save()
            with (out / f"{record['run_id']}-export.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config",
                        args.config,
                        "--export-only",
                        record["run_dir"],
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            record.update(
                status="complete", finished_utc=datetime.now(timezone.utc).isoformat()
            )
            save()
        except BaseException as error:
            record.update(status="failed", error=str(error))
            save()
            raise
    print("All requested experiments and validation exports completed.", flush=True)


if __name__ == "__main__":
    main()
