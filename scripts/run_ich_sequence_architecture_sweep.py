#!/usr/bin/env python
"""Train validation-only BiLSTM and Transformer sequence comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


CONDITIONS = [
    (architecture, seed)
    for architecture in ["lstm", "transformer"]
    for seed in [88, 89, 90]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="20260713")
    parser.add_argument("--devices", default="-1")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_dir(run_id: str) -> Path:
    return (
        Path("experiments/rsna_ich_effv2m_sequence_bigru")
        / run_id
        / "fixed_split/checkpoints"
    )


def main() -> None:
    args = parse_args()
    log_dir = Path("logs/sequence_architecture_sweep")
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (architecture, seed) in enumerate(CONDITIONS):
        run_id = f"seqarch_{architecture}_seed{seed}_{args.tag}"
        record = {
            "architecture": architecture,
            "seed": seed,
            "run_id": run_id,
        }
        manifest.append(record)
        if (checkpoint_dir(run_id) / "best.ckpt").exists() and not args.overwrite:
            print(f"[{index + 1}/{len(CONDITIONS)}] skipping completed {run_id}")
            continue

        command = [
            "uv",
            "run",
            "python",
            "-m",
            "skp.train",
            "rsna_ich_effv2m_sequence_bigru",
            "--run_id",
            run_id,
            "--devices",
            args.devices,
            "--strategy",
            "ddp",
            "--precision",
            args.precision,
            "--seed",
            str(seed),
            "--sequence_architecture",
            architecture,
            "--loss_params.series_weight",
            "0.25",
            "--loss_params.mil_weight",
            "0.1",
        ]
        if args.overwrite:
            command.append("--overwrite_run")

        log_path = log_dir / f"{run_id}.log"
        print(f"[{index + 1}/{len(CONDITIONS)}] running {run_id}", flush=True)
        env = os.environ.copy()
        env.setdefault("NCCL_SOCKET_IFNAME", "wlo1")
        env.setdefault("GLOO_SOCKET_IFNAME", "wlo1")
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{run_id} failed with exit code {result.returncode}; see {log_path}."
            )

    with (log_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
