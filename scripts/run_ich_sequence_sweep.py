#!/usr/bin/env python
"""Run the focused RSNA ICH frozen-feature sequence experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


CONDITIONS = [
    ("control", {}),
    ("sliceheavy", {"loss_params.series_weight": 0.25, "loss_params.mil_weight": 0.1}),
    ("nomil", {"loss_params.series_weight": 0.25, "loss_params.mil_weight": 0.0}),
    ("minaux", {"loss_params.series_weight": 0.1, "loss_params.mil_weight": 0.05}),
    (
        "lowseries_nomil",
        {"loss_params.series_weight": 0.1, "loss_params.mil_weight": 0.0},
    ),
    ("noise0", {"feature_noise_std": 0.0}),
    ("noise01", {"feature_noise_std": 0.01}),
    ("gru1", {"sequence_num_layers": 1}),
    ("hidden128", {"sequence_hidden_dim": 128}),
    ("lr1e4", {"optimizer_params.lr": 1e-4}),
    (
        "sliceheavy_noise0",
        {
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "feature_noise_std": 0.0,
        },
    ),
    (
        "sliceheavy_noise01",
        {
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "feature_noise_std": 0.01,
        },
    ),
    (
        "sliceheavy_gru1",
        {
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "sequence_num_layers": 1,
        },
    ),
    (
        "sliceheavy_hidden128",
        {
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "sequence_hidden_dim": 128,
        },
    ),
    (
        "sliceheavy_lr1e4",
        {
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "optimizer_params.lr": 1e-4,
        },
    ),
    (
        "sliceheavy_seed89",
        {
            "seed": 89,
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
        },
    ),
    (
        "sliceheavy_seed90",
        {
            "seed": 90,
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
        },
    ),
    (
        "sliceheavy_gru1_seed89",
        {
            "seed": 89,
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "sequence_num_layers": 1,
        },
    ),
    (
        "sliceheavy_gru1_seed90",
        {
            "seed": 90,
            "loss_params.series_weight": 0.25,
            "loss_params.mil_weight": 0.1,
            "sequence_num_layers": 1,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="20260713")
    parser.add_argument("--config", default="rsna_ich_effv2m_sequence_bigru")
    parser.add_argument(
        "--experiment-dir",
        default="experiments/rsna_ich_effv2m_sequence_bigru",
    )
    parser.add_argument("--log-dir", default="logs/sequence_sweep")
    parser.add_argument("--devices", default="-1")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_dir(experiment_dir: Path, run_id: str) -> Path:
    return experiment_dir / run_id / "fixed_split/checkpoints"


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (name, overrides) in enumerate(CONDITIONS):
        run_id = f"seqsweep_{index:02d}_{name}_{args.tag}"
        best = checkpoint_dir(experiment_dir, run_id) / "best.ckpt"
        record = {"index": index, "name": name, "run_id": run_id, **overrides}
        manifest.append(record)
        if best.exists() and not args.overwrite:
            print(f"[{index + 1}/{len(CONDITIONS)}] skipping completed {name}")
            continue

        command = [
            "uv",
            "run",
            "python",
            "-m",
            "skp.train",
            args.config,
            "--run_id",
            run_id,
            "--devices",
            args.devices,
            "--strategy",
            "ddp",
            "--precision",
            args.precision,
            "--seed",
            str(overrides.get("seed", 88)),
        ]
        if args.overwrite:
            command.append("--overwrite_run")
        for key, value in overrides.items():
            if key == "seed":
                continue
            command.extend([f"--{key}", str(value)])

        log_path = log_dir / f"{run_id}.log"
        print(f"[{index + 1}/{len(CONDITIONS)}] running {name}: {overrides}")
        env = os.environ.copy()
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
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
                f"Condition {name} failed with exit code {result.returncode}; "
                f"see {log_path}."
            )

    with (log_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
