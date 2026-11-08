"""Append NaFlexViT after the queued CSN arm, without interrupting it."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/naflexvit_ablation_20261108"
DEPENDENCY = ROOT / "logs/csn_ablation_20261108/queue.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "queue.json"
    if path.exists():
        raise FileExistsError(path)
    paths = [
        "src/skp/configs/rsna_ich_naflexvit_base_series_sampling.py",
        "src/skp/models/classification/naflexvit.py",
        "scripts/check_ich_naflexvit_ablation.py",
        "scripts/run_ich_naflexvit_ablation.py",
    ]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    (OUT / "source_hashes_at_launch.json").write_text(json.dumps(hashes, indent=2))
    record = dict(
        backbone="naflexvit_base",
        config="rsna_ich_naflexvit_base_series_sampling",
        epochs=50,
        input_shape=[9, 512, 512],
        status="waiting_for_csn",
        dependency=str(DEPENDENCY),
        queued_utc=datetime.now(timezone.utc).isoformat(),
        detail_queue=str(
            ROOT / "logs/series_sampling_20261108_backbone_naflexvit_base/queue.json"
        ),
    )

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps([record], indent=2))
        tmp.replace(path)

    save()
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        CUDA_VISIBLE_DEVICES="0,1",
    )
    try:
        while True:
            previous = json.loads(DEPENDENCY.read_text())
            if any(r["status"] in ("failed", "cancelled") for r in previous):
                raise RuntimeError(
                    "Preceding sweep did not finish successfully; inspect it before restarting NaFlexViT"
                )
            if previous and all(r["status"] == "complete" for r in previous):
                break
            time.sleep(30)
        for p, h in hashes.items():
            assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, (
                f"Source changed: {p}"
            )
        record.update(
            status="preflight", started_utc=datetime.now(timezone.utc).isoformat()
        )
        save()
        with (OUT / "preflight.log").open("w") as log:
            subprocess.run(
                [sys.executable, "scripts/check_ich_naflexvit_ablation.py"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record["status"] = "training"
        save()
        with (OUT / "training-queue.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_ich_series_sampling_experiments.py",
                    "--config",
                    record["config"],
                    "--epochs",
                    "50",
                    "--tag",
                    "20261108_backbone_naflexvit_base",
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


if __name__ == "__main__":
    main()
