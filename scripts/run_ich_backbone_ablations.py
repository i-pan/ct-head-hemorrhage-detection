"""Run fixed-protocol 50-epoch backbone ablations sequentially on both GPUs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["effnetv2m", "convnext_tiny", "coatnet0"]
OUT = ROOT / "logs/backbone_ablations_20261108"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue.json"
    resume = "--resume" in sys.argv
    if status_path.exists() and not resume:
        raise FileExistsError(status_path)
    previous = json.loads(
        (
            ROOT
            / "logs/series_sampling_20261107_seriesauc/source_hashes_at_launch.json"
        ).read_text()
    )
    for name, digest in previous.items():
        if name.startswith("src/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, (
                f"Baseline source changed: {name}"
            )
    paths = (
        list(previous)
        + [
            str(p.relative_to(ROOT))
            for p in (ROOT / "src/skp/configs").glob("rsna_ich_*_series_sampling.py")
        ]
        + [
            "scripts/run_ich_backbone_ablations.py",
            "scripts/check_ich_backbone_ablation.py",
        ]
    )
    (
        OUT
        / ("source_hashes_at_resume.json" if resume else "source_hashes_at_launch.json")
    ).write_text(
        json.dumps(
            {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
            indent=2,
        )
    )
    records = [
        dict(
            backbone=m,
            config=f"rsna_ich_{m}_series_sampling",
            epochs=50,
            status="queued",
        )
        for m in MODELS
    ]

    if resume:
        records = json.loads(status_path.read_text())

    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(status_path)

    save()
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        CUDA_VISIBLE_DEVICES="0,1",
    )
    for record in records:
        if record["status"] == "complete":
            continue
        if record["status"] in {"training", "exporting_validation"}:
            # Existing per-run launcher remains alive and owns training/export.
            while True:
                detail = json.loads(Path(record["detail_queue"]).read_text())
                if all(r["status"] == "complete" for r in detail):
                    record.update(
                        status="complete",
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    save()
                    break
                if any(r["status"] in {"failed", "cancelled"} for r in detail):
                    record.update(status="failed", error="Existing run failed")
                    save()
                    raise RuntimeError(record["error"])
                pid = record.get("existing_launcher_pid")
                if pid and not Path(f"/proc/{pid}").exists():
                    record.update(
                        status="failed",
                        error="Existing launcher exited before completion",
                    )
                    save()
                    raise RuntimeError(record["error"])
                time.sleep(15)
            continue
        if record["status"] != "queued":
            raise RuntimeError(f"Cannot resume state {record['status']}")
        try:
            record.update(
                status="preflight", started_utc=datetime.now(timezone.utc).isoformat()
            )
            save()
            with (OUT / f"{record['backbone']}-preflight.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/check_ich_backbone_ablation.py",
                        record["config"],
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            record["status"] = "training"
            save()
            tag = f"20261108_backbone_{record['backbone']}"
            record["detail_queue"] = str(
                ROOT / f"logs/series_sampling_{tag}/queue.json"
            )
            save()
            with (OUT / f"{record['backbone']}-queue.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_ich_series_sampling_experiments.py",
                        "--config",
                        record["config"],
                        "--epochs",
                        "50",
                        "--tag",
                        tag,
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
