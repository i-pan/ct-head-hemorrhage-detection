"""Sequential strict backbone ablations for stride-16 spatial logit pooling."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/spatial_backbones_20261109"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "queue.json"
    if path.exists():
        raise FileExistsError(path)
    records = []
    for n in ["convnext_tiny", "effnetv2m", "coatnet0"]:
        tag = f"20261109_{n}_spatial_s16"
        records.append(
            dict(
                backbone=n,
                config=f"rsna_ich_{n}_logitmax_s16_series_sampling",
                status="queued",
                epochs=50,
                tag=tag,
                detail_queue=str(ROOT / f"logs/series_sampling_{tag}/queue.json"),
            )
        )
    files = [
        ROOT / "scripts/run_ich_spatial_backbones.py",
        ROOT / "scripts/check_ich_spatial_backbones.py",
        ROOT / "scripts/run_ich_series_sampling_experiments.py",
        ROOT / "src/skp/models/classification/spatial_logit_backbones.py",
        ROOT / "src/skp/models/classification/net2d.py",
        ROOT / "src/skp/tasks/samplers.py",
        ROOT / "src/skp/configs/rsna_ich_maxvit_logitmax_s16_series_sampling.py",
    ] + [ROOT / f"src/skp/configs/{r['config']}.py" for r in records]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (OUT / "source_hashes_at_launch.json").write_text(json.dumps(hashes, indent=2))

    def save():
        t = path.with_suffix(".tmp")
        t.write_text(json.dumps(records, indent=2))
        t.replace(path)

    save()
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        CUDA_VISIBLE_DEVICES="0,1",
    )
    for r in records:
        try:
            for p, h in hashes.items():
                assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, (
                    f"Source changed: {p}"
                )
            r.update(
                status="preflight", started_utc=datetime.now(timezone.utc).isoformat()
            )
            save()
            with (OUT / (r["backbone"] + "-preflight.log")).open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/check_ich_spatial_backbones.py",
                        r["config"],
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            r["status"] = "training"
            save()
            with (OUT / (r["backbone"] + "-queue.log")).open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_ich_series_sampling_experiments.py",
                        "--config",
                        r["config"],
                        "--epochs",
                        "50",
                        "--tag",
                        r["tag"],
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            r.update(
                status="complete", finished_utc=datetime.now(timezone.utc).isoformat()
            )
            save()
        except BaseException as e:
            r.update(status="failed", error=str(e))
            save()
            raise


if __name__ == "__main__":
    main()
