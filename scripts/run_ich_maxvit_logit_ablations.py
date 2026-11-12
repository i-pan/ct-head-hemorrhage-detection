"""Queue spatial-logit MaxViT ablations after NaFlex without interrupting jobs."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/maxvit_logit_ablations_20261108"
DEPENDENCY = ROOT / "logs/naflexvit_ablation_20261108/queue.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "queue.json"
    if path.exists():
        raise FileExistsError(path)
    records = []
    for stride in (32, 16):
        tag = f"20261108_maxvit_logitmax_s{stride}"
        records.append(
            dict(
                backbone=f"maxvit_logitmax_s{stride}",
                config=f"rsna_ich_maxvit_logitmax_s{stride}_series_sampling",
                epochs=50,
                output_stride=stride,
                status="waiting_for_naflex",
                dependency=str(DEPENDENCY),
                tag=tag,
                queued_utc=datetime.now(timezone.utc).isoformat(),
                detail_queue=str(ROOT / f"logs/series_sampling_{tag}/queue.json"),
            )
        )
    sources = [
        ROOT / "scripts/run_ich_maxvit_logit_ablations.py",
        ROOT / "scripts/check_ich_maxvit_logit_ablation.py",
        ROOT / "scripts/run_ich_series_sampling_experiments.py",
        ROOT / "src/skp/models/classification/maxvit_logit_pool.py",
        ROOT / "src/skp/models/classification/net2d.py",
        ROOT / "src/skp/tasks/samplers.py",
        ROOT / "src/skp/configs/rsna_ich_maxvit_tiny_series_sampling.py",
        ROOT / "src/skp/configs/_rsna_ich_effv2m_fixedval_common.py",
    ] + [ROOT / f"src/skp/configs/{r['config']}.py" for r in records]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (OUT / "source_hashes_at_launch.json").write_text(json.dumps(hashes, indent=2))

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2))
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
                    "NaFlex did not complete; inspect dependency before resuming"
                )
            if previous and all(r["status"] in ("complete", "skipped") for r in previous):
                break
            time.sleep(30)
        for r in records:
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
                        "scripts/check_ich_maxvit_logit_ablation.py",
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
        for r in records:
            if r["status"] != "complete":
                r.update(status="failed", error=str(e))
        save()
        raise


if __name__ == "__main__":
    main()
