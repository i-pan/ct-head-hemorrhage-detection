"""Run independent positive-pass schedules on two GPUs; stop queue on failure."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/bhsd_convnext_schedules_20261110"
RESULTS = ROOT / "experiments/bhsd_convnext_s16_binary_schedules_20261110"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue_path = OUT / "queue.json"
    if queue_path.exists():
        raise FileExistsError(queue_path)
    sources = [
        "scripts/run_bhsd_convnext_schedules.py",
        "scripts/train_bhsd_convnext_schedule.py",
        "src/skp/toolbox/bhsd_convnext.py",
        "src/skp/models/segmentation/decoders/unet.py",
        "src/skp/configs/_rsna_ich_effv2m_fixedval_common.py",
    ]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    records = [
        dict(
            passes=p,
            fold=f,
            status="queued",
            output=str(RESULTS / f"passes_{p:02d}" / f"fold{f}"),
        )
        for p in (10, 25, 50)
        for f in range(5)
    ]

    def save():
        tmp = queue_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(queue_path)

    save()
    active = {}
    failed = False
    while active or (not failed and any(r["status"] == "queued" for r in records)):
        for gpu, (proc, record, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                record.update(
                    status="complete" if code == 0 else "failed",
                    exit_code=code,
                    finished_unix=time.time(),
                )
                failed |= code != 0
                del active[gpu]
                save()
        if not failed:
            for gpu in (0, 1):
                queued = next((r for r in records if r["status"] == "queued"), None)
                if gpu in active or queued is None:
                    continue
                for path, digest in hashes.items():
                    assert (
                        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                    ), f"Source changed: {path}"
                log_path = (
                    OUT / f"passes{queued['passes']:02d}_fold{queued['fold']}.log"
                )
                log = log_path.open("w")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    PYTHONUNBUFFERED="1",
                )
                cmd = [
                    sys.executable,
                    "scripts/train_bhsd_convnext_schedule.py",
                    "--fold",
                    str(queued["fold"]),
                    "--passes",
                    str(queued["passes"]),
                    "--output",
                    queued["output"],
                ]
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                queued.update(
                    status="running",
                    gpu=gpu,
                    pid=proc.pid,
                    started_unix=time.time(),
                    log=str(log_path),
                    command=cmd,
                )
                active[gpu] = (proc, queued, log)
                save()
        time.sleep(5)
    if failed:
        raise SystemExit("A run failed; remaining queued runs were not started.")


if __name__ == "__main__":
    main()
