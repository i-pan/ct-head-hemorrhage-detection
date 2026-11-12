from skp.toolbox.ich_paths import path
"""Run both GPU export shards and verify completion; never launches training."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = path("ICH_PSEUDOLABELS", ROOT / "data/pseudolabels/rsna_three_family_final_20261110")
LOG = ROOT / "logs/ich_teacher_pseudolabel_export_20261110"


def save(value):
    p = LOG / "queue.json"
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(value, indent=2))
    t.replace(p)


def main():
    LOG.mkdir(exist_ok=True, parents=True)
    provenance = json.loads((OUT / "provenance.json").read_text())
    records = []
    active = []
    for rank in (0, 1):
        log = (LOG / f"rank{rank}.log").open("a")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(rank),
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            PYTHONUNBUFFERED="1",
        )
        cmd = [
            sys.executable,
            "scripts/export_ich_teacher_pseudolabels.py",
            "--rank",
            str(rank),
            "--world-size",
            "2",
        ]
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        r = dict(
            rank=rank,
            gpu=rank,
            pid=proc.pid,
            status="running",
            started_unix=time.time(),
            log=str(LOG / f"rank{rank}.log"),
            command=cmd,
        )
        records.append(r)
        active.append((proc, r, log))
    save(records)
    while active:
        for proc, r, log in list(active):
            code = proc.poll()
            if code is not None:
                r.update(
                    status="complete" if code == 0 else "failed",
                    exit_code=code,
                    finished_unix=time.time(),
                )
                log.close()
                active.remove((proc, r, log))
                save(records)
        if active:
            time.sleep(5)
    if any(r["exit_code"] for r in records):
        raise SystemExit(
            "An export worker failed; completed shards retained for resume."
        )
    n = provenance["target_slices"]
    size = provenance["shard_size"]
    summaries = []
    for start in range(0, n, size):
        shard = start // size
        p = OUT / "shards" / f"{shard:05d}.npz"
        with np.load(p) as a:
            assert str(a["export_signature"]) == provenance["export_signature"]
            assert np.array_equal(
                a["target_index"], np.arange(start, min(start + size, n))
            )
            summary_path = OUT / "shards" / f"{shard:05d}_summary.csv"
            # Recover if a worker stopped after committing the map but before its CSV.
            if not summary_path.exists():
                prob, std = a["probability"], a["disagreement_std"]
                pd.DataFrame(
                    dict(
                        target_index=a["target_index"],
                        max_probability=prob.reshape(len(prob), -1).max(1),
                        mean_disagreement=std.reshape(len(std), -1)
                        .astype(np.float32)
                        .mean(1),
                        pixels_ge_06=(prob >= 0.6).sum((1, 2)),
                    )
                ).to_csv(summary_path, index=False)
        summaries.append(pd.read_csv(OUT / "shards" / f"{shard:05d}_summary.csv"))
    summary = pd.concat(summaries, ignore_index=True)
    assert np.array_equal(summary.target_index.to_numpy(), np.arange(n))
    summary.to_csv(OUT / "prediction_summary.csv", index=False)
    (OUT / "completion.json").write_text(
        json.dumps(
            dict(
                status="complete",
                target_slices=n,
                teachers=15,
                shards=len(summaries),
                finished_unix=time.time(),
                export_signature=provenance["export_signature"],
                slices_without_pixels_ge_06=int((summary.pixels_ge_06 == 0).sum()),
                note="Low teacher scores retained as soft predictions; do not interpret as confirmed empty masks for these known-positive slices.",
            ),
            indent=2,
        )
    )
    print(f"Export verified: {n} positive slices,15 teachers.", flush=True)


if __name__ == "__main__":
    main()
