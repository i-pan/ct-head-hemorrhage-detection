"""Monitor teacher export, verify masks, preflight, then train four auxiliary models."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PSEUDO = ROOT / "data/pseudolabels/rsna_three_family_final_20261110"
OUT = ROOT / "logs/ich_auxiliary_20261110"
FAMILIES = ["convnext_tiny", "maxvit_tiny", "effnetv2m", "coatnet0"]
EXPORT_UNIT = "ich-teacher-pseudolabel-export-20261110.service"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue_path = OUT / "queue.json"
    if queue_path.exists():
        raise FileExistsError(f"Refusing duplicate launch: {queue_path}")
    records = [
        dict(
            family=f,
            config=f"rsna_ich_{f}_s16_auxiliary",
            status="waiting_for_export",
            epochs=50,
            segmentation_weight=0.5,
            spatial_weight=0.5,
        )
        for f in FAMILIES
    ]
    state = dict(
        status="waiting_for_export",
        pid=os.getpid(),
        started_unix=time.time(),
        records=records,
    )

    def save():
        state["updated_unix"] = time.time()
        temp = queue_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(queue_path)

    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/check_ich_auxiliary.py",
        ROOT / "scripts/check_ich_auxiliary_data.py",
        ROOT / "scripts/prepare_ich_pseudolabel_training_cache.py",
        ROOT / "scripts/export_ich_checkpoint_predictions.py",
    ]
    files += list((ROOT / "src/skp").rglob("*.py"))
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (OUT / "source_hashes_at_launch.json").write_text(json.dumps(hashes, indent=2))

    def verify_sources():
        for p, h in hashes.items():
            assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, (
                f"Source changed: {p}"
            )

    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
    )

    def run(command, log, check=True):
        with (OUT / log).open("w") as stream:
            return subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=check,
            )

    save()
    try:
        while not (PSEUDO / "completion.json").exists():
            ranks = [
                json.loads(p.read_text())
                for p in sorted(PSEUDO.glob("rank*_status.json"))
            ]
            state["export_progress"] = ranks
            save()
            active = subprocess.run(
                ["systemctl", "--user", "is-active", EXPORT_UNIT],
                capture_output=True,
                text=True,
            )
            if active.stdout.strip() not in ["active", "activating"]:
                raise RuntimeError(
                    "Export stopped without verified completion; training not launched"
                )
            time.sleep(30)
        # The exporter writes the completion marker just before exiting. Wait for
        # exit so the marker is closed and both GPU contexts have been released.
        while subprocess.run(
            ["systemctl", "--user", "is-active", EXPORT_UNIT],
            capture_output=True,
            text=True,
        ).stdout.strip() in ["active", "activating"]:
            time.sleep(2)
        complete = json.loads((PSEUDO / "completion.json").read_text())
        assert (
            complete["status"] == "complete"
            and complete["target_slices"] == 91744
            and complete["teachers"] == 15
        )
        state.update(status="preparing_training_cache", export_completion=complete)
        save()
        verify_sources()
        run(
            [sys.executable, "scripts/prepare_ich_pseudolabel_training_cache.py"],
            "cache.log",
        )
        # Exact baseline batch/optimizer schedule; no automatic batch-size fallback.
        state["status"] = "preflight"
        save()
        for family in FAMILIES:
            run(
                [
                    sys.executable,
                    "scripts/check_ich_auxiliary.py",
                    family,
                    "--device",
                    "cuda:0",
                    "--batch-size",
                    "32",
                ],
                f"{family}-batch32-preflight.log",
            )
        run(
            [sys.executable, "scripts/check_ich_auxiliary_data.py"],
            "data-preflight.log",
        )
        state.update(
            batch_size_per_gpu=32,
            world_size=2,
            accumulate_grad_batches=1,
            effective_batch_size=64,
        )
        for record in records:
            record["status"] = "queued"
        save()
        for record in records:
            verify_sources()
            family, config = record["family"], record["config"]
            run_id = f"series_e050_p0.5_seed88_20261111_{family}_aux05_matched"
            directory = ROOT / "experiments" / config / run_id
            if directory.exists():
                raise FileExistsError(directory)
            command = [
                sys.executable,
                "-m",
                "skp.train",
                config,
                "--run_id",
                run_id,
                "--accelerator",
                "cuda",
                "--devices",
                "2",
                "--strategy",
                "ddp",
                "--precision",
                "bf16-mixed",
                "--check_val_every_n_epoch",
                "5",
                "--num_epochs",
                "50",
            ]
            record.update(
                status="training",
                started_unix=time.time(),
                run_dir=str(directory),
                command=command,
            )
            state["status"] = "training"
            save()
            run(command, f"{family}-train.log")
            record["status"] = "exporting_best_and_final"
            save()
            run(
                [
                    sys.executable,
                    "scripts/export_ich_checkpoint_predictions.py",
                    "--config",
                    config,
                    "--run-dir",
                    str(directory),
                    "--batch-size",
                    "64",
                    "--num-workers",
                    "2",
                ],
                f"{family}-export.log",
            )
            record.update(status="complete", finished_unix=time.time())
            save()
        state["status"] = "complete"
        save()
    except BaseException as error:
        state.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
