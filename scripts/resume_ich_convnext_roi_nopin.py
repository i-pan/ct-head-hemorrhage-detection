"""Two matched baseline ConvNeXt-S16 runs, then classification and BHSD exports."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/ich_convnext_roi_nopin_20261111"
ARMS = {
    "unrestricted": "rsna_ich_convnext_tiny_s16_auxiliary",
    "hu_teacher_union": "rsna_ich_convnext_tiny_s16_auxiliary_roi",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    records = [
        dict(arm=arm, config=config, status="queued", epochs=50)
        for arm, config in ARMS.items()
    ]
    state = dict(
        status="preflight",
        records=records,
        started_unix=time.time(),
        pid=os.getpid(),
        world_size=2,
        batch_size_per_gpu=32,
        accumulate_grad_batches=1,
        segmentation_weight=0.5,
        spatial_weight=0.5,
    )

    def save():
        state["updated_unix"] = time.time()
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(status_path)

    files = list((ROOT / "src/skp").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "scripts/check_ich_auxiliary.py",
        ROOT / "scripts/validate_ich_auxiliary_recovery.py",
        ROOT / "scripts/check_ich_auxiliary_data.py",
        ROOT / "scripts/evaluate_ich_auxiliary_bhsd.py",
        ROOT / "scripts/export_ich_checkpoint_predictions.py",
        ROOT / "scripts/analyze_ich_backbone_ensemble.py",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (OUT / "source_hashes_at_launch.json").write_text(json.dumps(hashes, indent=2))

    def verify():
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

    def run(command, log):
        with (OUT / log).open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    save()
    try:
        for folder, n in [
            ("rsna_three_family_final_20261110", 91744),
            ("rsna_three_family_bhsd_related_20261111", 709),
        ]:
            path = ROOT / "data/pseudolabels" / folder
            completion = json.loads((path / "completion.json").read_text())
            cache = json.loads((path / "training_cache.json").read_text())
            assert (
                completion["status"] == "complete" and completion["target_slices"] == n
            )
            assert completion["export_signature"] == cache["export_signature"]
        run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "scripts/validate_ich_auxiliary_recovery.py",
            ],
            "validation-gate.log",
        )
        assert (
            json.loads((OUT / "validation_gate.json").read_text())["status"]
            == "complete"
        )
        for record in records:
            verify()
            arm, config = record["arm"], record["config"]
            run_id = f"series_e050_p0.5_seed88_20261111_convnext_aux05_{arm}"
            directory = ROOT / "experiments" / config / run_id
            if directory.exists() and arm != "unrestricted":
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
            command += ["--pin_memory", "False"]
            if arm == "unrestricted":
                command += [
                    "--ckpt_path",
                    str(directory / "fold0/checkpoints/last.ckpt"),
                ]
            record.update(
                status="training",
                command=command,
                run_dir=str(directory),
                started_unix=time.time(),
            )
            state["status"] = "training"
            save()
            run(command, f"{arm}-train.log")
            record["status"] = "exporting_classification_best_and_final"
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
                f"{arm}-classification-export.log",
            )
            record["status"] = "evaluating_bhsd_final"
            save()
            run(
                [
                    sys.executable,
                    "scripts/evaluate_ich_auxiliary_bhsd.py",
                    "--config",
                    config,
                    "--run-dir",
                    str(directory),
                ],
                f"{arm}-bhsd-evaluation.log",
            )
            record.update(status="complete", finished_unix=time.time())
            save()
        state["status"] = "complete"
        save()
    except BaseException as error:
        for record in records:
            if record["status"] not in ["queued", "complete"]:
                record["status"] = "failed"
        state.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
