"""Portable selected recipe: classifiers -> teachers -> masks -> students.

All stages are explicit; default prints commands. --execute runs one stage.
Environment paths and checkpoint registry are documented in docs/ICH_PIPELINE.md.
"""

import argparse, json, os, subprocess, sys
from pathlib import Path

FAMILIES = ["convnext_tiny", "maxvit_tiny", "coatnet0"]
BASE = {
    "convnext_tiny": "rsna_ich_convnext_tiny_logitmax_s16_series_sampling",
    "maxvit_tiny": "rsna_ich_maxvit_logitmax_s16_series_sampling",
    "coatnet0": "rsna_ich_coatnet0_logitmax_s16_series_sampling",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["classifiers", "teachers", "pseudolabels", "students"]
    )
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    registry = {
        "classifiers": {
            f: str(work / "classifiers" / BASE[f] / f / "fold0/checkpoints/last.ckpt")
            for f in FAMILIES
        },
        "teachers": {f: str(work / "teachers" / f) for f in FAMILIES},
        "students": {
            f: str(work / "students" / f"rsna_ich_{f}_s16_auxiliary_roi_w025" / f)
            for f in FAMILIES
        },
    }
    reg = work / "checkpoints.json"
    if reg.exists():
        assert json.loads(reg.read_text()) == registry
    else:
        reg.write_text(json.dumps(registry, indent=2))
    (work / "student_runs.json").write_text(
        json.dumps(
            {
                ("maxvit" if f == "maxvit_tiny" else f): root
                for f, root in registry["students"].items()
            },
            indent=2,
        )
    )
    env = dict(
        os.environ,
        ICH_CHECKPOINT_REGISTRY=str(reg),
        ICH_PSEUDOLABELS=str(work / "pseudolabels"),
        ICH_PSEUDOLABEL_SUPPLEMENT="",
        OMP_NUM_THREADS="4",
        NCCL_SOCKET_IFNAME="lo",
    )
    required = [
        "ICH_RSNA_ROOT",
        "ICH_SPLITS",
        "ICH_BHSD_ROOT",
        "ICH_BHSD_MANIFEST",
        "ICH_BHSD_HU_CACHE",
    ]
    if a.execute:
        for key in required:
            if not env.get(key):
                raise ValueError(f"Set {key}; see docs/ICH_PIPELINE.md")
        from skp.toolbox.ich_paths import assert_teacher_isolation
        import pandas as pd

        assert_teacher_isolation(
            pd.read_csv(env["ICH_BHSD_MANIFEST"]), pd.read_csv(env["ICH_SPLITS"])
        )
    commands = []

    def add(*args):
        commands.append([sys.executable, *map(str, args)])

    if a.stage in ["classifiers", "students"]:
        for f in FAMILIES:
            config = (
                BASE[f]
                if a.stage == "classifiers"
                else f"rsna_ich_{f}_s16_auxiliary_roi_w025"
            )
            root = work / a.stage / config / f
            if a.execute and root.exists():
                raise FileExistsError(f"{root}: existing run; refusing overwrite")
            add(
                "-m",
                "skp.train",
                config,
                "--save_dir",
                work / a.stage,
                "--run_id",
                f,
                "--accelerator",
                "cuda",
                "--devices",
                2,
                "--strategy",
                "ddp",
                "--precision",
                "bf16-mixed",
                "--num_epochs",
                50,
                "--check_val_every_n_epoch",
                5,
            )
            if a.stage == "students":
                add(
                    "scripts/export_ich_checkpoint_predictions.py",
                    "--config",
                    config,
                    "--run-dir",
                    root,
                    "--batch-size",
                    64,
                    "--num-workers",
                    2,
                )
                add(
                    "scripts/evaluate_ich_auxiliary_bhsd.py",
                    "--config",
                    config,
                    "--run-dir",
                    root,
                )
    elif a.stage == "teachers":
        add("scripts/train_bhsd_convnext_schedule.py", "--prepare")
        for f in FAMILIES:
            for fold in range(5):
                add(
                    "scripts/train_bhsd_convnext_schedule.py",
                    "--backbone",
                    f,
                    "--fold",
                    fold,
                    "--passes",
                    25,
                    "--positive-fraction",
                    1,
                    "--dice-reduction",
                    "batch",
                    "--output",
                    work / "teachers" / f / f"fold{fold}",
                )
    else:
        add("scripts/export_ich_teacher_pseudolabels.py", "--prepare")
        add("scripts/run_ich_teacher_pseudolabel_export.py")
        add("scripts/prepare_ich_pseudolabel_training_cache.py")
    print(
        json.dumps(
            {
                "environment": {
                    k: env[k]
                    for k in [
                        "ICH_CHECKPOINT_REGISTRY",
                        "ICH_PSEUDOLABELS",
                        "ICH_PSEUDOLABEL_SUPPLEMENT",
                    ]
                },
                "commands": commands,
            },
            indent=2,
        ),
        flush=True,
    )
    if a.execute:
        for command in commands:
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
