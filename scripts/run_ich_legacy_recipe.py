"""Explicit stages for the prior MaxViT/BiGRU/frozen-DeepLab release recipe.
Default is a command plan; --execute opts into a single stage. Never deletes partial outputs.
"""

import argparse, json, os, subprocess, sys
from pathlib import Path

C = "rsna_ich_maxvit_tiny_fixedval_9ch"
S = "rsna_ich_maxvit_tiny_sequence_bigru_selected"
T = "bhsd_maxvit_tiny_seg_9ch"
P = "rsna_ich_joint_maxvit_tiny_9ch_deeplab_frozen"
F = "bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=[
            "classifier",
            "features",
            "sequence",
            "teachers",
            "pseudolabels",
            "decoder-pretrain",
            "decoders",
            "package",
        ],
    )
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    w = a.work.resolve()
    runs = w / "runs"

    def root(c):
        return runs / c / "recipe"

    classifier = root(C) / "fold0/checkpoints/last.ckpt"
    seq = root(S) / "fixed_split/checkpoints/best.ckpt"
    env = dict(
        os.environ,
        ICH_LEGACY_CLASSIFIER=str(classifier),
        ICH_LEGACY_SEQUENCE=str(seq),
        ICH_LEGACY_FEATURES=str(w / "features"),
        ICH_LEGACY_PSEUDOLABELS=str(w / "pseudolabels"),
        ICH_LEGACY_DECODER_PRETRAIN=str(root(P) / "fold0/checkpoints/last.ckpt"),
        NCCL_SOCKET_IFNAME="lo",
    )
    commands = []

    def add(*x):
        commands.append([sys.executable, *map(str, x)])

    def train(c, cv=False):
        if a.execute and root(c).exists():
            raise FileExistsError(root(c))
        extra = ["--kfold", "0,1,2,3,4"] if cv else []
        add(
            "-m",
            "skp.train",
            c,
            "--save_dir",
            runs,
            "--run_id",
            "recipe",
            "--accelerator",
            "cuda",
            "--devices",
            2,
            "--strategy",
            "ddp",
            "--precision",
            "bf16-mixed",
            "--keep_pretrained_weights_path",
            *extra,
        )

    if a.stage == "classifier":
        train(C)
    elif a.stage == "features":
        add(
            "scripts/extract_rsna_ich_features.py",
            "--config",
            C,
            "--checkpoint",
            classifier,
            "--output-dir",
            w / "features",
            "--splits",
            "train",
            "val",
            "--devices",
            "0,1",
        )
    elif a.stage == "sequence":
        train(S)
    elif a.stage == "teachers":
        train(T, True)
    elif a.stage == "pseudolabels":
        if a.execute and ((w / "pseudolabels").exists() or (w / "oof").exists()):
            raise FileExistsError(
                "Partial outputs preserved; use a new work directory or explicitly inspect/recover them."
            )
        add(
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "oof-threshold",
            "--bhsd-config",
            T,
            "--checkpoint-root",
            root(T),
            "--output-dir",
            w / "oof",
        )
        add(
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "pseudolabel-rsna",
            "--bhsd-config",
            T,
            "--rsna-config",
            C,
            "--checkpoint-root",
            root(T),
            "--threshold-file",
            w / "oof/threshold_summary.json",
            "--output-dir",
            w / "pseudolabels",
            "--splits",
            "train",
            "--temperature",
            1,
        )
        add(
            "scripts/generate_bhsd_9ch_pseudolabels.py",
            "merge-rsna-manifests",
            "--output-dir",
            w / "pseudolabels",
        )
    elif a.stage == "decoder-pretrain":
        train(P)
    elif a.stage == "decoders":
        train(F, True)
    else:
        add(
            "scripts/package_ich_legacy.py",
            "--classifier",
            classifier,
            "--sequence",
            seq,
            "--decoders",
            root(F),
            "--output",
            w / "model",
        )
    print(
        json.dumps(
            {
                "environment": {
                    k: v for k, v in env.items() if k.startswith("ICH_LEGACY")
                },
                "commands": commands,
            },
            indent=2,
        )
    )
    if a.execute:
        import pandas as pd
        from skp.toolbox.ich_paths import assert_teacher_isolation

        for key in ["ICH_RSNA_ROOT", "ICH_SPLITS", "ICH_BHSD_MANIFEST"]:
            if not env.get(key):
                raise ValueError(f"Set {key}")
        assert_teacher_isolation(
            pd.read_csv(env["ICH_BHSD_MANIFEST"]), pd.read_csv(env["ICH_SPLITS"])
        )
        w.mkdir(parents=True, exist_ok=True)
        (w / "recipe.json").write_text(
            json.dumps(
                {
                    "recipe": "prior MaxViT+BiGRU+frozen DeepLab",
                    "classifier": str(classifier),
                    "sequence": str(seq),
                    "decoders": str(root(F)),
                    "holdout": "teacher-patient-disjoint; differs from historical split",
                },
                indent=2,
            )
        )
        for cmd in commands:
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
