from skp.toolbox.ich_paths import path as configured_path, teacher_dir, assert_teacher_isolation
"""Resumable training-positive soft segmentation export from15 final teachers."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from skp.configs.rsna_ich_convnext_tiny_logitmax_s16_series_sampling import (
    cfg as classifier_cfg,
)
from skp.datasets.rsna_ich_2p5d import Dataset
from skp.toolbox.bhsd_convnext import Net as ConvNeXt, ROOT
from skp.toolbox.bhsd_teacher_backbones import Net as OtherTeacher

DEFAULT_OUT = configured_path("ICH_PSEUDOLABELS", ROOT / "data/pseudolabels/rsna_three_family_final_20261110")
SHARD_SIZE = 256
FAMILIES = ["convnext_tiny", "maxvit_tiny", "coatnet0"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def dataset():
    cfg = classifier_cfg.__deepcopy__()
    cfg.train_transforms = None
    cfg.depth_flip_p = cfg.horizontal_flip_p = cfg.vertical_flip_p = 0.0
    return Dataset(cfg, "train")


def prepare(out):
    ds = dataset()
    positive = ds.df[ds.df["any"] > 0].copy()
    bhsd = pd.read_csv(configured_path("ICH_BHSD_MANIFEST", ROOT / "data/bhsd/bhsd_segmentation_slices.csv"))
    excluded = positive[positive.patient_id.isin(bhsd.patient_id)].copy()
    target = positive[~positive.patient_id.isin(bhsd.patient_id)].reset_index(drop=True)
    split = pd.read_csv(classifier_cfg.annotations_file)
    assert_teacher_isolation(bhsd, split)
    assert set(target.patient_id).isdisjoint(
        split.loc[split.split.isin(["val", "test"]), "patient_id"]
    )
    assert (target.split == "train").all() and not target.excluded_bhsd.any()
    assert target.slice_path.is_unique and (target["any"] == 1).all()
    target.insert(0, "target_index", np.arange(len(target)))
    target["prediction_shard"] = [
        f"shards/{i // SHARD_SIZE:05d}.npz" for i in target.target_index
    ]
    target["prediction_offset"] = target.target_index % SHARD_SIZE
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    if manifest.exists():
        pd.testing.assert_frame_equal(pd.read_csv(manifest), target)
    else:
        target.to_csv(manifest, index=False)
    excluded.to_csv(out / "excluded_bhsd_related_positive_slices.csv", index=False)
    teachers = []
    for family in FAMILIES:
        for fold in range(5):
            root = (
                ROOT
                / "experiments/bhsd_convnext_s16_binary_ablations_20261110/p100_batch"
                if family == "convnext_tiny"
                else ROOT / "experiments/bhsd_teacher_backbones_20261110" / family
            ) / f"fold{fold}"
            root = teacher_dir(family, fold, root)
            status = json.loads((root / "status.json").read_text())
            config = json.loads((root / "config.json").read_text())
            assert (
                status["status"] == "complete"
                and status["epoch"] == 25
                and status["step"] == status["total_steps"]
            )
            assert (
                config["dice_reduction"] == "batch"
                and config["positive_fraction"] == 1.0
            )
            path = root / "last.pt"
            teachers.append(
                dict(
                    family=family,
                    fold=fold,
                    checkpoint=str(path),
                    sha256=sha(path),
                    step=status["step"],
                )
            )
    provenance = dict(
        target_slices=len(target),
        target_series=target.series_uid.nunique(),
        target_patients=target.patient_id.nunique(),
        original_positive_slices=len(positive),
        excluded_bhsd_related_positive_slices=len(excluded),
        manifest_sha256=sha(manifest),
        teachers=teachers,
        shard_size=SHARD_SIZE,
        input="Original classification 16-bit PNG HU processing,3 contiguous slices x3 windows,512x512,zero missing neighbors,no augmentation",
        targets="Only non-BHSD-patient training positives without manual BHSD masks. Negative slices implicit empty masks; val/test excluded.",
        aggregation="Arithmetic mean of15 sigmoid probabilities in float32, no sharpening/no threshold; float16 storage. std is population standard deviation across15 teachers, not standard error or calibrated uncertainty.",
        output="Each shard: probability[N,512,512], disagreement_std[N,512,512], target_index[N], export_signature. Global manifest maps image paths to shard offsets.",
        source_hashes={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "src/skp/toolbox/bhsd_convnext.py",
                ROOT / "src/skp/toolbox/bhsd_teacher_backbones.py",
                ROOT / "src/skp/datasets/rsna_ich_2p5d.py",
            ]
        },
    )
    signature = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    provenance["export_signature"] = signature
    if (out / "provenance.json").exists():
        assert json.loads((out / "provenance.json").read_text()) == provenance
    else:
        write_json(out / "provenance.json", provenance)
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in [
                    "target_slices",
                    "target_series",
                    "target_patients",
                    "excluded_bhsd_related_positive_slices",
                    "export_signature",
                ]
            }
        ),
        flush=True,
    )


def load_teachers(provenance):
    models = []
    for family in FAMILIES:
        template = ConvNeXt() if family == "convnext_tiny" else OtherTeacher(family)
        for entry in [t for t in provenance["teachers"] if t["family"] == family]:
            assert sha(entry["checkpoint"]) == entry["sha256"]
            state = torch.load(
                entry["checkpoint"], map_location="cpu", weights_only=False, mmap=True
            )
            assert state["epoch"] == 25 and state["global_step"] == entry["step"]
            model = copy.deepcopy(template)
            model.load_state_dict(state["model"], strict=True)
            models.append(model.eval().cuda())
            print(f"Loaded {family} fold{entry['fold']} final", flush=True)
            del state
        del template
    assert len(models) == 15
    return models


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


@torch.inference_mode()
def predict(models, x):
    mean = torch.zeros((len(x), 512, 512), device="cuda", dtype=torch.float32)
    m2 = torch.zeros_like(mean)
    for n, model in enumerate(models, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        p = logits[:, 0].float().sigmoid()
        delta = p - mean
        mean += delta / n
        m2 += delta * (p - mean)
    std = (m2 / len(models)).clamp_min(0).sqrt()
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert mean.min() >= 0 and mean.max() <= 1
    return mean.cpu().numpy().astype(np.float16), std.cpu().numpy().astype(np.float16)


def run(args):
    out = args.output
    provenance = json.loads((out / "provenance.json").read_text())
    assert sha(out / "manifest.csv") == provenance["manifest_sha256"]
    for p, digest in provenance["source_hashes"].items():
        assert sha(ROOT / p) == digest, f"Source changed: {p}"
    manifest = pd.read_csv(out / "manifest.csv")
    n = len(manifest)
    output = out / "preflight" if args.smoke else out
    (output / "shards").mkdir(parents=True, exist_ok=True)
    selected = []
    for shard, start in enumerate(range(0, n, SHARD_SIZE)):
        if shard % args.world_size != args.rank:
            continue
        path = output / "shards" / f"{shard:05d}.npz"
        ids = np.arange(start, min(start + SHARD_SIZE, n))
        if path.exists():
            with np.load(path) as saved:
                assert str(saved["export_signature"]) == provenance["export_signature"]
                assert np.array_equal(saved["target_index"], ids)
            continue
        selected.extend(ids.tolist())
        if args.smoke:
            break
    if not selected:
        write_json(
            output / f"rank{args.rank}_status.json",
            dict(status="complete", remaining_slices=0),
        )
        return
    ds = dataset()
    # Neighbors were constructed on all slices before selecting positive centers.
    available = ds.df.set_index("slice_path").loc[manifest.slice_path].reset_index()
    for column in [
        "patient_id",
        "study_id",
        "series_id",
        "prev_slice_path",
        "next_slice_path",
        "any",
        "rescale_slope",
        "rescale_intercept",
    ]:
        pd.testing.assert_series_equal(
            available[column], manifest[column], check_names=False
        )
    ds.df = manifest.reset_index(drop=True)
    loader = DataLoader(
        Subset(ds, selected),
        batch_size=32,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=worker_init,
    )
    models = load_teachers(provenance)
    tick = time.monotonic()
    prob_parts = []
    std_parts = []
    id_parts = []
    completed = 0
    for batch in loader:
        ids = batch["index"].numpy()
        assert (ids // SHARD_SIZE == ids[0] // SHARD_SIZE).all()
        prob, std = predict(models, batch["x"].cuda(non_blocking=True))
        prob_parts.append(prob)
        std_parts.append(std)
        id_parts.append(ids)
        if ids[-1] % SHARD_SIZE != SHARD_SIZE - 1 and ids[-1] != n - 1:
            continue
        shard = int(ids[0] // SHARD_SIZE)
        prob = np.concatenate(prob_parts)
        std = np.concatenate(std_parts)
        indices = np.concatenate(id_parts)
        assert np.array_equal(
            indices, np.arange(shard * SHARD_SIZE, min((shard + 1) * SHARD_SIZE, n))
        )
        path = output / "shards" / f"{shard:05d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            probability=prob,
            disagreement_std=std,
            target_index=indices,
            export_signature=np.array(provenance["export_signature"]),
        )
        # Verify the saved array metadata and exact roundtrip before committing a shard.
        with np.load(tmp) as saved:
            assert np.array_equal(saved["probability"], prob) and np.array_equal(
                saved["disagreement_std"], std
            )
        tmp.replace(path)
        pd.DataFrame(
            dict(
                target_index=indices,
                max_probability=prob.reshape(len(prob), -1).max(1),
                mean_disagreement=std.reshape(len(std), -1).astype(np.float32).mean(1),
                pixels_ge_06=(prob >= 0.6).sum((1, 2)),
            )
        ).to_csv(output / "shards" / f"{shard:05d}_summary.csv", index=False)
        completed += len(indices)
        elapsed = time.monotonic() - tick
        status = dict(
            status="complete" if completed == len(selected) else "running",
            rank=args.rank,
            newly_completed_slices=completed,
            planned_slices_this_invocation=len(selected),
            last_shard=shard,
            elapsed_seconds=elapsed,
            slices_per_second=completed / elapsed,
            eta_seconds=(len(selected) - completed) / (completed / elapsed),
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        write_json(output / f"rank{args.rank}_status.json", status)
        print(json.dumps(status), flush=True)
        prob_parts.clear()
        std_parts.clear()
        id_parts.clear()
    assert completed == len(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    assert 0 <= args.rank < args.world_size
    torch.set_num_threads(4)
    if args.prepare:
        prepare(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
