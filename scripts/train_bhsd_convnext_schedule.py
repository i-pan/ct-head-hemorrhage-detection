"""Single-GPU positive-only schedule study. Each epoch is one exact positive pass."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from skp.toolbox.bhsd_convnext import (
    CHECKPOINT,
    MANIFEST,
    Dataset,
    MixedSliceBatchSampler,
    Net,
    THRESHOLDS,
    learning_rates,
    loss_parts,
    prepare_cache,
    summarize_counts,
)


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    data = torch.utils.data.get_worker_info().dataset
    if data.augment:
        data.augment.set_random_seed(seed)
    torch.set_num_threads(1)


def resources():
    def rss(pid):
        try:
            return (
                int(Path(f"/proc/{pid}/statm").read_text().split()[1])
                * os.sysconf("SC_PAGE_SIZE")
                / 2**30
            )
        except (FileNotFoundError, ProcessLookupError):
            return 0.0

    children = (
        Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
    )
    ram = sum(rss(pid) for pid in [os.getpid(), *children])
    mem = {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return dict(
        process_tree_rss_gib=ram,
        host_ram_used_gib=(mem["MemTotal"] - mem["MemAvailable"]) / 2**20,
        gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30,
    )


@torch.inference_mode()
def validate(model, loader, output):
    model.eval()
    n = len(loader.dataset)
    probs = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(n, 512, 512)
    )
    counts = np.zeros((n, len(THRESHOLDS), 3), np.float64)
    losses = np.zeros(3)
    for x, y, indices in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        parts = loss_parts(logits, y)
        losses += np.asarray([v.item() for v in parts]) * len(x)
        p = logits.float().sigmoid()[:, 0]
        target = y[:, 0] > 0.5
        ids = indices.numpy()
        probs[ids] = p.cpu().numpy().astype(np.float16)
        for j, t in enumerate(THRESHOLDS):
            pred = p >= float(t)
            counts[ids, j] = (
                torch.stack(
                    ((pred & target).sum((1, 2)), pred.sum((1, 2)), target.sum((1, 2))),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
    probs.flush()
    del probs
    result = summarize_counts(counts, loader.dataset.df.series_uid.to_numpy())
    result.update(
        dict(zip(("val_loss", "val_dice_loss", "val_bce_loss"), (losses / n).tolist()))
    )
    return result, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        default="convnext_tiny",
        choices=("convnext_tiny", "maxvit_tiny", "effnetv2m", "coatnet0"),
    )
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--passes", type=int, default=10, choices=(10, 25, 50))
    ap.add_argument(
        "--positive-fraction", type=float, default=1.0, choices=(1.0, 0.75, 0.5)
    )
    ap.add_argument("--dice-reduction", default="slice", choices=("slice", "batch"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        print(json.dumps(dict(prepared_series=len(prepare_cache()))), flush=True)
        return
    from skp.toolbox.ich_paths import assert_teacher_isolation, path
    import pandas as pd
    assert_teacher_isolation(pd.read_csv(MANIFEST), pd.read_csv(path("ICH_SPLITS", "data/folds/rsna_ich_fixed_val_splits.csv")))
    torch.manual_seed(88)
    np.random.seed(88)
    random.seed(88)
    torch.backends.cudnn.benchmark = False
    train, val = (
        Dataset(args.fold, True, include_negatives=args.positive_fraction < 1),
        Dataset(args.fold, False),
    )
    positive_count = int(train.df.has_mask.sum())
    samples_per_epoch = positive_count
    assert set(train.df.patient_id).isdisjoint(val.df.patient_id)
    if args.positive_fraction == 1:
        assert train.df.has_mask.all()
    generator = torch.Generator().manual_seed(88)
    sampler = None
    loader_options = dict(
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_init,
        generator=generator,
    )
    if args.positive_fraction == 1:
        loader = DataLoader(train, batch_size=32, shuffle=True, **loader_options)
    else:
        sampler = MixedSliceBatchSampler(
            train.df.has_mask.to_numpy(), args.positive_fraction, samples_per_epoch
        )
        loader = DataLoader(train, batch_sampler=sampler, **loader_options)
    vloader = DataLoader(
        val,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_init,
    )
    checkpoint_path = CHECKPOINT
    if args.backbone == "convnext_tiny":
        model = Net().cuda()
    else:
        from skp.toolbox.bhsd_teacher_backbones import (
            Net as TeacherNet,
            classifier_checkpoint,
        )

        checkpoint_path = classifier_checkpoint(args.backbone)
        model = TeacherNet(args.backbone).cuda()
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.encoder.parameters(), lr=3e-5),
            dict(
                params=list(model.decoder.parameters()) + list(model.head.parameters()),
                lr=3e-4,
            ),
        ],
        weight_decay=0.01,
    )
    steps_per_pass = len(loader)
    total = args.passes * steps_per_pass
    if args.smoke:
        model.train()
        for step, (x, y, _) in enumerate(loader, 1):
            if step > 2:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.cuda())
            assert out.shape == (32, 1, 512, 512)
            loss = loss_parts(out, y.cuda(), args.dice_reduction)[0]
            loss.backward()
            assert torch.isfinite(loss) and all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            assert all(
                p.grad is not None
                for p in model.encoder.parameters()
                if p.requires_grad
            )
            optimizer.step()
            print(
                json.dumps(dict(smoke_step=step, loss=loss.item(), **resources())),
                flush=True,
            )
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert model(x.cuda()).shape == (32, 1, 512, 512)
        tick = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bhsd-preflight-") as temp:
            metrics, counts = validate(model, vloader, Path(temp) / "predictions.npy")
            assert counts.shape == (len(val), 9, 3)
            assert all(np.isfinite(v) for v in metrics.values())
            assert np.load(Path(temp) / "predictions.npy", mmap_mode="r").shape == (
                len(val),
                512,
                512,
            )
        print(
            json.dumps(
                dict(
                    smoke_validation_seconds=time.monotonic() - tick,
                    metrics=metrics,
                    **resources(),
                )
            ),
            flush=True,
        )
        return
    if args.output is None:
        ap.error("--output required")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    train.df.to_csv(out / "train_manifest.csv", index=False)
    val.df.to_csv(out / "validation_manifest.csv", index=False)
    provenance = dict(
        backbone=args.backbone,
        encoder_geometry=getattr(model, "geometry", None),
        fold=args.fold,
        passes=args.passes,
        seed=88,
        train_positive_slices=positive_count,
        train_available_slices=len(train),
        samples_per_epoch=samples_per_epoch,
        positive_fraction=args.positive_fraction,
        dice_reduction=args.dice_reduction,
        validation_slices=len(val),
        batch_size=32,
        steps_per_pass=steps_per_pass,
        total_steps=total,
        warmup_steps=math.ceil(total * 0.05),
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=hashlib.file_digest(
            checkpoint_path.open("rb"), "sha256"
        ).hexdigest(),
        manifest_sha256=hashlib.file_digest(MANIFEST.open("rb"), "sha256").hexdigest(),
        input="3 contiguous slices x 3 HU windows; original NIfTI HU; classification windowing and [-1,1] normalization; zero missing neighbors",
        loss=f"per-{args.dice_reduction} soft Dice + mean BCE, equal weights",
        validation_loss="Common per-slice Dice + mean BCE for every arm; unchanged from baseline",
        sampling="Positive-only uses original shuffle; mixed uses independent cycling bins with exact full-batch ratio; baseline sample/update count and final partial batch preserved",
        primary_metric="positive_slice_dice_05",
        decoder="U-Net skips at strides 16,8,4 then upsample to 2,1",
        optimizer="AdamW wd=.01 encoder3e-5 decoder3e-4; ratio10 throughout;5%linear warmup+cosine floor decoder1e-6 encoder1e-7",
        thresholds=THRESHOLDS.tolist(),
    )
    write_json(out / "config.json", provenance)
    shutil.copy2(__file__, out / "training_source.py")
    import skp.toolbox.bhsd_convnext as implementation

    shutil.copy2(implementation.__file__, out / "model_data_source.py")
    if args.backbone != "convnext_tiny":
        import skp.toolbox.bhsd_teacher_backbones as backbone_implementation

        shutil.copy2(backbone_implementation.__file__, out / "backbone_source.py")
    step = 0
    best = -1
    started = time.monotonic()
    for epoch in range(1, args.passes + 1):
        model.train()
        sums = np.zeros(3)
        seen = 0
        positive_seen = 0
        tick = time.monotonic()
        for x, y, indices in loader:
            step += 1
            lr = learning_rates(step, total)
            for group, rate in zip(optimizer.param_groups, lr):
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            parts = loss_parts(logits, y.cuda(non_blocking=True), args.dice_reduction)
            if not torch.isfinite(parts[0]):
                raise RuntimeError(f"Non-finite loss at step {step}")
            parts[0].backward()
            optimizer.step()
            sums += np.asarray([v.item() for v in parts]) * len(x)
            seen += len(x)
            positive_seen += int(train.df.iloc[indices.numpy()].has_mask.sum())
            if step % 10 == 0:
                write_json(
                    out / "status.json",
                    dict(
                        status="training",
                        epoch=epoch,
                        step=step,
                        total_steps=total,
                        completed_passes=epoch - 1,
                        samples_this_pass=seen,
                        positive_slices=positive_count,
                        encoder_lr=lr[0],
                        decoder_lr=lr[1],
                        elapsed_seconds=time.monotonic() - started,
                        **resources(),
                    ),
                )
        assert seen == samples_per_epoch
        train_seconds = time.monotonic() - tick
        write_json(
            out / "status.json",
            dict(
                status="validating",
                epoch=epoch,
                step=step,
                total_steps=total,
                **resources(),
            ),
        )
        tick = time.monotonic()
        metrics, counts = validate(model, vloader, out / "predictions_pending.npy")
        val_seconds = time.monotonic() - tick
        record = dict(
            epoch=epoch,
            step=step,
            total_steps=total,
            sampled_positive_slices=positive_seen,
            sampled_negative_slices=seen - positive_seen,
            train_loss=float(sums[0] / seen),
            train_dice_loss=float(sums[1] / seen),
            train_bce_loss=float(sums[2] / seen),
            train_seconds=train_seconds,
            seconds_per_iteration=train_seconds / steps_per_pass,
            validation_seconds=val_seconds,
            encoder_lr=lr[0],
            decoder_lr=lr[1],
            **metrics,
            **resources(),
        )
        improved = metrics["positive_slice_dice_05"] > best
        best = max(best, metrics["positive_slice_dice_05"])
        checkpoint = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            global_step=step,
            best=best,
            metrics=record,
            config=provenance,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            loader_rng=generator.get_state(),
            mixed_sampler_state=sampler.state_dict() if sampler else None,
        )
        torch.save(checkpoint, out / "checkpoint_pending.pt")
        (out / "checkpoint_pending.pt").replace(out / "last.pt")
        (out / "predictions_pending.npy").replace(
            out / "validation_predictions_final.npy"
        )
        np.save(out / "validation_counts_final.npy", counts)
        write_json(
            out / "validation_predictions_final.json",
            dict(
                **record,
                dtype="float16",
                row_order="validation_manifest.csv",
                checkpoint="last.pt",
                predictions="validation_predictions_final.npy",
            ),
        )
        if improved:
            shutil.copy2(out / "last.pt", out / "best.pt")
            shutil.copy2(
                out / "validation_predictions_final.npy",
                out / "validation_predictions_best.npy",
            )
            shutil.copy2(
                out / "validation_counts_final.npy", out / "validation_counts_best.npy"
            )
            write_json(
                out / "validation_predictions_best.json",
                dict(
                    **record,
                    dtype="float16",
                    row_order="validation_manifest.csv",
                    checkpoint="best.pt",
                    predictions="validation_predictions_best.npy",
                ),
            )
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        write_json(
            out / "status.json",
            dict(
                status="complete" if epoch == args.passes else "between_passes",
                **record,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    assert step == total


if __name__ == "__main__":
    main()
