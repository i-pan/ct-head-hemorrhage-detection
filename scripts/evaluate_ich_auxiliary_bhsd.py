"""Exploratory, overlap-affected BHSD decoder Dice and spatial pointing evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from scipy.ndimage import distance_transform_edt
from analyze_ich_backbone_ensemble import load_model
from skp.toolbox.bhsd_convnext import Dataset

THRESHOLDS = np.arange(1, 10) / 10


def pointing(logit_map, target, spacing):
    up = F.interpolate(
        logit_map.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
    )
    flat = up[:, 0].flatten(1).argmax(1).cpu().numpy()
    coarse = logit_map[:, 0].flatten(1).argmax(1)
    coarse_mask = F.adaptive_max_pool2d(target.float(), logit_map.shape[-2:])[
        :, 0
    ].flatten(1)
    coarse_hit = coarse_mask.gather(1, coarse[:, None])[:, 0].bool().cpu().numpy()
    y = target[:, 0].bool().cpu().numpy()
    rows = []
    for i, pixel in enumerate(flat):
        row, col = np.unravel_index(pixel, y.shape[1:])
        positive = bool(y[i].any())
        distance = (
            float(distance_transform_edt(~y[i], sampling=spacing[i])[row, col])
            if positive
            else None
        )
        rows.append(
            dict(
                point_row=int(row),
                point_col=int(col),
                positive=positive,
                point_hit=bool(y[i, row, col]) if positive else None,
                coarse_cell_hit=bool(coarse_hit[i]) if positive else None,
                distance_mm=distance,
                within_5mm=distance <= 5 if positive else None,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    checkpoint = args.run_dir / "fold0/checkpoints/last.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert state["epoch"] == 49 and state["global_step"] > 0
    metadata = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.file_digest(
            checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        epoch=state["epoch"],
        step=state["global_step"],
        config=args.config,
        caveat="Exploratory BHSD evaluation with acknowledged training/teacher overlap. Not independent generalization or OOF performance. Final checkpoint only.",
        pointing="Peak of bilinear-upsampled any-hemorrhage logit map at 512x512 (align_corners=False). Argmax ties take first raster pixel. Primary hit requires annotated positive pixel. Coarse-cell overlap and distance within5mm are additional, more lenient metrics.",
        thresholds=THRESHOLDS.tolist(),
    )
    del state
    dest = args.run_dir / "bhsd_final"
    dest.mkdir(exist_ok=True)
    if (dest / "completion.json").exists():
        old = json.loads((dest / "completion.json").read_text())
        assert old["checkpoint_sha256"] == metadata["checkpoint_sha256"]
        print("Verified existing BHSD evaluation", flush=True)
        return
    model, cfg = load_model(args.config, checkpoint, "cuda:0")
    datasets = [Dataset(f, False) for f in range(5)]
    manifest = pd.concat([d.df for d in datasets], ignore_index=True)
    assert len(manifest) > 0
    manifest.to_csv(dest / "manifest.csv", index=False)
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )
    probability = np.lib.format.open_memmap(
        dest / "decoder_probabilities.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(manifest), 512, 512),
    )
    maps = np.lib.format.open_memmap(
        dest / "any_logit_maps.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(manifest), 32, 32),
    )
    counts = np.zeros((len(manifest), len(THRESHOLDS), 3), np.int64)
    points = []
    logits = []
    start = 0
    with torch.inference_mode():
        for x, y, _ in loader:
            end = start + len(x)
            target = y.cuda(non_blocking=True) > 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    {"x": x.cuda(non_blocking=True)},
                    return_features=True,
                    return_segmentation=True,
                )
            p = out["segmentation_logits"].float().sigmoid()
            logit_map = out["logit_map"][
                :, cfg.label_columns.index("any") : cfg.label_columns.index("any") + 1
            ].float()
            assert torch.isfinite(p).all() and torch.isfinite(logit_map).all()
            probability[start:end] = p[:, 0].cpu().numpy().astype(np.float16)
            maps[start:end] = logit_map[:, 0].cpu().numpy().astype(np.float16)
            logits.append(out["logits"].float().cpu().numpy())
            for j, t in enumerate(THRESHOLDS):
                pred = p >= float(t)
                counts[start:end, j] = (
                    torch.stack(
                        (
                            (pred & target).sum((1, 2, 3)),
                            pred.sum((1, 2, 3)),
                            target.sum((1, 2, 3)),
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
            spacing = manifest.iloc[start:end][
                ["row_spacing_mm", "col_spacing_mm"]
            ].to_numpy()
            points.extend(pointing(logit_map, target, spacing))
            start = end
            if start % 512 == 0:
                print(f"{start}/{len(manifest)} BHSD slices", flush=True)
    assert start == len(manifest)
    probability.flush()
    maps.flush()
    np.save(dest / "counts.npy", counts)
    np.save(dest / "classification_logits.npy", np.concatenate(logits))
    pts = pd.concat(
        [
            manifest[["series_uid", "slice_index", "fold"]].reset_index(drop=True),
            pd.DataFrame(points),
        ],
        axis=1,
    )
    pts["mask_pixels"] = counts[:, 0, 2]
    pts["mask_area_mm2"] = (
        pts.mask_pixels * manifest.row_spacing_mm * manifest.col_spacing_mm
    )
    pts.to_csv(dest / "pointing.csv", index=False)
    positive = counts[:, 0, 2] > 0
    assert np.array_equal(positive, manifest.has_mask.to_numpy())
    dice = 2 * counts[:, :, 0] / np.maximum(counts[:, :, 1] + counts[:, :, 2], 1)
    volumes = []
    for uid, indices in manifest.groupby("series_uid").indices.items():
        raw = counts[indices].sum(0)
        gated = counts[indices].copy()
        gated[~positive[indices], :, 1] = 0
        gated = gated.sum(0)
        for j, t in enumerate(THRESHOLDS):
            volumes.append(
                dict(
                    series_uid=uid,
                    threshold=t,
                    volume_dice=2 * raw[j, 0] / max(raw[j, 1] + raw[j, 2], 1),
                    gated_volume_dice=2
                    * gated[j, 0]
                    / max(gated[j, 1] + gated[j, 2], 1),
                )
            )
    volumes = pd.DataFrame(volumes)
    volumes.to_csv(dest / "volume_metrics.csv", index=False)
    summary = []
    for j, t in enumerate(THRESHOLDS):
        v = volumes[np.isclose(volumes.threshold, t)]
        summary.append(
            dict(
                threshold=t,
                positive_slice_dice=dice[positive, j].mean(),
                volume_dice=v.volume_dice.mean(),
                gated_volume_dice=v.gated_volume_dice.mean(),
                negative_slice_fp_fraction=(counts[~positive, j, 1] > 0).mean(),
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(dest / "segmentation_thresholds.csv", index=False)
    point_summary = []
    for name, selected in [
        ("all_positive", pts[pts.positive]),
        ("small_positive_area_le10mm2", pts[pts.positive & (pts.mask_area_mm2 <= 10)]),
    ]:
        point_summary.append(
            dict(
                group=name,
                n=len(selected),
                point_hit_rate=selected.point_hit.mean(),
                coarse_cell_hit_rate=selected.coarse_cell_hit.mean(),
                within_5mm_rate=selected.within_5mm.mean(),
                median_distance_mm=selected.distance_mm.median(),
            )
        )
    pd.DataFrame(point_summary).to_csv(dest / "pointing_summary.csv", index=False)
    metadata.update(
        status="complete",
        slices=len(manifest),
        positive_slices=int(positive.sum()),
        dice_at_05=summary.iloc[4].to_dict(),
        best_positive_slice_dice=summary.loc[
            summary.positive_slice_dice.idxmax()
        ].to_dict(),
        best_volume_dice=summary.loc[summary.volume_dice.idxmax()].to_dict(),
        pointing=point_summary,
        pointing_method=metadata["pointing"],
    )
    temp = dest / "completion.tmp.json"
    temp.write_text(json.dumps(metadata, indent=2))
    temp.replace(dest / "completion.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
