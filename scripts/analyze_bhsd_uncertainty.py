"""CPU-only OOF uncertainty audit, restricted to known-positive slices."""

import json
import numpy as np
import pandas as pd
from analyze_bhsd_teacher_ensemble import ROOT, ROOTS

OUT = ROOT / "diagnostics/bhsd_uncertainty_20261110"
BINS = 2048
NAMES = ["std", "entropy", "mutual_information"]


def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def summarize(hist):
    rows = []
    code = np.arange(8)
    gt = (code & 1) != 0
    for mi, name in enumerate(NAMES):
        for group in [
            "all",
            "foreground",
            "background",
            "small_positive_slices",
            "predicted_foreground",
        ]:
            for threshold, bit in [(0.5, 2), (0.6, 4)]:
                pred = (code & bit) != 0
                selected = np.ones(8, bool)
                if group == "foreground":
                    selected = gt
                if group == "background":
                    selected = ~gt
                if group == "predicted_foreground":
                    selected = pred
                h = hist[mi, 1] if group == "small_positive_slices" else hist[mi].sum(0)
                h = h * selected[:, None]
                count = h.sum(0)
                errors = h[gt != pred].sum(0)
                foreground = h[gt].sum(0)
                correct_foreground = h[gt & pred].sum(0)
                small_foreground = (hist[mi, 1] * selected[:, None])[gt].sum(0)
                total = count.sum()
                if not total:
                    continue
                for retention in [1.0, 0.95, 0.9, 0.75, 0.5]:
                    budget = retention * total
                    weights = np.clip(
                        (budget - (np.cumsum(count) - count)) / np.maximum(count, 1),
                        0,
                        1,
                    )
                    kept_errors = (weights * errors).sum()
                    rows.append(
                        dict(
                            measure=name,
                            group=group,
                            threshold=threshold,
                            retention=retention,
                            pixels=int(total),
                            error_rate=kept_errors / budget,
                            error_removed_fraction=1
                            - kept_errors / max(errors.sum(), 1),
                            hemorrhage_retained_fraction=(weights * foreground).sum()
                            / max(foreground.sum(), 1),
                            correct_foreground_retained_fraction=(
                                weights * correct_foreground
                            ).sum()
                            / max(correct_foreground.sum(), 1),
                            small_hemorrhage_retained_fraction=(
                                weights * small_foreground
                            ).sum()
                            / max(small_foreground.sum(), 1),
                        )
                    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_hist = np.zeros((3, 2, 8, BINS), dtype=np.int64)
    records = []
    sources = []
    slices = small_slices = 0
    for fold in range(5):
        arrays = []
        df = None
        for name in ["convnext", "maxvit", "coatnet"]:
            run = ROOTS[name] / f"fold{fold}"
            manifest = pd.read_csv(run / "validation_manifest.csv")
            train = pd.read_csv(run / "train_manifest.csv")
            assert set(train.patient_id).isdisjoint(manifest.patient_id)
            meta = json.loads((run / "validation_predictions_final.json").read_text())
            assert meta["epoch"] == 25 and meta["checkpoint"] == "last.pt"
            assert meta["step"] == meta["total_steps"]
            if df is None:
                df = manifest
            else:
                pd.testing.assert_frame_equal(df, manifest)
            path = run / "validation_predictions_final.npy"
            arrays.append(np.load(path, mmap_mode="r"))
            sources.append(
                dict(
                    path=str(path),
                    bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns,
                )
            )
        hist = np.zeros_like(all_hist)
        masks = {
            uid: np.load(rows.iloc[0].mask_volume_path, mmap_mode="r")
            for uid, rows in df.groupby("series_uid")
        }
        for i, row in enumerate(df.itertuples()):
            if not row.has_mask:
                continue
            y = masks[row.series_uid][..., int(row.slice_index)] > 0
            assert y.any()
            small = int(y.sum() <= 262)  # <=0.1% of 512x512 image area
            slices += 1
            small_slices += small
            p = np.stack([a[i].astype(np.float32) for a in arrays])
            assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
            mean = p.mean(0)
            ent = entropy(mean)
            measures = [p.std(0) * 2, ent, np.maximum(ent - entropy(p).mean(0), 0)]
            code = y.astype(np.int64) + 2 * (mean >= 0.5) + 4 * (mean >= 0.6)
            for j, value in enumerate(measures):
                bins = np.minimum((value * BINS).astype(np.int64), BINS - 1)
                hist[j, small] += np.bincount(
                    (code * BINS + bins).ravel(), minlength=8 * BINS
                ).reshape(8, BINS)
        all_hist += hist
        np.save(OUT / f"fold{fold}_hist.npy", hist)
        records.extend([dict(fold=fold, **r) for r in summarize(hist)])
        print(
            f"Completed fold {fold}; positive slices {slices}; small {small_slices}",
            flush=True,
        )
    pd.DataFrame(records).to_csv(OUT / "fold_metrics.csv", index=False)
    result = pd.DataFrame(summarize(all_hist))
    result.to_csv(OUT / "pooled_metrics.csv", index=False)
    np.save(OUT / "pooled_hist.npy", all_hist)
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                positive_slices=slices,
                small_positive_slices=small_slices,
                bins=BINS,
                method="Final held-out teacher per family, three teachers per case. Known-positive slices only. Small means <=262 ground-truth positive pixels on a slice, not volume. Population std normalized by 0.5, entropy/MI in bits. Low-uncertainty retention ranked within each reported stratum, fractional ties within histogram bins. Descriptive OOF audit, not independent selection validation. No GPU use or inference. Ensemble uses original float16 teacher probabilities averaged in float32.",
            ),
            indent=2,
        )
    )
    print(
        result[(result.retention == 0.9) & (result.threshold == 0.5)].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
