"""OOF teacher-defined ROI coverage, small components and HU unions; CPU only."""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/bhsd_teacher_roi_refined_20261111"
TH = np.array([0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09], np.float32)
NB = len(TH) + 1


def tails(hist):
    return np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1][..., 1:]


def hist(values):
    return np.bincount(np.searchsorted(TH, values.ravel(), side="right"), minlength=NB)


def grouped(groups, values, n):
    bins = np.searchsorted(TH, values, side="right")
    return np.bincount(groups * NB + bins, minlength=n * NB).reshape(n, NB)


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    arrays = {}
    for fold in range(5):
        base = ROOT / "diagnostics/bhsd_teacher_ensemble_20261110" / f"fold{fold}"
        frame = pd.read_csv(base / "validation_manifest.csv")
        frame["prediction_index"] = np.arange(len(frame))
        frame["oof_fold"] = fold
        frames.append(frame)
        arrays[fold] = np.load(base / "three_probabilities.npy", mmap_mode="r")
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 6404 and df.groupby("patient_id").oof_fold.nunique().max() == 1
    series_rows = []
    slice_rows = []
    component_rows = []
    for number, (uid, rows) in enumerate(df.groupby("series_uid"), 1):
        rows = rows.sort_values("slice_index")
        assert rows.oof_fold.nunique() == 1
        p = np.asarray(
            arrays[int(rows.iloc[0].oof_fold)][rows.prediction_index.to_numpy()],
            dtype=np.float32,
        )
        mask = (
            np.load(rows.iloc[0].mask_volume_path, mmap_mode="r").transpose(2, 0, 1) > 0
        )
        assert p.shape == mask.shape
        positive = mask.any((1, 2))
        assert np.array_equal(positive, rows.has_mask.to_numpy())
        cc, ncomp = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
        z = np.nonzero(mask)[0]
        component = cc[mask] - 1
        slice_sizes = np.bincount(z, minlength=len(p))
        comp_sizes = np.bincount(component, minlength=ncomp)
        voxel_ml = (
            float(
                rows.iloc[0].row_spacing_mm
                * rows.iloc[0].col_spacing_mm
                * rows.iloc[0].slice_spacing_mm
            )
            / 1000
        )
        hu = np.load(ROOT / "data/bhsd/hu_volumes" / f"{uid}.npy", mmap_mode="r")
        hu_roi = (hu >= 0) & (hu <= 100)
        hu_target = hu_roi[mask]
        hu_slices = np.bincount(z[hu_target], minlength=len(p))
        hu_comps = np.bincount(component[hu_target], minlength=ncomp)
        variants = [
            (
                "hu_0_100",
                0,
                0,
                hu_slices,
                hu_comps,
                int(hu_roi.sum()),
                int(hu_roi[positive].sum()),
            )
        ]
        for radius in [0, 3, 5]:
            if radius:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                )
                score = np.stack([cv2.dilate(s, kernel) for s in p])
            else:
                score = p
            all_area = tails(hist(score))
            pos_area = tails(hist(score[positive]))
            values = score[mask]
            slices = tails(grouped(z, values, len(p)))
            components = tails(grouped(component, values, ncomp))
            for j, t in enumerate(TH):
                variants.append(
                    (
                        "teacher",
                        float(t),
                        radius,
                        slices[:, j],
                        components[:, j],
                        int(all_area[j]),
                        int(pos_area[j]),
                    )
                )
            # Union preserves HU-covered tissue, adds teacher regions beyond HU band.
            area_union = tails(hist(score[~hu_roi])) + hu_roi.sum()
            pos_union = (
                tails(hist(score[positive][~hu_roi[positive]])) + hu_roi[positive].sum()
            )
            slice_union = (
                tails(grouped(z[~hu_target], values[~hu_target], len(p)))
                + hu_slices[:, None]
            )
            comp_union = (
                tails(grouped(component[~hu_target], values[~hu_target], ncomp))
                + hu_comps[:, None]
            )
            for j, t in enumerate(TH):
                variants.append(
                    (
                        "teacher_union_hu_0_100",
                        float(t),
                        radius,
                        slice_union[:, j],
                        comp_union[:, j],
                        int(area_union[j]),
                        int(pos_union[j]),
                    )
                )
        for method, t, radius, covered_slice, covered_comp, area, posarea in variants:
            common = dict(
                method=method,
                threshold=round(t, 3),
                radius_pixels=radius,
                series_uid=uid,
                patient_id=rows.iloc[0].patient_id,
                fold=int(rows.iloc[0].oof_fold),
            )
            series_rows.append(
                dict(
                    **common,
                    positive_pixels=int(mask.sum()),
                    covered_pixels=int(covered_slice.sum()),
                    all_roi_pixels=area,
                    all_pixels=mask.size,
                    positive_roi_pixels=posarea,
                    positive_image_pixels=int(positive.sum() * 512 * 512),
                    coverage=covered_slice.sum() / mask.sum(),
                )
            )
            for i in np.flatnonzero(positive):
                slice_rows.append(
                    dict(
                        **common,
                        slice_index=int(i),
                        pixels=int(slice_sizes[i]),
                        covered_pixels=int(covered_slice[i]),
                        coverage=covered_slice[i] / slice_sizes[i],
                    )
                )
            for i, n in enumerate(comp_sizes):
                component_rows.append(
                    dict(
                        **common,
                        component=i + 1,
                        pixels=int(n),
                        volume_ml=float(n * voxel_ml),
                        covered_pixels=int(covered_comp[i]),
                        coverage=covered_comp[i] / n,
                    )
                )
        if number % 16 == 0:
            print(f"{number}/192 series", flush=True)
    series = pd.DataFrame(series_rows)
    slices = pd.DataFrame(slice_rows)
    components = pd.DataFrame(component_rows)
    series.to_csv(OUT / "series.csv", index=False)
    slices.to_csv(OUT / "positive_slices.csv", index=False)
    components.to_csv(OUT / "components.csv", index=False)
    keys = ["method", "threshold", "radius_pixels"]
    summary = []
    strata = []
    for key, g in series.groupby(keys):
        sl = slices[(slices[keys] == key).all(axis=1)]
        co = components[(components[keys] == key).all(axis=1)]
        small = co[co.volume_ml <= 0.1]
        pos_roi = g.positive_roi_pixels.sum()
        summary.append(
            dict(
                zip(keys, key),
                pooled_hemorrhage_coverage=g.covered_pixels.sum()
                / g.positive_pixels.sum(),
                positive_roi_fraction=pos_roi / g.positive_image_pixels.sum(),
                all_roi_fraction=g.all_roi_pixels.sum() / g.all_pixels.sum(),
                roi_hemorrhage_fraction=g.covered_pixels.sum() / pos_roi,
                series_macro_coverage=g.coverage.mean(),
                positive_slice_macro_coverage=sl.coverage.mean(),
                missed_positive_slices=int((sl.coverage == 0).sum()),
                small_component_macro_coverage=small.coverage.mean(),
                small_component_misses=int((small.coverage == 0).sum()),
                small_components=len(small),
            )
        )
        for lower, upper in [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]:
            c = co[(co.volume_ml > lower) & (co.volume_ml <= upper)]
            strata.append(
                dict(
                    zip(keys, key),
                    lower_ml=lower,
                    upper_ml=upper,
                    n=len(c),
                    macro_coverage=c.coverage.mean(),
                    p10_coverage=c.coverage.quantile(0.1),
                    miss_fraction=(c.coverage == 0).mean(),
                    below_half_fraction=(c.coverage < 0.5).mean(),
                )
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "component_size_strata.csv", index=False)
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                method="Held-out three-family mean probabilities, float16 saved OOF ensemble. Teacher ROI score >= threshold; elliptical in-plane dilation radii 0/3/5 pixels. HU union inclusive [0,100]. Whole image, no brain mask. Small components <=0.1mL, 26-connected 3D binary masks, not adjudicated lesions. Pooled pixel and macro metrics separately. Descriptive threshold sweep; not independent selection evaluation; three OOF teachers rather than 15 production teachers.",
                thresholds=TH.tolist(),
                ensemble_provenance=str(
                    ROOT / "diagnostics/bhsd_teacher_ensemble_20261110/provenance.json"
                ),
            ),
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Reproduce the initial broad threshold sweep instead of the refined sweep.",
    )
    args = parser.parse_args()
    if args.initial:
        TH = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], np.float32)
        OUT = ROOT / "diagnostics/bhsd_teacher_roi_20261111"
    main()
