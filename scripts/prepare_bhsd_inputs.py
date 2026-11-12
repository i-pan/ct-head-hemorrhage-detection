"""Generate prepared BHSD PNGs/manifest from released paired NIfTI volumes.

Writes a new output directory only. Use prepare_bhsd_segmentation.py afterward.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from prepare_bhsd_segmentation import reorient_nifti_data


def prepare(source, output):
    import nibabel as nib

    if output.exists():
        raise FileExistsError(output)
    images = sorted(
        p
        for p in (source / "images").glob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    if not images:
        raise FileNotFoundError(source / "images")
    # Check every pair before creating output.
    for p in images:
        if not (source / "labels" / p.name).exists():
            raise FileNotFoundError(p.name)
    output.mkdir(parents=True)
    (output / "png_3ch").mkdir()
    for folder in ["images", "labels"]:
        (output / "label_192" / folder).mkdir(parents=True)
        for p in images:
            name = p.name if p.name.endswith(".nii.gz") else p.name + ".nii.gz"
            (output / "label_192" / folder / name).symlink_to(
                (source / folder / p.name).resolve()
            )
    rows = []
    for p in images:
        uid = p.name.removesuffix(".nii.gz")
        parts = uid.split("_")
        if len(parts) != 4 or parts[0] != "ID" or parts[2] != "ID":
            raise ValueError(f"Unexpected patient/study filename: {uid}")
        image = nib.load(output / "label_192/images" / f"{uid}.nii.gz")
        label = nib.load(output / "label_192/labels" / f"{uid}.nii.gz")
        if image.shape != label.shape or not np.allclose(image.affine, label.affine):
            raise ValueError(f"Image/mask geometry differs: {uid}")
        hu = reorient_nifti_data(image.get_fdata(dtype=np.float32))
        mask = reorient_nifti_data(label.get_fdata()).astype(np.uint8)
        if hu.shape[:2] != (512, 512) or not set(np.unique(mask)).issubset(range(6)):
            raise ValueError(uid)
        names = [f"{uid}_{z:03d}_image.png" for z in range(hu.shape[2])]
        for z, name in enumerate(names):
            channels = [
                np.clip((hu[..., z] - (c - w / 2)) / w, 0, 1)
                for c, w in [(40, 80), (80, 200), (600, 2800)]
            ]
            assert cv2.imwrite(
                str(output / "png_3ch" / name),
                (np.stack(channels, -1) * 255).astype(np.uint8),
            )
            labelname = f"{uid}_{z:03d}_label.png" if mask[..., z].any() else "empty"
            if labelname != "empty":
                assert cv2.imwrite(str(output / "png_3ch" / labelname), mask[..., z])
            rows.append(
                dict(
                    PatientID="_".join(parts[:2]),
                    SeriesID="_".join(parts[2:]),
                    image=name,
                    label=labelname,
                    image_2dc=",".join(
                        [names[max(0, z - 1)], name, names[min(len(names) - 1, z + 1)]]
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(
        output / "train_all_slices_png_3ch_kfold.csv", index=False
    )
    (output / "COMPLETE").write_text(f"{len(rows)} slices\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.output)
