"""Evaluate a local prior-recipe package with its fixed classifier/BiGRU blends."""

import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--teacher-manifest", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from skp.configs.rsna_ich_maxvit_tiny_fixedval_9ch import cfg
    from skp.datasets.rsna_ich_2p5d import Dataset

    ds = Dataset(cfg, a.split)
    teachers = set(pd.read_csv(a.teacher_manifest).patient_id.astype(str))
    assert not teachers & set(ds.df.patient_id.astype(str)), "Teacher-patient overlap"
    spec = importlib.util.spec_from_file_location(
        "legacy_inference", a.model_dir / "inference.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model = m.HemorrhageSliceModel(local_dir=a.model_dir)
    a.output.mkdir(parents=True)
    series = []
    slice_rows = []
    for uid, ids in ds.df.groupby("series_uid", sort=False).indices.items():
        slabs = np.stack([ds[int(i)]["x"].numpy() for i in ids])
        out = model.predict_series(slabs, return_segmentation=False)
        target = ds.df.iloc[ids][cfg.label_columns].to_numpy()
        record = {"series_uid": uid, "patient_id": ds.df.iloc[ids[0]].patient_id}
        for j, label in enumerate(cfg.label_columns):
            record[label + "_target"] = float(target[:, j].max())
            record[label + "_probability"] = float(out["series_classification"][j])
        series.append(record)
        for k, i in enumerate(ids):
            r = {"slice_path": ds.df.iloc[i].slice_path, "series_uid": uid}
            for j, label in enumerate(cfg.label_columns):
                r[label + "_target"] = float(target[k, j])
                r[label + "_probability"] = float(out["slice_classification"][k, j])
            slice_rows.append(r)
        print(f"{len(series)} series", flush=True)
    results = {}
    for level, rows in [("series", series), ("slice", slice_rows)]:
        d = pd.DataFrame(rows)
        d.to_csv(a.output / f"{level}_predictions.csv", index=False)
        results[level] = {
            c: roc_auc_score(d[c + "_target"], d[c + "_probability"])
            if d[c + "_target"].nunique() == 2
            else None
            for c in cfg.label_columns
        }
    (a.output / "results.json").write_text(
        json.dumps(
            {
                "split": a.split,
                "blend": "fixed prior recipe; no tuning",
                "auc": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
