"""Locked final-checkpoint three-model test evaluation; two GPU shards/model."""

import os, sys, json, subprocess, hashlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
import argparse

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument(
    "--runs", type=Path, required=True, help="JSON name -> student run directory"
)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--teacher-manifest", type=Path, required=True)
ap.add_argument("--rank", type=int, choices=[0, 1])
ap.add_argument("--model")
args = ap.parse_args()
OUT = args.output.resolve()
RUNS = json.loads(args.runs.read_text())
assert set(RUNS) == {"convnext_tiny", "maxvit", "coatnet0"}, (
    "Expected exactly three selected backbone names"
)


def worker(name, rank):
    import torch
    from torch.utils.data import DataLoader, Subset
    from analyze_ich_backbone_ensemble import load_model
    from importlib import import_module

    torch.set_num_threads(4)
    root = Path(RUNS[name])
    meta = json.loads((root / "validation_predictions_final.json").read_text())
    cfgname = meta["config"]
    meta = json.loads((root / "validation_predictions_final.json").read_text())
    ckpt = Path(meta["checkpoint"])
    assert meta["epoch"] == 49 and meta["global_step"] > 0
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == meta["checkpoint_sha256"]
    model, cfg = load_model(cfgname, ckpt, f"cuda:{rank}")
    cfg.test_transforms = cfg.val_transforms
    ds = import_module("skp.datasets." + cfg.dataset).Dataset(cfg, "test")
    teacher_patients = set(pd.read_csv(args.teacher_manifest).patient_id.astype(str))
    assert not set(ds.df.patient_id.astype(str)) & teacher_patients, (
        "Teacher/test patient overlap; use new splits or report existing predictions with audit_ich_holdout.py"
    )
    assert set(ds.df.split) == {"test"} and not ds.df.excluded_bhsd.any()
    splits = pd.read_csv(cfg.annotations_file)
    assert not set(ds.df.patient_id) & set(
        splits.loc[splits.split.isin(["train", "val"]), "patient_id"]
    )
    ids = np.arange(rank, len(ds), 2)
    if rank == 0:
        ds.df.to_csv(OUT / f"{name}_manifest.csv", index=False)
        (OUT / f"{name}_provenance.json").write_text(json.dumps(meta, indent=2))
    loader = DataLoader(Subset(ds, ids), batch_size=64, num_workers=2, pin_memory=False)
    logits = []
    ys = []
    groups = []
    with torch.inference_mode():
        for i, b in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model({"x": b["x"].to(f"cuda:{rank}")})
            logits.append(o["logits"].float().cpu().numpy())
            ys.append(b["y"].numpy())
            groups.append(b["group_index"].numpy())
            if i % 50 == 0:
                print(name, rank, i, len(loader), flush=True)
    np.savez_compressed(
        OUT / f"{name}_rank{rank}.npz",
        indices=ids,
        logits=np.concatenate(logits),
        targets=np.concatenate(ys),
        groups=np.concatenate(groups),
    )


if args.rank is not None:
    worker(args.model, args.rank)
    sys.exit()
if OUT.exists():
    raise FileExistsError(OUT)
OUT.mkdir(parents=True)
(OUT / "protocol.json").write_text(
    json.dumps(
        dict(
            models=list(RUNS),
            checkpoint="final epoch50",
            split="test excluding BHSD",
            ensemble="equal mean of slice any logits then T*log(mean(exp(z/T))) across slices",
            temperature=0.25,
            calibration="none",
            decision_threshold=0.5,
            test_tuning=False,
        ),
        indent=2,
    )
)
for name in RUNS:
    ps = [
        subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--runs",
                str(args.runs),
                "--output",
                str(OUT),
                "--teacher-manifest",
                str(args.teacher_manifest),
                "--model",
                name,
                "--rank",
                str(r),
            ],
            env=dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
        )
        for r in [0, 1]
    ]
    codes = [p.wait() for p in ps]
    assert codes == [0, 0], codes
from scipy.special import logsumexp, expit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, confusion_matrix

arrays = []
ref = None
refm = None
for name in RUNS:
    shards = [np.load(OUT / f"{name}_rank{r}.npz") for r in [0, 1]]
    order = np.argsort(np.concatenate([s["indices"] for s in shards]))
    a = {
        k: np.concatenate([s[k] for s in shards])[order]
        for k in ["indices", "logits", "targets", "groups"]
    }
    assert np.array_equal(a["indices"], np.arange(len(a["indices"])))
    m = pd.read_csv(OUT / f"{name}_manifest.csv")
    if ref is not None:
        for k in ["targets", "groups"]:
            np.testing.assert_array_equal(ref[k], a[k])
        np.testing.assert_array_equal(refm.slice_path, m.slice_path)
    else:
        ref = a
        refm = m
    assert np.isfinite(a["logits"]).all()
    arrays.append(a["logits"][:, -1].astype(float))
z = np.mean(arrays, axis=0)
rows = []
for g in np.unique(ref["groups"]):
    ix = ref["groups"] == g
    logit = 0.25 * (logsumexp(z[ix] / 0.25) - np.log(ix.sum()))
    r = refm.loc[ix].iloc[0]
    rows.append(
        dict(
            series_uid=r.series_uid,
            patient_id=r.patient_id,
            target=int(ref["targets"][ix, -1].max()),
            n_slices=int(ix.sum()),
            logit=logit,
            probability=expit(logit),
        )
    )
d = pd.DataFrame(rows)
d.to_csv(OUT / "test_series_predictions.csv", index=False)
y = d.target.to_numpy()
p = d.probability.to_numpy()
tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel()
bins = np.minimum((p * 10).astype(int), 9)
r = dict(
    status="complete",
    series=len(d),
    patients=d.patient_id.nunique(),
    slices=len(z),
    positives=int(y.sum()),
    auc=roc_auc_score(y, p),
    brier=brier_score_loss(y, p),
    logloss=log_loss(y, p),
    ece=sum(
        np.mean(bins == i) * abs(y[bins == i].mean() - p[bins == i].mean())
        for i in range(10)
        if (bins == i).any()
    ),
    threshold=0.5,
    sensitivity=tp / (tp + fn),
    specificity=tn / (tn + fp),
    tp=int(tp),
    fp=int(fp),
    tn=int(tn),
    fn=int(fn),
)
(OUT / "results.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r), flush=True)
