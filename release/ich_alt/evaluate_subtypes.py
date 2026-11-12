"""Evaluate six outputs from saved aligned final-test logits; no model inference."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from skp.ich_alt import LABELS, sha256


def evaluate(evidence, teachers):
    logits = []
    provenance = {}
    for name in ("convnext_tiny", "maxvit", "coatnet0"):
        paths = [evidence / f"{name}_rank{rank}.npz" for rank in (0, 1)]
        with np.load(paths[0]) as s0, np.load(paths[1]) as s1:
            order = np.argsort(np.concatenate([s0["indices"], s1["indices"]]))
            a = {
                k: np.concatenate([s0[k], s1[k]])[order]
                for k in ("indices", "logits", "targets", "groups")
            }
        m = pd.read_csv(evidence / f"{name}_manifest.csv")
        np.testing.assert_array_equal(a["indices"], np.arange(len(m)))
        np.testing.assert_array_equal(a["targets"], m[list(LABELS)].to_numpy())
        assert a["logits"].shape == (len(m), 6) and np.isfinite(a["logits"]).all()
        if logits:
            np.testing.assert_array_equal(m.slice_path, manifest.slice_path)
            for key in ("targets", "groups"):
                np.testing.assert_array_equal(a[key], ref[key])
        else:
            ref, manifest = a, m
        logits.append(a["logits"].astype(np.float64))
        meta = json.loads((evidence / f"{name}_provenance.json").read_text())
        assert meta["epoch"] == 49 and meta["prediction_weights"] == "raw"
        provenance[name] = dict(
            checkpoint_sha256=meta["checkpoint_sha256"],
            archive_sha256=[sha256(p) for p in paths],
            manifest_sha256=sha256(evidence / f"{name}_manifest.csv"),
        )
    z = np.mean(logits, axis=0)
    groups = ref["groups"]
    series_z = []
    series_y = []
    series_patient = []
    for g in np.unique(groups):
        ix = groups == g
        assert (
            manifest.loc[ix, "series_uid"].nunique() == 1
            and manifest.loc[ix, "patient_id"].nunique() == 1
        )
        series_z.append(0.25 * (logsumexp(z[ix] / 0.25, axis=0) - np.log(ix.sum())))
        series_y.append(ref["targets"][ix].max(axis=0))
        series_patient.append(str(manifest.loc[ix, "patient_id"].iloc[0]))
    series_z = np.asarray(series_z)
    series_y = np.asarray(series_y)
    teacher_ids = set(pd.read_csv(teachers).patient_id.astype(str))
    result = dict(
        protocol="Three final raw epoch50 models; equal mean slice logits, sigmoid for slices; normalized LSE T=.25 over slices then sigmoid for series. No fitting or subtype-specific tuning. Retrospective already-consumed test set.",
        labels=list(LABELS),
        source_predictions=provenance,
        teacher_manifest_sha256=sha256(teachers),
        cohorts={},
    )
    for cohort in ("original", "teacher_disjoint"):
        si = (
            np.ones(len(z), bool)
            if cohort == "original"
            else ~manifest.patient_id.astype(str).isin(teacher_ids).to_numpy()
        )
        vi = (
            np.ones(len(series_z), bool)
            if cohort == "original"
            else np.array([p not in teacher_ids for p in series_patient])
        )
        rows = {}
        for j, label in enumerate(LABELS):
            row = {}
            for level, y, p in [
                ("slice", ref["targets"][si, j], expit(z[si, j])),
                ("series", series_y[vi, j], expit(series_z[vi, j])),
            ]:
                row[level] = dict(
                    positives=int(y.sum()),
                    auc=float(roc_auc_score(y, p)),
                    brier=float(brier_score_loss(y, p)),
                    logloss=float(log_loss(y, p, labels=[0, 1])),
                )
            rows[label] = row
        result["cohorts"][cohort] = dict(
            slices=int(si.sum()),
            series=int(vi.sum()),
            patients=len(set(np.asarray(series_patient)[vi])),
            metrics=rows,
            macro_auc={
                level: float(np.mean([r[level]["auc"] for r in rows.values()]))
                for level in ("slice", "series")
            },
        )
    previous = json.loads((evidence / "results.json").read_text())
    for key in ("auc", "brier", "logloss"):
        np.testing.assert_allclose(
            result["cohorts"]["original"]["metrics"]["any"]["series"][key],
            previous[key],
            rtol=1e-12,
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--teacher-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    r = evaluate(a.evidence, a.teacher_manifest)
    a.output.write_text(json.dumps(r, indent=2))
    print(json.dumps(r["cohorts"], indent=2))
