"""Audit teacher/patient overlap and report fixed predictions without refitting."""

import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, confusion_matrix


def metrics(d):
    y = d.target.to_numpy()
    p = d.probability.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return dict(
        series=len(d),
        patients=d.patient_id.nunique(),
        positives=int(y.sum()),
        auc=roc_auc_score(y, p) if len(set(y)) == 2 else None,
        brier=brier_score_loss(y, p),
        logloss=log_loss(y, p, labels=[0, 1]),
        sensitivity=tp / (tp + fn) if tp + fn else None,
        specificity=tn / (tn + fp) if tn + fp else None,
        tp=int(tp),
        fn=int(fn),
        tn=int(tn),
        fp=int(fp),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--teacher-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    d = pd.read_csv(a.predictions)
    teachers = set(pd.read_csv(a.teacher_manifest).patient_id.astype(str))
    overlap = d.patient_id.astype(str).isin(teachers)
    result = dict(
        protocol="No refitting; original T=.25 predictions, fixed probability .5; exclude all teacher patients, independent of outcomes.",
        original=metrics(d),
        teacher_disjoint=metrics(d[~overlap]),
        excluded_series=int(overlap.sum()),
        excluded_patients=int(d.loc[overlap, "patient_id"].nunique()),
        limitation="Post-hoc cohort correction, not a fresh untouched holdout. Validation-based selection previously included one teacher patient.",
    )
    a.output.mkdir(parents=True)
    d[~overlap].to_csv(a.output / "teacher_disjoint_predictions.csv", index=False)
    d[overlap].to_csv(a.output / "excluded_predictions.csv", index=False)
    (a.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
