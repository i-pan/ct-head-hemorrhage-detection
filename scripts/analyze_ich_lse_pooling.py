"""Patient-grouped cross-validation of frozen-slice logit pooling and calibration."""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.path.insert(0, "scripts")
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--runs", type=Path, required=True, help="JSON name -> student run directory"
)
parser.add_argument("--splits", type=Path, required=True)
parser.add_argument("--teacher-manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.mkdir(parents=True)
runs = json.loads(args.runs.read_text())
assert set(runs) == {"convnext_tiny", "maxvit", "coatnet0"}, (
    "Expected exactly three selected backbone names"
)


def load_final(run):
    root = Path(run)
    p = root / "validation_predictions_final.npz"
    meta = json.loads(p.with_suffix(".json").read_text())
    assert meta["role"] == "final" and meta["epoch"] == 49
    import hashlib

    assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["archive_sha256"]
    with np.load(p) as a:
        arrays = {k: a[k] for k in a.files}
    return arrays, meta


data = {}
prov = {}
for k, p in runs.items():
    data[k], prov[k] = load_final(p)
r = data["maxvit"]
g = r["groups"]
ys = r["targets"][:, -1]
for a in data.values():
    np.testing.assert_array_equal(a["targets"], r["targets"])
    np.testing.assert_array_equal(a["groups"], g)
u = np.unique(g)
assert np.array_equal(u, np.arange(len(u)))
df = pd.read_csv(args.splits)
df = df[(df["split"] == "val") & (~df["excluded_bhsd"])].copy()
df["uid"] = (
    df.patient_id.astype(str)
    + "/"
    + df.study_id.astype(str)
    + "/"
    + df.series_id.astype(str)
)
m = df[["uid", "patient_id"]].drop_duplicates().sort_values("uid")
assert len(m) == len(u)
expected = df.groupby("uid", sort=True).size()
np.testing.assert_array_equal(expected.to_numpy(), np.bincount(g))
patients = m.patient_id.to_numpy()
assert not set(map(str, patients)) & set(pd.read_csv(args.teacher_manifest).patient_id.astype(str)), "Teacher/validation patient overlap"
idx = [np.flatnonzero(g == i) for i in u]
y = np.array([ys[j].max() for j in idx], int)
n = np.array([len(j) for j in idx])
folds = list(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=88).split(
        np.zeros(len(y)), y, patients
    )
)
assert all(not set(patients[a]) & set(patients[b]) for a, b in folds)
z = {k: data[k]["logits"][:, -1].astype(float) for k in runs}


def pad(x):
    a = np.full((len(u), n.max()), -np.inf)
    for i, j in enumerate(idx):
        a[i, : len(j)] = x[j]
    return a


def pool(a, t):
    return a.max(1) if t == "max" else t * (logsumexp(a / t, axis=1) - np.log(n))


def ece(p):
    b = np.minimum((p * 10).astype(int), 9)
    return sum(
        np.mean(b == i) * abs(y[b == i].mean() - p[b == i].mean())
        for i in range(10)
        if (b == i).any()
    )


rows = []
scores = {"target": y, "patient": patients}
parameters = []


def report(name, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    neg = p[y == 0]
    thr = np.nextafter(np.sort(neg)[::-1][5], np.inf)
    rows.append(
        dict(
            method=name,
            auc=roc_auc_score(y, p),
            brier=brier_score_loss(y, p),
            logloss=log_loss(y, p),
            ece=ece(p),
            sensitivity_at_5fp=((p >= thr) & (y == 1)).sum() / y.sum(),
            threshold_descriptive=thr,
        )
    )
    scores[name] = p


def test(label, logits):
    a = pad(logits)
    for t in ["max", 0.25, 0.5, 1.0, 2.0]:
        x = pool(a, t)
        report(f"{label} | {t} | raw", expit(x))
        p = np.zeros(len(y))
        for tr, te in folds:
            model = LogisticRegression(C=1e4, max_iter=1000).fit(x[tr, None], y[tr])
            p[te] = model.predict_proba(x[te, None])[:, 1]
        report(f"{label} | {t} | platt", p)
    for cal in ["none", "platt"]:
        pred = np.zeros(len(y))
        for f, (tr, te) in enumerate(folds):
            # Learn positive bounded LSE temperature from training-fold log loss.
            initial = [np.log(0.5)] if cal == "none" else [np.log(0.5), 0.0, 0.0]
            bounds = (
                [(np.log(0.03), np.log(5.0))]
                if cal == "none"
                else [(np.log(0.03), np.log(5.0)), (-3, 3), (-10, 10)]
            )

            # Pooling takes a subset, so use subset-specific slice counts.
            def objective(v):
                t = np.exp(v[0])
                x = t * (logsumexp(a[tr] / t, axis=1) - np.log(n[tr]))
                s = x if cal == "none" else np.exp(v[1]) * x + v[2]
                return np.mean(np.logaddexp(0, s) - y[tr] * s) + (
                    0 if cal == "none" else 0.001 * (v[1] ** 2 + v[2] ** 2)
                )

            result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
            assert result.success, result.message
            t = np.exp(result.x[0])
            x = t * (logsumexp(a[te] / t, axis=1) - np.log(n[te]))
            s = x if cal == "none" else np.exp(result.x[1]) * x + result.x[2]
            pred[te] = expit(s)
            parameters.append(
                dict(
                    method=label,
                    fit=cal,
                    fold=f,
                    temperature=t,
                    coefficients=result.x.tolist(),
                )
            )
        report(f"{label} | learned T | {cal}", pred)


for k in runs:
    test(k, z[k])
ensembles = [
    ("maxvit", "coatnet0"),
    ("maxvit", "convnext_tiny"),
    ("coatnet0", "convnext_tiny"),
    tuple(runs),
]
for members in ensembles:
    name = "+".join(members)
    test(name + " mean slice logits", np.mean([z[k] for k in members], axis=0))
    for t in ["max", 0.5, 1.0]:
        X = np.column_stack([pool(pad(z[k]), t) for k in members])
        for C in [0.01, 0.1, 1.0]:
            pred = np.zeros(len(y))
            for tr, te in folds:
                mu = X[tr].mean(0)
                sd = X[tr].std(0)
                sd[sd == 0] = 1
                mod = LogisticRegression(C=C, max_iter=1000).fit(
                    (X[tr] - mu) / sd, y[tr]
                )
                pred[te] = mod.predict_proba((X[te] - mu) / sd)[:, 1]
            report(f"{name} separate pooled logits | {t} | C={C}", pred)
out = args.output
pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
pd.DataFrame(scores).to_csv(out / "oof_series_scores.csv", index=False)
(out / "provenance.json").write_text(
    json.dumps(
        dict(
            runs=prov,
            n_series=len(y),
            n_patients=len(set(patients)),
            positives=int(y.sum()),
            folds=5,
            seed=88,
            learned_parameters=parameters,
            warning="Exploratory patient-grouped calibration folds on fixed model-development validation cohort. Multiple comparisons; no independent test. Threshold for 5 FP is chosen on the full validation predictions, descriptive only.",
            lse="T * log(mean(exp(logit / T)))",
            learning="frozen slice logits; positive bounded temperature fit by training-fold log loss; optional positive-slope Platt fit jointly; multivariate L2 logistic standardization fit per fold",
        ),
        indent=2,
    )
)
print(
    pd.DataFrame(rows)
    .sort_values("auc", ascending=False)
    .head(20)
    .to_string(index=False)
)
