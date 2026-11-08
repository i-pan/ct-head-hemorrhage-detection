"""Equal-weight final-checkpoint ensembles; excludes the 75-epoch schedule arm."""

from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "maxvit": "rsna_ich_maxvit_tiny_series_sampling/series_e050_p0.5_seed88_20261107_seriesauc",
    "effnet": "rsna_ich_effnetv2m_series_sampling/series_e050_p0.5_seed88_20261108_backbone_effnetv2m",
    "convnext": "rsna_ich_convnext_tiny_series_sampling/series_e050_p0.5_seed88_20261108_backbone_convnext_tiny",
}


def load_final(run):
    root = ROOT / "experiments" / run
    path = root / "validation_predictions_final.npz"
    final = torch.load(
        root / "fold0/checkpoints/last.ckpt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    assert final["epoch"] == 49 and final["global_step"] == 14950
    if path.exists():
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["epoch"] == 49 and metadata["global_step"] == 14950
        source = "explicit final export"
    else:
        best = torch.load(
            root / "fold0/checkpoints/best.ckpt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        assert best["epoch"] == 49 and best["global_step"] == 14950, (
            "Final predictions are not yet available"
        )
        assert best["state_dict"].keys() == final["state_dict"].keys()
        assert all(
            torch.equal(v, final["state_dict"][k])
            for k, v in best["state_dict"].items()
        ), "Best/final weights differ"
        path = root / "validation_predictions.npz"
        source = (
            "legacy best export; all checkpoint tensors verified identical to final"
        )
    with np.load(path) as a:
        arrays = {k: a[k] for k in a.files}
    return arrays, dict(path=str(path), epoch=50, source=source)


def main():
    torch.set_num_threads(2)
    data = {}
    provenance = {}
    for name, run in RUNS.items():
        data[name], provenance[name] = load_final(run)
    ref = data["maxvit"]
    for a in data.values():
        np.testing.assert_array_equal(a["targets"], ref["targets"])
        np.testing.assert_array_equal(a["groups"], ref["groups"])
        assert np.isfinite(a["logits"]).all()
    groups = ref["groups"]
    y = ref["targets"][:, -1]
    table = (
        pd.DataFrame(dict(group=groups, y=y))
        .groupby("group")
        .agg(target=("y", "max"), positive=("y", "sum"), n=("y", "size"))
    )
    table["fraction"] = table.positive / table.n
    negative = table.target.eq(0).to_numpy()

    def pool(x):
        return pd.Series(x).groupby(groups).max().loc[table.index].to_numpy()

    probabilities = {
        k: expit(a["logits"][:, -1].astype(float)) for k, a in data.items()
    }
    singles = {k: pool(p) for k, p in probabilities.items()}
    scores = dict(singles)
    for count in [2, 3]:
        for members in combinations(data, count):
            name = "+".join(members)
            scores[name + " / slice_mean_then_max"] = pool(
                np.mean([probabilities[k] for k in members], axis=0)
            )
            scores[name + " / mean_series_max"] = np.mean(
                [singles[k] for k in members], axis=0
            )
    rows = []
    for name, score in scores.items():
        threshold = np.nextafter(np.sort(score[negative])[::-1][5], np.inf)
        called = score >= threshold
        row = dict(
            model=name,
            auc=roc_auc_score(table.target, score),
            false_positives=int(called[negative].sum()),
            sensitivity=float(called[~negative].mean()),
            threshold=float(threshold),
        )
        for lo, hi, label in [
            (0, 0.1, "0_10"),
            (0.1, 0.25, "10_25"),
            (0.25, 0.5, "25_50"),
            (0.5, 1, "50_100"),
        ]:
            mask = (table.fraction > lo) & (table.fraction <= hi)
            row["detected_" + label] = int(called[mask].sum())
            row["n_" + label] = int(mask.sum())
        rows.append(row)
        table[name] = score
    output = ROOT / "diagnostics/ich_final_ensemble_20261108"
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    table.to_csv(output / "series_scores.csv")
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                runs=provenance,
                n_series=len(table),
                negatives=int(negative.sum()),
                policy="Equal weights; no tuning. Threshold on same validation negatives permits <=5 FP. Descriptive fixed validation, not OOF or independent test.",
            ),
            indent=2,
        )
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
