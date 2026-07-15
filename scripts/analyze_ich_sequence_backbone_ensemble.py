#!/usr/bin/env python
"""Compare selected EfficientNet and MaxViT sequence heads on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
ALPHAS = np.linspace(0.0, 1.0, 11, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--efficientnet-results", default="logs/sequence_sweep/results.csv"
    )
    parser.add_argument(
        "--efficientnet-experiments",
        default="experiments/rsna_ich_effv2m_sequence_bigru",
    )
    parser.add_argument(
        "--maxvit-results", default="logs/maxvit_sequence_sweep/results.csv"
    )
    parser.add_argument(
        "--maxvit-experiments",
        default="experiments/rsna_ich_maxvit_tiny_sequence_bigru",
    )
    parser.add_argument("--output-dir", default="logs/sequence_backbone_ensemble")
    return parser.parse_args()


def select_run(results_path: Path) -> str:
    results = pd.read_csv(results_path)
    baseline_mean = float(results.iloc[0].baseline_slice_auc_mean)
    eligible = results[results.slice_auc_mean >= baseline_mean - 0.001]
    if eligible.empty:
        eligible = results
    return str(eligible.sort_values("slice_auc_any", ascending=False).iloc[0].run_id)


def load_predictions(experiment_dir: Path, run_id: str) -> dict[str, np.ndarray]:
    path = experiment_dir / run_id / "validation_predictions.npz"
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def class_aucs(targets: np.ndarray, logits: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(logits)
    return np.asarray(
        [
            roc_auc_score(targets[:, index], probabilities[:, index])
            for index in range(targets.shape[1])
        ]
    )


def optimize_blend(
    targets: np.ndarray,
    efficientnet_logits: np.ndarray,
    maxvit_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.zeros(targets.shape[1], dtype=np.float32)
    blended = np.empty_like(efficientnet_logits)
    for class_index in range(targets.shape[1]):
        scores = []
        for alpha in ALPHAS:
            logits = (1.0 - alpha) * efficientnet_logits[
                :, class_index
            ] + alpha * maxvit_logits[:, class_index]
            scores.append(roc_auc_score(targets[:, class_index], sigmoid(logits)))
        selected[class_index] = ALPHAS[int(np.argmax(scores))]
        alpha = selected[class_index]
        blended[:, class_index] = (1.0 - alpha) * efficientnet_logits[
            :, class_index
        ] + alpha * maxvit_logits[:, class_index]
    return selected, blended


def summarize(name: str, targets: np.ndarray, logits: np.ndarray) -> dict:
    aucs = class_aucs(targets, logits)
    result = {
        "condition": name,
        "auc_mean": float(aucs.mean()),
        "auc_any": float(aucs[-1]),
    }
    result.update(
        {
            f"auc_{class_name}": float(aucs[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }
    )
    return result


def analyze_level(
    level: str,
    targets: np.ndarray,
    efficientnet_logits: np.ndarray,
    maxvit_logits: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    equal_blend = (efficientnet_logits + maxvit_logits) / 2
    alphas, optimized_blend = optimize_blend(
        targets, efficientnet_logits, maxvit_logits
    )
    rows = [
        summarize(f"efficientnet_{level}_contextual", targets, efficientnet_logits),
        summarize(f"maxvit_{level}_contextual", targets, maxvit_logits),
        summarize(f"equal_{level}_contextual_blend", targets, equal_blend),
        summarize(f"optimized_{level}_contextual_blend", targets, optimized_blend),
    ]
    return rows, alphas


def main() -> None:
    args = parse_args()
    efficientnet_run = select_run(Path(args.efficientnet_results))
    maxvit_run = select_run(Path(args.maxvit_results))
    efficientnet = load_predictions(
        Path(args.efficientnet_experiments), efficientnet_run
    )
    maxvit = load_predictions(Path(args.maxvit_experiments), maxvit_run)

    for key in [
        "slice_targets",
        "slice_series_index",
        "series_targets",
        "series_index",
    ]:
        np.testing.assert_array_equal(efficientnet[key], maxvit[key])

    slice_rows, slice_alphas = analyze_level(
        "slice",
        efficientnet["slice_targets"],
        efficientnet["slice_logits"],
        maxvit["slice_logits"],
    )
    series_rows, series_alphas = analyze_level(
        "series",
        efficientnet["series_targets"],
        efficientnet["series_logits"],
        maxvit["series_logits"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(slice_rows + series_rows)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    metadata = {
        "efficientnet_run_id": efficientnet_run,
        "maxvit_run_id": maxvit_run,
        "selection_guardrail": "slice_auc_mean >= same-backbone baseline - 0.001",
        "blend_space": "logit",
        "alpha_grid": ALPHAS.tolist(),
        "maxvit_weight_by_class_for_slice": dict(
            zip(CLASS_NAMES, slice_alphas.tolist())
        ),
        "maxvit_weight_by_class_for_series": dict(
            zip(CLASS_NAMES, series_alphas.tolist())
        ),
        "test_set_accessed": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
