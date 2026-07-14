#!/usr/bin/env python
"""Compare and ensemble sequence architectures using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ich_sequence_sweep import class_aucs, export_predictions, optimize_blend


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
GRU_RUNS = [
    "seqsweep_01_sliceheavy_20260713",
    "seqsweep_15_sliceheavy_seed89_20260713",
    "seqsweep_16_sliceheavy_seed90_20260713",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="logs/sequence_architecture_sweep/manifest.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_predictions(run_id: str, args: argparse.Namespace) -> dict[str, np.ndarray]:
    path = export_predictions(
        run_id, args.device, args.batch_size, args.num_workers, split="val"
    )
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def summarize(name: str, members: list[str], predictions: list[dict]) -> dict:
    reference = predictions[0]
    for candidate in predictions[1:]:
        for key in ["slice_targets", "series_targets", "slice_base_logits", "series_base_logits"]:
            np.testing.assert_array_equal(reference[key], candidate[key])

    slice_logits = np.mean([item["slice_logits"] for item in predictions], axis=0)
    series_logits = np.mean([item["series_logits"] for item in predictions], axis=0)
    slice_auc = class_aucs(reference["slice_targets"], slice_logits)
    series_auc = class_aucs(reference["series_targets"], series_logits)
    baseline_slice_auc = class_aucs(
        reference["slice_targets"], reference["slice_base_logits"]
    )
    baseline_series_auc = class_aucs(
        reference["series_targets"], reference["series_base_logits"]
    )
    slice_alpha, slice_blend_auc, _ = optimize_blend(
        reference["slice_targets"], reference["slice_base_logits"], slice_logits
    )
    series_alpha, series_blend_auc, _ = optimize_blend(
        reference["series_targets"], reference["series_base_logits"], series_logits
    )
    result = {
        "name": name,
        "members": ",".join(members),
        "num_heads": len(members),
        "slice_auc_any": slice_auc[-1],
        "slice_auc_mean": slice_auc.mean(),
        "series_auc_any": series_auc[-1],
        "series_auc_mean": series_auc.mean(),
        "blend_slice_auc_any": slice_blend_auc[-1],
        "blend_slice_auc_mean": slice_blend_auc.mean(),
        "blend_series_auc_any": series_blend_auc[-1],
        "blend_series_auc_mean": series_blend_auc.mean(),
        "slice_blend_alphas": json.dumps(slice_alpha.tolist()),
        "series_blend_alphas": json.dumps(series_alpha.tolist()),
        "blend_slice_any_delta_vs_classifier": (
            slice_blend_auc[-1] - baseline_slice_auc[-1]
        ),
        "blend_slice_mean_delta_vs_classifier": (
            slice_blend_auc.mean() - baseline_slice_auc.mean()
        ),
        "blend_series_any_delta_vs_classifier": (
            series_blend_auc[-1] - baseline_series_auc[-1]
        ),
        "blend_series_mean_delta_vs_classifier": (
            series_blend_auc.mean() - baseline_series_auc.mean()
        ),
        "context_slice_any_delta_vs_classifier": (
            slice_auc[-1] - baseline_slice_auc[-1]
        ),
        "context_slice_mean_delta_vs_classifier": (
            slice_auc.mean() - baseline_slice_auc.mean()
        ),
        "context_series_any_delta_vs_classifier": (
            series_auc[-1] - baseline_series_auc[-1]
        ),
        "context_series_mean_delta_vs_classifier": (
            series_auc.mean() - baseline_series_auc.mean()
        ),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"context_slice_auc_{class_name}"] = slice_auc[index]
        result[f"context_slice_delta_{class_name}_vs_classifier"] = (
            slice_auc[index] - baseline_slice_auc[index]
        )
        result[f"context_series_auc_{class_name}"] = series_auc[index]
        result[f"context_series_delta_{class_name}_vs_classifier"] = (
            series_auc[index] - baseline_series_auc[index]
        )
        result[f"blend_slice_auc_{class_name}"] = slice_blend_auc[index]
        result[f"blend_slice_delta_{class_name}_vs_classifier"] = (
            slice_blend_auc[index] - baseline_slice_auc[index]
        )
        result[f"blend_series_auc_{class_name}"] = series_blend_auc[index]
        result[f"blend_series_delta_{class_name}_vs_classifier"] = (
            series_blend_auc[index] - baseline_series_auc[index]
        )
    result["blend_slice_regressed_class_count"] = int(
        np.sum(slice_blend_auc < baseline_slice_auc)
    )
    result["blend_series_regressed_class_count"] = int(
        np.sum(series_blend_auc < baseline_series_auc)
    )
    result["blend_slice_worst_class_delta_vs_classifier"] = float(
        np.min(slice_blend_auc - baseline_slice_auc)
    )
    result["blend_series_worst_class_delta_vs_classifier"] = float(
        np.min(series_blend_auc - baseline_series_auc)
    )
    result["context_slice_regressed_class_count"] = int(
        np.sum(slice_auc < baseline_slice_auc)
    )
    result["context_series_regressed_class_count"] = int(
        np.sum(series_auc < baseline_series_auc)
    )
    result["context_slice_worst_class_delta_vs_classifier"] = float(
        np.min(slice_auc - baseline_slice_auc)
    )
    result["context_series_worst_class_delta_vs_classifier"] = float(
        np.min(series_auc - baseline_series_auc)
    )
    return result


def main() -> None:
    args = parse_args()
    with Path(args.manifest).open() as f:
        new_records = json.load(f)
    groups = {"bigru": GRU_RUNS}
    for architecture in ["lstm", "transformer"]:
        groups[architecture] = [
            record["run_id"]
            for record in new_records
            if record["architecture"] == architecture
        ]

    cache = {
        run_id: load_predictions(run_id, args)
        for run_ids in groups.values()
        for run_id in run_ids
    }
    rows = []
    for architecture, run_ids in groups.items():
        for run_id in run_ids:
            rows.append(summarize(run_id, [run_id], [cache[run_id]]))
        rows.append(
            summarize(
                f"{architecture}_seed_ensemble",
                run_ids,
                [cache[run_id] for run_id in run_ids],
            )
        )
    all_runs = [run_id for run_ids in groups.values() for run_id in run_ids]
    rows.append(
        summarize(
            "all_architectures_ensemble",
            all_runs,
            [cache[run_id] for run_id in all_runs],
        )
    )
    results = pd.DataFrame(rows)
    selected_bigru = results.loc[results.name == GRU_RUNS[0]].iloc[0]
    comparison_metrics = [
        "blend_slice_auc_any",
        "blend_slice_auc_mean",
        "blend_series_auc_any",
        "blend_series_auc_mean",
    ]
    for metric in comparison_metrics:
        results[f"{metric}_delta_vs_selected_bigru"] = (
            results[metric] - selected_bigru[metric]
        )
    results = results.sort_values(
        ["blend_slice_auc_any", "blend_slice_auc_mean"], ascending=False
    )
    output = Path("logs/sequence_architecture_sweep/validation_results.csv")
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved validation-only comparison to {output}")


if __name__ == "__main__":
    main()
