#!/usr/bin/env python
"""Export and analyze validation predictions from the ICH sequence sweep."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from skp.configs import Config
from skp.datasets.ich_sequence_features import Dataset
from skp.models.classification.ich_sequence import Net


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]
ALPHAS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="logs/sequence_sweep/manifest.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def run_dir(run_id: str) -> Path:
    return Path("experiments/rsna_ich_effv2m_sequence_bigru") / run_id


def load_run_config(path: Path) -> Config:
    with path.open("rb") as f:
        return Config(**pickle.load(f))


def export_predictions(
    run_id: str,
    device: str,
    batch_size: int,
    num_workers: int,
    split: str = "val",
) -> Path:
    root = run_dir(run_id)
    filename = "validation_predictions.npz" if split == "val" else f"{split}_predictions.npz"
    output_path = root / filename
    if output_path.exists():
        return output_path

    cfg = load_run_config(root / "config.pkl")
    dataset = Dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    model = Net(cfg)
    checkpoint = torch.load(
        root / "fixed_split/checkpoints/best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in [
            "slice_logits",
            "slice_base_logits",
            "slice_targets",
            "slice_series_index",
            "series_logits",
            "mil_logits",
            "series_base_logits",
            "series_targets",
            "series_index",
        ]
    }
    with torch.inference_mode():
        for batch in loader:
            gpu_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(gpu_batch)
            mask = batch["valid_mask"].bool()
            group = batch["group_index"].unsqueeze(1).expand_as(mask)
            arrays["slice_logits"].append(
                out["slice_logits"].detach().float().cpu()[mask].numpy()
            )
            arrays["slice_base_logits"].append(batch["base_logits"][mask].numpy())
            arrays["slice_targets"].append(batch["y"][mask].numpy())
            arrays["slice_series_index"].append(group[mask].numpy())
            arrays["series_logits"].append(
                out["series_logits"].detach().float().cpu().numpy()
            )
            arrays["mil_logits"].append(
                out["mil_logits"].detach().float().cpu().numpy()
            )
            base_probabilities = batch["base_logits"].sigmoid().masked_fill(
                ~mask.unsqueeze(-1), -1.0
            )
            base_series_probability = base_probabilities.amax(dim=1).clamp(
                1e-6, 1 - 1e-6
            )
            arrays["series_base_logits"].append(
                torch.logit(base_series_probability).numpy()
            )
            arrays["series_targets"].append(batch["series_y"].numpy())
            arrays["series_index"].append(batch["group_index"].numpy())

    combined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    np.savez_compressed(output_path, **combined)
    return output_path


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def class_aucs(targets: np.ndarray, logits: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(logits)
    aucs = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size < 2:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(targets[:, index], probabilities[:, index]))
    return np.asarray(aucs)


def optimize_blend(
    targets: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_alpha = np.zeros(targets.shape[1], dtype=np.float32)
    best_auc = np.zeros(targets.shape[1], dtype=np.float64)
    blended_logits = np.empty_like(candidate_logits)
    for class_index in range(targets.shape[1]):
        aucs = []
        for alpha in ALPHAS:
            logits = (
                (1.0 - alpha) * baseline_logits[:, class_index]
                + alpha * candidate_logits[:, class_index]
            )
            aucs.append(roc_auc_score(targets[:, class_index], sigmoid(logits)))
        best_index = int(np.argmax(aucs))
        best_alpha[class_index] = ALPHAS[best_index]
        best_auc[class_index] = aucs[best_index]
        alpha = best_alpha[class_index]
        blended_logits[:, class_index] = (
            (1.0 - alpha) * baseline_logits[:, class_index]
            + alpha * candidate_logits[:, class_index]
        )
    return best_alpha, best_auc, blended_logits


def paired_patient_bootstrap(
    targets: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
    patient_ids: np.ndarray,
    num_samples: int,
    seed: int = 88,
) -> dict[str, dict[str, float]]:
    unique_patients = np.unique(patient_ids)
    patient_rows = [np.flatnonzero(patient_ids == patient) for patient in unique_patients]
    rng = np.random.default_rng(seed)
    any_deltas = np.empty(num_samples, dtype=np.float64)
    mean_deltas = np.empty(num_samples, dtype=np.float64)
    for sample_index in range(num_samples):
        sampled = rng.integers(0, len(unique_patients), len(unique_patients))
        rows = np.concatenate([patient_rows[index] for index in sampled])
        baseline_auc = class_aucs(targets[rows], baseline_logits[rows])
        candidate_auc = class_aucs(targets[rows], candidate_logits[rows])
        any_deltas[sample_index] = candidate_auc[-1] - baseline_auc[-1]
        mean_deltas[sample_index] = candidate_auc.mean() - baseline_auc.mean()

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_delta": float(values.mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "probability_positive": float((values > 0).mean()),
        }

    return {"auc_any": summarize(any_deltas), "auc_mean": summarize(mean_deltas)}


def evaluate_run(record: dict, predictions: dict[str, np.ndarray]) -> tuple[dict, dict]:
    slice_auc = class_aucs(predictions["slice_targets"], predictions["slice_logits"])
    baseline_slice_auc = class_aucs(
        predictions["slice_targets"], predictions["slice_base_logits"]
    )
    series_auc = class_aucs(
        predictions["series_targets"], predictions["series_logits"]
    )
    mil_auc = class_aucs(predictions["series_targets"], predictions["mil_logits"])
    baseline_series_auc = class_aucs(
        predictions["series_targets"], predictions["series_base_logits"]
    )
    slice_alpha, slice_blend_auc, slice_blend_logits = optimize_blend(
        predictions["slice_targets"],
        predictions["slice_base_logits"],
        predictions["slice_logits"],
    )
    series_alpha, series_blend_auc, series_blend_logits = optimize_blend(
        predictions["series_targets"],
        predictions["series_base_logits"],
        predictions["series_logits"],
    )
    mil_alpha, mil_blend_auc, mil_blend_logits = optimize_blend(
        predictions["series_targets"],
        predictions["series_base_logits"],
        predictions["mil_logits"],
    )
    summary = {
        **record,
        "slice_auc_any": slice_auc[-1],
        "slice_auc_mean": slice_auc.mean(),
        "series_auc_any": series_auc[-1],
        "series_auc_mean": series_auc.mean(),
        "mil_auc_any": mil_auc[-1],
        "mil_auc_mean": mil_auc.mean(),
        "baseline_slice_auc_any": baseline_slice_auc[-1],
        "baseline_slice_auc_mean": baseline_slice_auc.mean(),
        "baseline_series_auc_any": baseline_series_auc[-1],
        "baseline_series_auc_mean": baseline_series_auc.mean(),
        "blend_slice_auc_any": slice_blend_auc[-1],
        "blend_slice_auc_mean": slice_blend_auc.mean(),
        "blend_series_auc_any": series_blend_auc[-1],
        "blend_series_auc_mean": series_blend_auc.mean(),
        "blend_mil_auc_any": mil_blend_auc[-1],
        "blend_mil_auc_mean": mil_blend_auc.mean(),
        "slice_any_alpha": slice_alpha[-1],
        "series_any_alpha": series_alpha[-1],
        "mil_any_alpha": mil_alpha[-1],
    }
    details = {
        "slice_alpha": slice_alpha,
        "series_alpha": series_alpha,
        "mil_alpha": mil_alpha,
        "slice_blend_logits": slice_blend_logits,
        "series_blend_logits": series_blend_logits,
        "mil_blend_logits": mil_blend_logits,
    }
    return summary, details


def main() -> None:
    args = parse_args()
    with Path(args.manifest).open() as f:
        manifest = json.load(f)
    series_metadata = pd.read_csv(
        "data/features/rsna_ich_effv2m_9ch/val_series.csv"
    )
    summaries = []
    details_by_run = {}
    predictions_by_run = {}
    for record in manifest:
        path = export_predictions(
            record["run_id"], args.device, args.batch_size, args.num_workers
        )
        with np.load(path) as loaded:
            predictions = {key: loaded[key] for key in loaded.files}
        summary, details = evaluate_run(record, predictions)
        summaries.append(summary)
        details_by_run[record["run_id"]] = details
        predictions_by_run[record["run_id"]] = predictions

    results = pd.DataFrame(summaries)
    baseline_mean = float(results.iloc[0].baseline_slice_auc_mean)
    eligible = results[results.slice_auc_mean >= baseline_mean - 0.001]
    if eligible.empty:
        eligible = results
    selected = eligible.sort_values("slice_auc_any", ascending=False).iloc[0]
    selected_id = selected.run_id
    predictions = predictions_by_run[selected_id]
    details = details_by_run[selected_id]
    slice_patient_ids = series_metadata.patient_id.to_numpy()[
        predictions["slice_series_index"]
    ]
    series_patient_ids = series_metadata.patient_id.to_numpy()[
        predictions["series_index"]
    ]
    bootstrap = {
        "selected_run_id": selected_id,
        "selection_guardrail": "slice_auc_mean >= baseline - 0.001",
        "slice_contextual": paired_patient_bootstrap(
            predictions["slice_targets"],
            predictions["slice_base_logits"],
            predictions["slice_logits"],
            slice_patient_ids,
            args.bootstrap_samples,
        ),
        "slice_blend": paired_patient_bootstrap(
            predictions["slice_targets"],
            predictions["slice_base_logits"],
            details["slice_blend_logits"],
            slice_patient_ids,
            args.bootstrap_samples,
        ),
        "series_contextual": paired_patient_bootstrap(
            predictions["series_targets"],
            predictions["series_base_logits"],
            predictions["series_logits"],
            series_patient_ids,
            args.bootstrap_samples,
        ),
        "series_blend": paired_patient_bootstrap(
            predictions["series_targets"],
            predictions["series_base_logits"],
            details["series_blend_logits"],
            series_patient_ids,
            args.bootstrap_samples,
        ),
        "mil_blend": paired_patient_bootstrap(
            predictions["series_targets"],
            predictions["series_base_logits"],
            details["mil_blend_logits"],
            series_patient_ids,
            args.bootstrap_samples,
        ),
    }
    output_dir = Path("logs/sequence_sweep")
    results.sort_values("slice_auc_any", ascending=False).to_csv(
        output_dir / "results.csv", index=False
    )
    with (output_dir / "bootstrap.json").open("w") as f:
        json.dump(bootstrap, f, indent=2)
    print(results.sort_values("slice_auc_any", ascending=False).to_string(index=False))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
