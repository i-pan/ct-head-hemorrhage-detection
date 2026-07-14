#!/usr/bin/env python
"""Compare EfficientNetV2-M and MaxViT-Tiny on the fixed ICH validation set."""

from __future__ import annotations

import argparse
import copy
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


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
        "--efficientnet-checkpoint",
        default=(
            "experiments/rsna_ich_effv2m_fixedval_9ch/definite-crayfish-7226/"
            "fold0/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--maxvit-checkpoint",
        default=(
            "experiments/rsna_ich_maxvit_tiny_fixedval_9ch/"
            "maxvit_tiny_9ch_20260714/fold0/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--output-dir", default="diagnostics/ich_backbone_ensemble_20260714"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def class_aucs(targets: np.ndarray, logits: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(logits)
    return np.asarray(
        [
            roc_auc_score(targets[:, index], probabilities[:, index])
            for index in range(targets.shape[1])
        ]
    )


def series_max_logits(logits: np.ndarray, groups: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(logits)
    unique_groups = np.unique(groups)
    series_probabilities = np.stack(
        [probabilities[groups == group].max(axis=0) for group in unique_groups]
    )
    series_probabilities = np.clip(series_probabilities, 1e-6, 1 - 1e-6)
    return np.log(series_probabilities / (1.0 - series_probabilities))


def load_model(config_name: str, checkpoint: Path, device: str):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.pretrained = False
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["state_dict"]
    state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), cfg


def export_predictions(
    config_name: str,
    checkpoint: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if output.exists():
        with np.load(output) as loaded:
            return {key: loaded[key] for key in loaded.files}

    model, cfg = load_model(config_name, checkpoint, args.device)
    dataset = import_module(f"skp.datasets.{cfg.dataset}").Dataset(cfg, "val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    logits = []
    targets = []
    groups = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_dict = model({"x": x})
            logits.append(output_dict["logits"].float().cpu().numpy())
            targets.append(batch["y"].numpy())
            groups.append(batch["group_index"].numpy())
    arrays = {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "groups": np.concatenate(groups),
    }
    np.savez_compressed(output, **arrays)
    return arrays


def optimize_alphas(
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


def summarize(
    name: str,
    slice_targets: np.ndarray,
    series_targets: np.ndarray,
    slice_logits: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | str]:
    slice_auc = class_aucs(slice_targets, slice_logits)
    series_auc = class_aucs(
        series_targets,
        series_max_logits(slice_logits, groups),
    )
    result: dict[str, float | str] = {
        "condition": name,
        "slice_auc_mean": float(slice_auc.mean()),
        "slice_auc_any": float(slice_auc[-1]),
        "series_auc_mean_max": float(series_auc.mean()),
        "series_auc_any_max": float(series_auc[-1]),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"slice_auc_{class_name}"] = float(slice_auc[index])
        result[f"series_auc_{class_name}_max"] = float(series_auc[index])
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    efficientnet = export_predictions(
        "rsna_ich_effv2m_fixedval_9ch",
        Path(args.efficientnet_checkpoint),
        output_dir / "efficientnet_validation_predictions.npz",
        args,
    )
    maxvit = export_predictions(
        "rsna_ich_maxvit_tiny_fixedval_9ch",
        Path(args.maxvit_checkpoint),
        output_dir / "maxvit_validation_predictions.npz",
        args,
    )
    np.testing.assert_array_equal(efficientnet["targets"], maxvit["targets"])
    np.testing.assert_array_equal(efficientnet["groups"], maxvit["groups"])

    targets = efficientnet["targets"]
    groups = efficientnet["groups"]
    series_targets = np.stack(
        [targets[groups == group].max(axis=0) for group in np.unique(groups)]
    )
    equal_blend = (efficientnet["logits"] + maxvit["logits"]) / 2
    slice_alphas, optimized_slice_blend = optimize_alphas(
        targets,
        efficientnet["logits"],
        maxvit["logits"],
    )
    efficientnet_series = series_max_logits(efficientnet["logits"], groups)
    maxvit_series = series_max_logits(maxvit["logits"], groups)
    series_alphas, _ = optimize_alphas(
        series_targets,
        efficientnet_series,
        maxvit_series,
    )

    rows = [
        summarize(
            "efficientnetv2_m",
            targets,
            series_targets,
            efficientnet["logits"],
            groups,
        ),
        summarize(
            "maxvit_tiny",
            targets,
            series_targets,
            maxvit["logits"],
            groups,
        ),
        summarize(
            "equal_logit_blend",
            targets,
            series_targets,
            equal_blend,
            groups,
        ),
        summarize(
            "optimized_slice_logit_blend",
            targets,
            series_targets,
            optimized_slice_blend,
            groups,
        ),
    ]
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    metadata = {
        "maxvit_weight_by_class_for_slice": dict(
            zip(CLASS_NAMES, slice_alphas.tolist())
        ),
        "maxvit_weight_by_class_for_series": dict(
            zip(CLASS_NAMES, series_alphas.tolist())
        ),
        "blend_space": "logit",
        "alpha_grid": ALPHAS.tolist(),
        "efficientnet_checkpoint": str(Path(args.efficientnet_checkpoint).resolve()),
        "maxvit_checkpoint": str(Path(args.maxvit_checkpoint).resolve()),
        "test_set_accessed": False,
    }
    (output_dir / "blend_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
