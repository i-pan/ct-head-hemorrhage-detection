from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "hf_release"
MODEL_DIR = RELEASE / "model"
SPACE_DIR = RELEASE / "space"

CLASSIFIER_CHECKPOINT = ROOT / (
    "experiments/rsna_ich_maxvit_tiny_fixedval_9ch/maxvit_tiny_9ch_20260714/"
    "fold0/checkpoints/last.ckpt"
)
SEQUENCE_CHECKPOINT = ROOT / (
    "experiments/rsna_ich_maxvit_tiny_sequence_bigru/"
    "seqsweep_01_sliceheavy_maxvit_20260714/fixed_split/checkpoints/best.ckpt"
)
SEGMENTATION_PATTERN = ROOT / (
    "experiments/bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen/"
    "maxvit_final_pseudolabel_any2_20260715/fold{fold}/checkpoints/best.ckpt"
)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    return checkpoint["state_dict"]


def strip_prefix(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(prefix): value.contiguous()
        for key, value in state.items()
        if key.startswith(prefix)
    }


def classifier_to_feature_encoder_key(key: str) -> str | None:
    if not key.startswith("backbone.") or key.startswith("backbone.head."):
        return None
    key = key.removeprefix("backbone.")
    if key.startswith("stages."):
        key = "stages_" + key.removeprefix("stages.")
    return key


def export_weights() -> None:
    (MODEL_DIR / "classifier").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "sequence").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "segmentation").mkdir(parents=True, exist_ok=True)

    classifier = strip_prefix(checkpoint_state(CLASSIFIER_CHECKPOINT), "model.")
    save_file(
        classifier,
        MODEL_DIR / "classifier" / "model.safetensors",
        metadata={
            "architecture": "maxvit_tiny_tf_512.in1k",
            "input_channels": "9",
            "source_checkpoint": "last.ckpt",
        },
    )

    sequence = strip_prefix(checkpoint_state(SEQUENCE_CHECKPOINT), "model.")
    save_file(
        sequence,
        MODEL_DIR / "sequence" / "model.safetensors",
        metadata={
            "architecture": "two-layer bidirectional GRU",
            "source_checkpoint": "best.ckpt",
            "seed": "88",
        },
    )

    classifier_encoder = {}
    for key, value in classifier.items():
        encoder_key = classifier_to_feature_encoder_key(key)
        if encoder_key is not None:
            classifier_encoder[encoder_key] = value
    for fold in range(5):
        state = strip_prefix(
            checkpoint_state(Path(str(SEGMENTATION_PATTERN).format(fold=fold))),
            "model.",
        )
        encoder = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if encoder.keys() != classifier_encoder.keys():
            missing = sorted(encoder.keys() - classifier_encoder.keys())
            extra = sorted(classifier_encoder.keys() - encoder.keys())
            raise RuntimeError(
                f"Fold {fold} encoder is incompatible with classifier: "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
        changed_encoder_state = [
            key
            for key, value in encoder.items()
            if not torch.equal(value, classifier_encoder[key])
        ]
        if changed_encoder_state:
            raise RuntimeError(
                f"Fold {fold} changed frozen encoder state: "
                f"{changed_encoder_state[:5]}"
            )

        compact = {
            key: value
            for key, value in state.items()
            if key.startswith(("decoder.", "segmentation_head."))
        }
        save_file(
            compact,
            MODEL_DIR / "segmentation" / f"fold{fold}.safetensors",
            metadata={
                "architecture": "DeepLabV3Plus",
                "fold": str(fold),
                "source_checkpoint": "best.ckpt",
                "shared_encoder": "../classifier/model.safetensors",
                "encoder_state": "canonical classifier encoder; permanently eval",
            },
        )


def write_checksums() -> None:
    records = []
    for path in sorted(MODEL_DIR.rglob("*")):
        if (
            not path.is_file()
            or path.name == "checksums.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": str(path.relative_to(MODEL_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    (MODEL_DIR / "checksums.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    export_weights()
    write_checksums()
    print(f"Exported model artifacts to {MODEL_DIR}")
