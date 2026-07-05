# Quick Start: ImageWoof Classification

This walkthrough trains a small 10-class ImageWoof classifier using the standard
SKPv2 classification path:

```text
Config -> simple2d Dataset -> classification Task -> timm model -> MLflow logs
```

ImageWoof is part of the fast.ai Imagenette/ImageWoof dataset family. It contains
10 dog-breed classes and is available in full-size, 320 px, and 160 px variants.
This tutorial uses the 160 px archive because it is small enough for a quick
smoke test. See the upstream dataset page for more detail:
https://github.com/fastai/imagenette

## 1. Build The Environment

From the repository root:

```bash
uv sync --group train --group dev
```

For this quick start you do not need the optional `medical`, `video`, or `wds`
groups.

## 2. Download ImageWoof-160

```bash
mkdir -p data
curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz \
  -o data/imagewoof2-160.tgz
tar -xzf data/imagewoof2-160.tgz -C data
```

After extraction, the important folders should look like:

```text
data/imagewoof2-160/
  train/
    n02086240/
    n02087394/
    ...
  val/
    n02086240/
    n02087394/
    ...
```

## 3. Create An SKPv2 Annotation CSV

SKPv2's reusable `simple2d` dataset expects a tabular annotation file. The image
path is stored relative to `cfg.data_dir`, the target is an integer class label,
and the `fold` column controls the train/validation split.

Here we set:

- `fold = 1` for ImageWoof's `train/` images.
- `fold = 0` for ImageWoof's `val/` images.
- `cfg.split_column = "fold"` and `cfg.fold = 0` in the config, so SKPv2 trains
  on `fold != 0` and validates on `fold == 0`.

```bash
uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path("data/imagewoof2-160")
class_dirs = sorted((root / "train").iterdir())
class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}

rows = []
for split, fold in [("train", 1), ("val", 0)]:
    for class_dir in sorted((root / split).iterdir()):
        if not class_dir.is_dir():
            continue
        target = class_to_idx[class_dir.name]
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append(
                {
                    "image": str(image_path.relative_to(root)),
                    "target": target,
                    "class_name": class_dir.name,
                    "fold": fold,
                }
            )

df = pd.DataFrame(rows)
df.to_csv(root / "annotations.csv", index=False)
(root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))

print(df.head())
print(f"\nWrote {len(df):,} rows to {root / 'annotations.csv'}")
print(f"Wrote class map to {root / 'class_to_idx.json'}")
PY
```

## 4. Review The Config

This repository includes `src/skp/configs/imagewoof_quickstart.py`. It is shown
below so the quick-start experiment remains easy to inspect and modify:

```python
import albumentations as A
import cv2

from skp.configs import Config
from skp.configs.defaults import (
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
classification_2d_defaults(cfg)

# Task/model
cfg.project = "imagewoof_quickstart"
cfg.task = "classification"
cfg.model = "classification.net2d"
cfg.backbone = "resnet18"
cfg.pretrained = False  # Set True if you want timm ImageNet weights.
cfg.num_input_channels = 3
cfg.num_classes = 10
cfg.pool = "avg"
cfg.dropout = 0.0
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

# Data
cfg.fold = 0
cfg.split_column = "fold"
cfg.dataset = "simple2d"
cfg.data_dir = "./data/imagewoof2-160"
cfg.annotations_file = "./data/imagewoof2-160/annotations.csv"
cfg.inputs = "image"
cfg.targets = ["target"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR

# Optimization
cfg.loss = "classification.CrossEntropyLoss"
cfg.loss_params = {}
cfg.batch_size = 64
cfg.val_batch_size = cfg.batch_size
cfg.num_epochs = 1
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"

# Metrics/checkpointing
cfg.metrics = ["classification.Accuracy"]
cfg.metric_activation_fn = "softmax"
cfg.val_metric = "accuracy"
cfg.val_track = "max"
cfg.mlflow_system_monitor = False

# Images/transforms
cfg.image_height = 160
cfg.image_width = 160
cfg.train_transforms = A.Compose(
    [
        A.Resize(height=cfg.image_height, width=cfg.image_width, p=1),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
    ]
)
cfg.val_transforms = A.Compose(
    [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]
)

# Smaller worker count is friendlier for laptops and CI smoke tests.
cfg.num_workers = 2
```

## 5. Run A Tiny CPU Smoke Test

This checks that the config, dataset, model, loss, metrics, optimizer, scheduler,
trainer, checkpointing, and MLflow logger all connect.

```bash
CUDA_VISIBLE_DEVICES="" uv run skp-train imagewoof_quickstart \
  --accelerator cpu \
  --devices 1 \
  --strategy auto \
  --precision 32-true \
  --no_sync_batchnorm \
  --debug \
  --debug_num_workers 0 \
  --limit_train_batches 5 \
  --limit_val_batches 5
```

Why `--no_sync_batchnorm`? The default CLI behavior enables SyncBatchNorm for
DDP training. When running on CPU or with `--strategy auto`, disable it.

On machines with a CUDA-enabled PyTorch build but an incompatible or unavailable
driver, PyTorch may still emit a CUDA initialization warning during backward. If
Lightning reports `GPU available: False` and the command completes, the CPU smoke
test still passed. A CPU-only PyTorch build or matching NVIDIA driver will silence
that warning.

## 6. Run A Real Single-GPU Experiment

```bash
uv run skp-train imagewoof_quickstart \
  --accelerator cuda \
  --devices 1 \
  --strategy auto \
  --no_sync_batchnorm
```

For multi-GPU DDP training, use the default DDP strategy:

```bash
uv run skp-train imagewoof_quickstart \
  --accelerator cuda \
  --devices 2 \
  --strategy ddp
```

If `--devices` is omitted for CUDA training, SKPv2 uses all visible GPUs.

Outputs are written under:

```text
experiments/imagewoof_quickstart/<run_id>/fold0/
```

That folder contains checkpoints, MLflow artifacts, the serialized final config,
and the `best.ckpt` symlink. MLflow names the run
`imagewoof_quickstart/<run_id>` so shared experiments remain easy to scan.

## 7. Common Tweaks

Use pretrained weights:

```python
cfg.pretrained = True
```

Train longer:

```python
cfg.num_epochs = 10
```

Use a different backbone:

```python
cfg.backbone = "convnext_tiny"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
```

Disable the scheduler:

```python
cfg.scheduler = None
cfg.scheduler_params = {}
```

Run a small built-in sweep:

```python
cfg.hyperparameter_sweep = {
    "num_trials": 4,
    "optimizer_params__lr": {"min": 1e-4, "max": 3e-3, "scaling": "log"},
    "batch_size": {"feasible_points": [32, 64]},
}
```

## 8. What This Demonstrates

This toy project exercises the intended SKPv2 workflow:

1. Start from the reusable template.
2. Add a project-specific config.
3. Convert the dataset into a simple annotations table.
4. Use a reusable dataset/task/model/loss/metric path.
5. Keep experiment outputs under `experiments/`.
6. Promote reusable fixes back into the SKPv2 template when you discover them.

For a real project, keep private data paths, labels, and project-only modeling
choices in the project repo. Move bug fixes, reusable utilities, broadly useful
tests, and documentation improvements back into the SKPv2 template.
