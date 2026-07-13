# Decisions

This file tracks shared project decisions as SKPv2 is cleaned up into a reusable experiment template.

## 2026-06-03

- Initialized Git locally, but intentionally did not make an initial commit yet.
- Renamed the default branch from `master` to `main` for GitHub readiness.
- Initialized `uv` project metadata with project name `skp` and Python `>=3.11`.
- Kept the current source layout unchanged for now. The old package name was `skp`; the directory is currently named `src` to align with modern `uv`/Python packaging expectations. The exact import/package layout still needs a deliberate cleanup decision.
- Adopted the standard `src` package layout: repository source files now live under `src/skp`, preserving `skp` as the importable package name while keeping `src` as the packaging container.
- Added `skp-train = "skp.train:main"` as the future console-script entry point.
- Removed experiment-specific config directories from `src/skp/configs` to keep the template repository uncluttered from the outset.
- Kept only the shared `Config` class plus starter templates for 2D classification, 2D segmentation, 2D classification+segmentation, and 3D segmentation.
- Changed `Config` from silently returning `None` for missing attributes to a strict namespace. Missing attributes now raise `AttributeError`; optional config values should be read intentionally through `cfg.get("name", default)`.
- Added `Config.require(...)` for future explicit validation of required config keys.
- Updated core starter paths to use explicit optional config reads, with remaining legacy task/model modules to be handled as they are cleaned or retained.
- Introduced composable config defaults in `src/skp/configs/defaults.py`. Templates now apply relevant defaults first, then define/override experiment-specific values below.
- Removed prior non-MLflow experiment logging support. MLflow is now the only supported experiment logger.
- Removed experiment-specific dataset directories from `src/skp/datasets`.
- Kept reusable generic dataset loaders and added a custom dataset skeleton under `src/skp/datasets/templates`.
- Added `test` as an accepted dataset mode. Current reusable loaders treat `test` like `val`; `inference` remains the mode for running over the full annotation file.
- Added small dataset helper functions in `src/skp/datasets/_utils.py` for mode validation, annotation loading, mode/fold filtering, transform selection, and collate selection. Dataset-specific sample loading and target construction remain local to each dataset module.
- Standardized reusable datasets on albumentations-style transforms and removed TorchVision transform handling.
- Made dataset annotation loading infer common tabular file types from extension instead of assuming CSV.
- Updated the custom dataset template to mirror the common reusable dataset structure, so it can be copied and edited directly.
- Removed historical custom/stochastic loss modules from the reusable template.
- Kept MONAI loss wrappers and MONAI-backed task features optional by using lazy imports with explicit error messages.
- Added a custom loss template under `src/skp/losses/templates`.
- Audited reusable loss functions for correctness. Dice/Tversky-style losses should return conventional positive losses to minimize, subloss modules must be registered with PyTorch, and class-weighted reductions should normalize by the class weight sum.
- Kept EMA and MLflow system monitoring as core callbacks. Custom GPU stat logging is now opt-in because MLflow system metrics should be the default monitoring path.
- Default checkpoints now save full training state rather than weights only, so optimizer/scheduler/callback state including EMA can be resumed. `cfg.save_weights_only = True` remains available when a lean checkpoint is preferred.
- Kept custom metric calculations rather than relying on built-in TorchMetrics implementations. TorchMetrics remains useful as a state container, while sklearn/PyTorch helper functions provide explicit control over edge cases and competition-specific behavior.
- Reduced core metrics to reusable classification/regression helpers and generic segmentation Dice metrics. Experiment-specific metrics were removed from core and replaced with copy/paste templates under `src/skp/metrics/templates`.
- Reduced `src/skp/models` to reusable core model paths plus templates. The core keeps 2D classification, 2D/3D segmentation wrappers, generic decoders, 3D encoder helpers, and MedNeXt. Old domain-specific and experiment-specific model folders were removed from the reusable template.
- Consolidated 2D timm-backed segmentation on `segmentation.base` and removed the narrower `segmentation.timm_unet` wrapper. The reusable U-Net path should not use the raw input image as a decoder skip connection.
- Removed the stale standalone `segmentation.deeplabv3` wrapper. DeepLabV3+ remains available through generic segmentation wrappers via `DeepLabV3PlusDecoder` and `DeepLabV3PlusDecoder3d`.
- Moved MedNeXt SparK/sparse pretraining code out of `segmentation/mednext` and into `models/pretraining`, keeping `segmentation/mednext` focused on segmentation models.
- Removed legacy standalone `convnextv2_3d`, `convsqueeze3d`, and `efficientnet_3d` model files. Core 3D encoder support now stays concentrated in `models/encoders3d.py` and dedicated 3D segmentation wrappers.
- Removed `models/blocks`; its reusable 3D block helpers were no longer imported by retained models after the legacy 3D encoder cleanup.
- Kept the active sparse MedNeXt pretraining path mask-based rather than adding sparse-convolution library dependencies. Removed the older unused gather/scatter sparse normalization file so sparse logic lives in `models/pretraining/sparse_mednext_blocks.py`.
- Sparse pretraining masks should use an exact per-sample visible patch count, cover border voxels when dimensions are not divisible by patch size, and compute reconstruction loss over the full masked element count including channels.
- Sparse encoder convolutions should zero masked inputs before convolution, preserve the active support for stride-1 convolution blocks, propagate support through resolution-changing convolutions, and re-mask outputs before normalization or downstream layers.
- Channel-wise LayerNorm does not need sparse statistics because it normalizes across channels independently at each spatial location. In sparse blocks, `norm_type="layer"` uses the dense-equivalent channel LayerNorm and only re-applies the mask after affine.
- Simplified `LinearWarmupCosineAnnealingLR` to explicit `init_lr`, `max_lr`, `final_lr`, and either `warmup_steps` or `pct_start`. The optimizer LR remains the default peak LR unless `scheduler_params.max_lr` is set.
- When `scheduler_params.max_lr` is explicitly set, it must match the optimizer's actual param-group learning rates. Warmup percentages are interpreted as a fraction of total optimizer steps after DDP world-size and gradient-accumulation adjustment.
- Removed AdamP from the reusable optimizer core. AdamW remains the recommended default unless an experiment has a specific reason to choose another standard PyTorch optimizer.
- Kept dataloader construction and training samplers under `tasks/` because they are training policy rather than dataset definition. Added a small `BaseTask` for common Lightning plumbing while leaving each task's train/validation logic copy-paste friendly.
- Removed the experiment-specific aneurysm crop classification-segmentation task from the reusable template.
- Made task scheduler handling optional, fixed dataloader worker/WebDataset edge cases, and made 3D gradient norm logging opt-in through `cfg.log_grad_norm`.
- Moved lightweight hyperparameter search utilities from `toolbox/halton.py` to first-class `skp.search`, and moved progressive-resizing/sweep orchestration out of `train.py` into `skp.runners`.
- Cleaned `toolbox/` into named optional utility modules: `cropping`, `cv`, `dicom`, `ensembling`, `images`, `masks`, and `plotting`. Removed duplicated legacy DICOM utilities and the generic grab-bag modules.
- Kept DICOM support optional and lazy. Importing the core package or toolbox no longer requires pydicom/dicomsdl unless DICOM functions are called.
- Added machine-readable config field documentation in `CONFIG_FIELD_DOCS` next to the default values, so ambiguous defaults can be explained in one maintained location.
- Added dependency groups to `pyproject.toml`: `dev` for tests/linting, `train` for the core training stack, `medical` for MONAI/DICOM support, `video` for optional pytorchvideo backbones, and `wds` for WebDataset.
- Added a detailed README covering project structure, config philosophy, training flow, optional features, sweeps, tests, and extension points.
- Added `QUICK_START.md` using ImageWoof-160 as a small real-image classification example that exercises the standard config, `simple2d` dataset, classification task, timm model, optimizer/scheduler, metrics, checkpoints, and MLflow logging path.
- Removed default segmentation decoder types from config defaults. Decoder architecture is experiment-defining, so segmentation configs/templates should set `cfg.decoder_type` explicitly.
- Added `data/` to `.gitignore` for local quick-start datasets and other experiment data.
- CPU-only quick-start commands should set `CUDA_VISIBLE_DEVICES=""` and `--precision 32-true` so the smoke test remains independent of any visible but unusable CUDA installation.
- Environment reporting in `train.py` should not initialize cuDNN unless CUDA training is requested.
- MLflow system monitoring should degrade to a warning when the installed MLflow version does not expose the expected system-metrics monitor API.
- Run IDs should use `petname` in `<word>-<word>-<4digit_int>` format rather than opaque random hashes.
- Dataset configs must explicitly set `cfg.split_column`. K-fold splits use `cfg.fold`; fixed train/val/test splits use `cfg.train_split_values`, `cfg.val_split_values`, and `cfg.test_split_values` with guardrails for missing columns, overlapping values, and empty subsets.
- CLI batch-limit arguments should preserve integer-looking values as integer batch counts. Fractional values such as `0.25` remain percentage limits for Lightning.
- Pin the core training environment to the official PyTorch CUDA 12.4 wheel family (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`) with explicit uv source mapping for those packages only.
- GPU/DDP training should fail early when CUDA is unavailable, too few devices are visible, NCCL is missing, or SyncBatchNorm is requested outside CUDA DDP.
- Custom metrics should manually gather prediction/target state across distributed ranks before computing validation scores, preserving explicit metric control while making DDP metrics global rather than per-rank averages.
- Added `.env.example` and automatic `.env` loading so remote MLflow tracking
  URIs and credentials can be configured locally without committing secrets.
- Added a GCP GPU environment cleanup for CUDA runs that forces NCCL onto the
  socket backend, removes inherited `LD_LIBRARY_PATH`, and can be disabled with
  `GCP_NCCL_SOCKET_HACK_DISABLE=1`.
- Removed the default MLflow experiment name from `runtime_defaults`. Experiment
  configs now set `cfg.project` explicitly so cloned projects do not all log to
  the template-level `skp` experiment.
- Added `TEMPLATE_BACKLOG.md` as a slow lane for small reusable template improvements that should be batched instead of committed and pushed one by one.
- MLflow run names now use `<config_name>/<run_id>` while local artifacts keep
  the existing `experiments/<config_name>/<run_id>/...` layout, making shared
  experiments easier to scan without opening run details.
- CUDA training now defaults to `--devices -1`, resolving to all visible CUDA
  devices during trainer validation before Lightning trainer construction.

## 2026-07-13

- Added `cfg.seed = 88` to runtime defaults and seed Python, NumPy, PyTorch, and
  DataLoader workers before constructing datasets and models. Controlled model
  comparisons must not depend on implicit process RNG state.
