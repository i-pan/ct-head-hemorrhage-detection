# SKPv2

SKPv2 is a personal deep learning experiment framework for rapid prototyping.
It is designed to be cloned at the start of a new experiment, edited directly,
and kept flexible enough for Kaggle-style iteration while still being readable
and reproducible for company work.

The project intentionally uses Python config files. They are more verbose than
YAML, but they make transforms, augmentations, loss composition, and small
experiment-specific decisions explicit and inspectable.

## Quick Start

```bash
git clone <repo-url> my-experiment
cd my-experiment
uv sync --group train --group dev
uv run pytest
```

Copy one of the starter configs from `src/skp/configs/templates/` into
`src/skp/configs/`, edit the data paths and experiment settings, then run:

```bash
uv run skp-train my_config --devices 1 --strategy auto --accelerator cuda
```

For two-GPU DDP:

```bash
uv run skp-train my_config --accelerator cuda --devices 2 --strategy ddp
```

The CLI validates CUDA availability, visible device count, NCCL availability,
and SyncBatchNorm compatibility before the Lightning trainer starts.

For a CPU smoke test, use:

```bash
uv run skp-train my_config \
  --devices 1 \
  --strategy auto \
  --accelerator cpu \
  --no_sync_batchnorm \
  --debug
```

For a complete toy example using ImageWoof, see `QUICK_START.md`.

## Using SKPv2 For New Projects

The recommended workflow is to treat SKPv2 as a project template, not as a
third-party library dependency.

For a new experiment or company project:

```bash
git clone <skpv2-template-url> my-new-project
cd my-new-project
git remote rename origin template
git remote add origin <new-project-repo-url>
uv sync --group train --group dev
```

Then edit the cloned repository directly. This gives each experiment its own
configs, datasets, custom losses, custom metrics, and project-specific
dependencies without forcing those details back into the reusable template.

Why not install SKPv2 as an editable package from another repo? You can, but it
is usually worse for rapid experimentation. Most real experiments need small
changes to datasets, models, tasks, and configs. Keeping the code in the project
repo makes those changes visible, versioned, and easy to ship with the model.

Why not copy only `src/skp` into a new repo? That works, but cloning the full
template is safer because it keeps tests, README guidance, uv metadata,
dependency groups, templates, and decision history together.

## Environment Strategy

Each experiment should own its own uv environment and lockfile. Start from the
template dependency groups, then add project-specific packages in the experiment
repo:

```bash
uv sync --group train --group dev
uv add some-project-package
uv add --group medical pydicom
```

Use dependency groups to avoid making every project install every optional
stack:

- `dev`: tests and linting.
- `train`: core training stack.
- `medical`: MONAI/DICOM-related tooling.
- `video`: optional video backbones.
- `wds`: WebDataset support.

If a dependency is broadly useful for future experiments, add it to the template
and document why in `DECISIONS.md`. If it is only needed for one project, keep it
in that project's `pyproject.toml`.

The core training environment is pinned to the official PyTorch CUDA 12.4 wheel
family: `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`. The uv source
mapping uses the PyTorch CUDA 12.4 index only for those packages; everything
else resolves from PyPI.

## Updating The Template From A Project

Most useful template improvements are discovered during real experiments. The
cleanest way to send them back is to keep the template as an upstream remote in
each project.

Suggested remote layout inside an experiment repo:

```bash
git remote -v
# origin    <project repo>
# template  <SKPv2 template repo>
```

When you fix something reusable in a project:

1. Decide whether it is template-worthy. Good candidates are bug fixes,
   reusable datasets/tasks/losses/metrics, docs, tests, and dependency cleanup.
   Project-specific configs, labels, paths, private data logic, and competition
   tricks should usually stay in the project.
2. Put reusable changes on a small branch, separated from experiment-specific
   work when possible.
3. Cherry-pick or patch that branch into the SKPv2 template repo.
4. Add or update tests in the template.
5. Record the decision in `DECISIONS.md`.
6. Pull the template changes back into active projects when useful.

Useful commands:

```bash
# In the experiment repo, inspect reusable commits.
git log --oneline

# In the SKPv2 template repo, bring over a reusable fix.
git cherry-pick <commit-sha>

# In an experiment repo, pull template improvements back in.
git fetch template
git merge template/main
```

If a change mixes reusable and project-specific edits, prefer making a clean
patch manually in the template rather than cherry-picking the whole commit.

When several experiment repos are active at the same time, treat the SKPv2
template repo as the single source of truth for reusable framework code. Each
project can move at its own pace:

```bash
# In a project repo, pull the latest reusable template changes when ready.
git fetch template
git merge template/main
```

Do not automatically push every project fix back into the template. First decide
whether the change is broadly reusable, then port it back intentionally with a
small commit, tests, and a `DECISIONS.md` note when the change affects project
direction.

There is intentionally no built-in `update-template` command yet. Git already
handles fetches, merges, cherry-picks, and patches well, and template updates
often require human judgment about whether code is reusable or project-specific.
A future helper command may be useful if it stays thin, for example to inspect
template drift or generate a patch, but it should not hide merge decisions.

## Project Layout

```text
src/skp/
  callbacks/      Lightning callbacks: EMA, MLflow system monitor, GPU stats.
  configs/        Strict Python configs, defaults, and starter templates.
  datasets/       Reusable dataset classes plus copy/paste templates.
  losses/         Core classification, segmentation, combined, and optional MONAI losses.
  metrics/        Explicit custom metric implementations.
  models/         Classification, segmentation, pretraining, decoders, and templates.
  optim/          Optimizer construction, parameter groups, LR schedulers.
  tasks/          LightningModule wrappers and training dataloader/sampler policy.
  toolbox/        Optional analysis/preprocessing utilities outside the core train path.
  search.py       Lightweight built-in Halton-style hyperparameter search helpers.
  runners.py      Optional progressive-resize and hyperparameter-sweep runners.
  train.py        Main training entry point.
```

## Config Philosophy

Configs are strict namespaces. Accessing a missing attribute raises
`AttributeError`; optional values should use `cfg.get("name", default)`.

The normal pattern is:

```python
from skp.configs import Config
from skp.configs.defaults import runtime_defaults, dataloader_defaults

cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)

cfg.task = "classification"
cfg.model = "classification.net2d"
cfg.dataset = "simple2d"
cfg.split_column = "fold"
cfg.fold = 0
```

Defaults are applied first, and experiment-specific overrides are written below.
This keeps templates compact while avoiding silent `None` values for forgotten
settings. Field explanations live in `CONFIG_FIELD_DOCS` inside
`src/skp/configs/defaults.py`.

Core config groups:

- `runtime_defaults`: save paths, MLflow, checkpointing, EMA, debug/profile flags.
- `dataloader_defaults`: worker counts, samplers, WebDataset toggles, inference transforms.
- `classification_2d_defaults`: backbone loading, pooling, MixUp, sampling columns.
- `segmentation_2d_defaults`: decoder options, deep supervision, sliding-window validation.
- `cls_seg_2d_defaults`: joint segmentation/classification extras.
- `segmentation_3d_defaults`: 3D decoder/backbone options, stochastic heads, MONAI mixing.

## Dataset Splits

Every train/val/test dataset config must explicitly set `cfg.split_column`.
Inference mode reads the full annotation file and does not require a split.

For k-fold training, point `cfg.split_column` at the fold column and set the held
out fold:

```python
cfg.split_column = "fold"
cfg.fold = 0
```

SKPv2 trains on rows where `fold != cfg.fold`; validation and test modes use
rows where `fold == cfg.fold`.

For fixed splits, define mutually exclusive values for each mode:

```python
cfg.split_column = "split"
cfg.train_split_values = "train"
cfg.val_split_values = ["val", "valid"]
cfg.test_split_values = "test"
```

The split helper raises clear errors if the split column is missing, fixed split
values overlap, a fixed split mode is not defined, or a selected subset is empty.

## Starter Configs

Available templates:

- `classification_2d.py`
- `segmentation_2d.py`
- `cls_seg_2d.py`
- `segmentation_3d.py`

They are intentionally plain Python files. Copy one, rename it, and edit it
directly. Dataset and model templates also exist under `datasets/templates`,
`models/templates`, `losses/templates`, and `metrics/templates`.

## Training Flow

`skp.train` performs the main orchestration:

1. Parse CLI args and config overrides.
2. Import `skp.configs.<config_name>.cfg`.
3. Build model, loss, datasets, optimizer, scheduler, metrics, and task.
4. Build a Lightning `Trainer`.
5. Train, checkpoint, symlink `best.ckpt`, and save EMA weights if enabled.

Task files are LightningModule wrappers. Datasets define sample loading; tasks
own training-time policy such as samplers and dataloader construction. This is
why `samplers.py` lives under `tasks/`.

## CLI Overrides

Unknown CLI args are interpreted as config overrides:

```bash
uv run skp-train my_config --optimizer_params.lr 1e-4 --batch_size 16
```

Nested dictionary keys use dot notation for CLI overrides. Built-in sweeps use
double underscores for nested dictionaries, for example
`optimizer_params__lr`.

## Logging And Outputs

MLflow is the only experiment logger. By default, outputs are written under:

```text
experiments/<config_name>/<run_id>/fold<fold>/
```

Run IDs use the human-readable `petname` format `<word>-<word>-<4digit_int>`,
for example `silver-lake-0427`. Fixed-split runs without `cfg.fold` use:

```text
experiments/<config_name>/<run_id>/fixed_split/
```

Checkpoints are saved under `checkpoints/`, with `best.ckpt` symlinked to the
best monitored checkpoint.

## Optional Features

MONAI is optional and only required when enabling MONAI-backed losses,
3D MixUp/CutMix, or sliding-window inference.

WebDataset is optional and only imported when `cfg.wds = True`.

DICOM utilities live under `skp.toolbox.dicom` and lazily require `pydicom`.
`dicomsdl` is supported as an optional faster backend when installed.

Progressive resizing and built-in hyperparameter sweeps are available but live
outside the main training path in `skp.runners`.

To build every supported optional path into the environment, run:

```bash
uv sync --all-groups
```

## Hyperparameter Sweeps

Set `cfg.hyperparameter_sweep` to enable the lightweight built-in runner:

```python
cfg.hyperparameter_sweep = {
    "num_trials": 8,
    "optimizer_params__lr": {"min": 1e-5, "max": 1e-3, "scaling": "log"},
    "batch_size": {"feasible_points": [8, 16, 32]},
}
```

This is intentionally simple. For large distributed tuning jobs, prefer an
external orchestrator and pass CLI overrides into `skp-train`.

## Tests

Run:

```bash
uv run pytest
```

The current tests cover strict configs, default docs, dataset helpers, search
generation, schedulers, samplers/dataloaders, task base behavior, and toolbox
array helpers. GPU training integration tests are not included yet.

## Development Notes

- Keep experiment-specific files out of the reusable core.
- Prefer copyable templates over deep inheritance.
- Use `cfg.get(...)` only for genuinely optional settings.
- Keep MONAI, WebDataset, pydicom, and dicomsdl optional.
- Record shared cleanup decisions in `DECISIONS.md`.
