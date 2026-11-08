# ICH project decision log

See [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) for runs and results.
Reusable framework history remains in the root DECISIONS.md.

## Series sampling and schedule correctness

One sampled center slice per series per epoch limits domination by long series.
Positive and negative bins must cycle independently: selecting a bin does not
consume slices from the other bin. Empty-label-bin series fall back to their
available bin. Draws are reproducible by seed and epoch, including replay.

Shorter epochs require a different optimizer-step budget. Derive it from global
series count, DDP sharding, batch size and accumulation; preserve the final partial
batch. Runtime assertions compare the planned schedule to Lightning and record
optimizer/scheduler alignment. Do not reuse a full-slice epoch schedule.

## Primary metric and comparisons

Select checkpoints by any-hemorrhage series AUC from maximum slice scores.
Keep all six slice targets during training. Compare backbone arms at matched
completed validation epochs and label best/final results separately. Schedule
ablations require explicit step/exposure budgets and LR-phase distinctions.
Positive-slice fraction is a proxy for axial extent, not bleed volume.

## Controlled backbone changes

Keep downstream inputs, data, augmentation, loss, optimization and seed fixed.
CoAtNet passes the same 512px size to its constructor for positional adaptation.
CSN adds an explicit depth dimension and changes pretraining to Kinetics, which
must be disclosed as a comparison confounder. NaFlexViT uses native token average
pooling excluding register tokens and its pretrained final normalization. Adapt
its linear HWC patch weights by repeating RGB channel groups and scaling by 1/3,
matching the other backbones' nine-channel adaptation without random fallback.

GPU memory preflight must pass at the configured batch sizes before each added
arm starts. Pending-arm removal must preserve the active run. The revised queue
controller can wait for an existing per-run launcher before starting later arms.

## Weak labels and calibration: proposed, not launched

Discussed series-level MIL on frozen logits, normalized log-sum-exp temperature
ablations, monotone calibration of series-max scores, and selective slice
pseudo-labeling. These remain proposals; the committed queues run backbone
ablations only. Calibration needs patient-separated fitting/evaluation. Missing
subtype labels must not become negatives, and uncertain slices in positive
series should not automatically receive negative pseudo-labels.
