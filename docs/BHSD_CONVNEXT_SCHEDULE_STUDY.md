# BHSD binary segmentation schedule study (2026-11-10)

Initial study: ConvNeXt-Tiny, five patient-separated BHSD folds, independent
10/25/50-positive-pass training budgets (15 runs). Sampling/reduction ablations
are deferred until this establishes a suitable training duration.

- Initialize the complete encoder strictly from the final 50-epoch stride-16
  spatial-logit hemorrhage classifier. Retain its trained head normalization;
  replace classification logits with a U-Net decoder.
- Four encoder stages at strides 4/8/16/16; decoder fuses actual skip sizes then
  upsamples to 512. Input: three contiguous slices, three windows each; output:
  binary any-hemorrhage mask for the center slice.
- Read original BHSD NIfTI HU, reorient with the verified BHSD transform, cache
  losslessly as float32. Apply classification windows (40,80), (80,200),
  (600,2800), normalize to [-1,1]. Missing neighbors are zero-window channels.
  Preparation checks all mask-presence labels, patient splits, classification
  exclusion, and one center PNG per series against recomputed windowing.
- All positive center slices, shuffle without replacement once per pass; retain
  final partial batch. Batch32 on one GPU per run, independent folds on two GPUs.
- Equal-weight per-slice soft Dice + mean pixel BCE. Classification-like affine,
  intensity, noise, blur, flips; no cutout. Depth reversal probability0.25.
- AdamW weight_decay0.01. Encoder3e-5, decoder3e-4 peak LR. Warmup5% rounded up,
  cosine decay to encoder1e-7/decoder1e-6 at the final optimizer update. Ratio10
  throughout. Each budget starts independently from the same weights/seed88.
- Pass lengths by fold: 60,59,59,60,61 updates (1907,1878,1874,1890,1923 slices).
  Budgets are passes multiplied by these update counts, not the full slice count.
- Validate every pass on all slices. Primary checkpoint ranking: macro Dice over
  ground-truth-positive slices at0.5. Sweep0.1 through0.9, reporting best positive
  slice Dice and threshold (lowest threshold wins exact ties). Report macro
  volume Dice at0.5 and that same selected threshold, and all threshold curves.
  Volume Dice includes negative-slice predictions, without label-based gating.
- Save best AND most-recent checkpoints, optimizer/RNG states, raw float16
  full-resolution probability arrays, row-order manifests, per-slice overlap
  counts, metrics/provenance. Metrics use float32 probabilities before export
  quantization. Final checkpoints are the main cross-run comparison.
- Known negative slices may be gated to empty masks during later pseudo-labeling;
  this study also reports ungated negative-slice false-positive fractions.

Run with `.venv/bin/python scripts/run_bhsd_convnext_schedules.py` after
`.venv/bin/python scripts/train_bhsd_convnext_schedule.py --prepare` and `--smoke`.
The queue refuses to overwrite prior runs or continue after a failed run.

Queue/status: `logs/bhsd_convnext_schedules_20261110/queue.json`.
Results: `experiments/bhsd_convnext_s16_binary_schedules_20261110/passes_XX/foldN/`.
Each run has `status.json`, `metrics.jsonl`, `config.json`, `last.pt`, `best.pt`,
and `validation_predictions_{final,best}.{npy,json}`. This standalone study logs
locally rather than to MLflow. The scheduler runs as the user systemd service
`ich-bhsd-convnext-schedules-20261110.service`.

Verification: scheduler endpoint/ratio tests across all15 schedules; synthetic
positive-only/volume Dice checks; loss/window numerical tests; strict encoder
weight loading; real batch32 two-step forward/backward and full-fold validation
with prediction export. GPU peak15.32GiB on the smoke test.

## Queue cancellation update

The user cancelled all 50-pass runs. Fold 0 was terminated shortly after launch;
folds 1–4 were cancelled before launch. Remaining 25-pass training continues.
The original controller is SIGSTOPped to disable further dispatch; observer
`ich-bhsd-finish25-20261110.service` tracks the preserved training and retires
the original controller after it exits. Existing partial artifacts are retained.

## Sampling and Dice reduction ablations launched 2026-11-10

The 10- and 25-pass five-fold schedule runs completed. The remaining study uses
25 baseline-equivalent passes: five new configurations, five folds each.
Arms are `p100_batch`, `p075_slice`, `p075_batch`, `p050_slice`, `p050_batch`.
The existing positive-only per-slice 25-pass runs are the reference.

All arms preserve each fold's baseline sample count per epoch (1907,1878,1874,
1890,1923) and batch sizes, including the final partial batch. Full mixed batches
contain24/8 or16/16 positives/negatives; the last partial batch rounds the positive
quota to nearest integer, halves rounded up. Bins independently reshuffle after
exhaustion and persist across epoch boundaries. Per-epoch sampled class counts
are recorded. For mixed arms, "pass" means baseline-equivalent exposure, not one
complete traversal of all positive slices. Total updates remain1500,1475,1475,
1500,1525; validation every60,59,59,60,61 updates respectively.

Only training Dice reduction and sampling change. BCE remains mean pixel BCE;
batch Dice pools all batch pixels before computing its ratio. Architecture,
classifier initialization, augmentation, seed, optimizer and LR schedule remain
fixed. The positive-only batch-Dice arm uses the original positive-only shuffle.

Validation loss deliberately remains **per-slice Dice plus mean BCE in every
arm**, matching the reference, even when the training loss uses batch Dice.
Thus validation loss is comparable across arms; training-loss magnitudes across
Dice reductions are different objectives. All positive-slice, threshold sweep,
volume, negative-slice FP and best/final export metrics remain unchanged.

Verification:7 tests pass, including independent-bin cycle/ratio tests, all-fold
budget and validation-manifest comparisons, and AST equality against the saved
baseline for architecture, windows, LR schedule, validation and metric routines.
A real mixed batch32/batch-Dice GPU smoke test passed forward/backward and full
validation/probability export, with15.32GiB peak allocated GPU memory.

Launcher: `scripts/run_bhsd_convnext_ablations.py`.
Service: `ich-bhsd-convnext-ablations-20261110.service`.
Queue: `logs/bhsd_convnext_ablations_20261110/queue.json`.
Results: `experiments/bhsd_convnext_s16_binary_ablations_20261110/<arm>/foldN/`.
Baseline references are recorded in the queue. Each run saves its effective
sampling/loss configuration, source snapshots, class exposure counts, metrics,
checkpoints and both best/final prediction exports. Metrics remain in local JSONL.

## Selected recipe and additional backbone teachers

Following final-checkpoint negative-slice gating analysis, Ian selected
positive-only training with batch Dice plus BCE, using final25-pass checkpoints.
ConvNeXt gated volume Dice is0.6665 at0.5 (positive-slice Dice0.6153), compared
with0.6554/0.6111 for positive-only per-slice Dice. Details:
`diagnostics/bhsd_negative_gating_20261110/summary.csv`.

Launched15 additional teachers: MaxViT-Tiny, EffNetV2-M and CoAtNet-0, five
patient-separated folds per family. Each encoder initializes strictly from its
own final50-epoch stride16 spatial-logit classification checkpoint. The original
classification model constructors preserve stride modifications and, for
CoAtNet, the trained32x32 relative-position geometry. Full classifier weights
load strictly before retaining the encoder; segmentation decoder weights are new.

Use identical five-block U-Net decoder widths256/128/64/32/16 and the same binary
head. Only encoder-facing channel widths change. Select the four encoder scales
at strides4/8/16/16, preserving the ConvNeXt skip geometry; do not add an extra
stride2 skip merely because a family exposes one. Keep the original spatial head
normalization/pre-logit transformations. Inputs, normalization, augmentation,
positive-only shuffle, batch32,25 passes, fold splits, seed88, AdamW and10x LR
ratio, warmup/cosine step budgets, validation intervals and metric definitions
are unchanged. No gradient accumulation or activation checkpointing was needed.
Common validation loss remains per-slice Dice+BCE. Save best and most-recent
checkpoints/predictions, but use final checkpoints for the teacher collection.

All three families passed batch32 two-step forward/backward, finite gradients,
512x512 output, full validation and export. Peak allocated GPU memory:
MaxViT32.41GiB, EffNet40.34GiB, CoAtNet20.02GiB. Existing seven protocol tests pass.

Service: `ich-bhsd-teacher-backbones-20261110.service`.
Launcher: `scripts/run_bhsd_teacher_backbones.py`.
Queue: `logs/bhsd_teacher_backbones_20261110/queue.json`.
New results: `experiments/bhsd_teacher_backbones_20261110/<family>/foldN/`.
Registry: `experiments/bhsd_teacher_backbones_20261110/teacher_registry.json`.
The registry includes five completed ConvNeXt teachers and15 planned paths;
verify the25-pass endpoint before using a newly written last checkpoint.
When complete, the selected collection contains20 teachers across four families.
