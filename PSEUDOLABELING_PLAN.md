# RSNA ICH Segmentation Pseudolabeling Plan

## Goal

Create visually useful hemorrhage localization masks for the larger RSNA
classification training set after the BHSD segmentation models finish training.
The primary target is an `any` hemorrhage mask. Subtype masks are secondary and
only need to support visualization or subtype coloring.

## Core Strategy

Use the ensemble `any` hemorrhage probability map as the localization authority.
Subtype assignment should be derived afterward from:

- slice-level classification labels, which determine which subtypes are possible
  on that slice; and
- subtype segmentation probabilities, only when a slice has more than one
  positive subtype label.

This keeps localization focused on the most clinically useful signal and avoids
letting weak subtype masks fragment or suppress the main hemorrhage region.

## Model Ensemble

Use the 5 trained 9-channel adjacent-slice BHSD segmentation models. The
single-slice models underperformed and should not be used for the first
pseudolabel generation loop.

For each RSNA slice, average model probabilities per pixel. The initial
pseudolabel target should be:

```text
p_any = mean(model_i_any_probability)
```

Subtype probabilities can also be averaged and stored for later subtype
partitioning:

```text
p_subtype[c] = mean(model_i_subtype_c_probability)
```

## Slice-Label Gating

Use the existing slice-wise classification labels as hard constraints:

```text
if any == 0:
    pseudomask_any = 0
    pseudomask_subtypes = 0

if any == 1:
    pseudomask_any comes from the ensemble any probability

for each subtype c:
    if label_c == 0:
        pseudomask_c = 0
```

For slices with exactly one positive subtype, assign the `any` mask to that
subtype:

```text
if exactly one subtype label is positive:
    pseudomask_that_subtype = pseudomask_any
```

For slices with more than one positive subtype, partition or weight the `any`
mask using subtype probabilities only among label-allowed classes:

```text
allowed = {c: label_c == 1}
subtype_weight[c] = p_subtype[c] / sum(p_subtype[k] for k in allowed)
pseudomask_c = pseudomask_any * subtype_weight[c]
```

Hard subtype masks, if needed for visualization, can use `argmax` over allowed
subtype probabilities inside the `any` mask.

## Threshold Selection

Use out-of-fold BHSD predictions to calibrate the `any` threshold.

1. Generate OOF `any` probability maps for every BHSD slice.
2. Sweep thresholds from low to high, for example `0.01, 0.02, ..., 0.99`.
3. For each threshold, binarize:

```text
pred_any_hard = p_any >= threshold
```

4. Compute Dice against the true BHSD `any` mask.
5. Choose the threshold that maximizes Dice.

Track both:

- slice-level Dice threshold, excluding empty/empty slice-class cases; and
- volume-level Dice threshold over full BHSD series.

Default starting choice: use the slice-level Dice-optimal threshold for visual
slice localization. If masks are too speckled, compare against the volume-level
threshold and simple connected-component filtering.

## Soft Pseudolabel Rescaling

Prefer soft pseudolabels for training rather than only hard masks. Use the
Dice-optimal threshold as the calibrated midpoint.

Initial calibration:

```text
logit_threshold = logit(threshold_any)
p_any_soft = sigmoid((logit(p_any) - logit_threshold) / temperature)
```

Starting temperature:

```text
temperature = 1.0
```

This makes `threshold_any` map to `0.5`, preserves uncertainty below and above
the threshold, and avoids immediate saturation from `p / threshold` style
rescaling.

Also save an inspectable hard mask:

```text
p_any_hard = p_any >= threshold_any
```

## Low-Confidence Positive Slices

If the classification label says `any == 1` but the segmentation ensemble does
not localize a convincing region, do not force an empty mask.

Compute simple confidence features:

```text
max_prob = max(p_any)
topk_mean = mean(top 0.1% pixels of p_any)
hard_area = count(p_any >= threshold_any)
soft_mass = sum(p_any_soft)
```

Use OOF BHSD predictions to choose conservative cutoffs for these features.

Recommended handling:

- confident positive slice: train with the soft pseudomask
- weak positive slice: ignore segmentation loss for that slice, or apply a very
  low segmentation loss weight
- `any == 0` slice: train confidently with an empty mask

This avoids teaching the segmentation model false empty masks on difficult
positive slices.

## Training With Pseudolabels

Treat BHSD masks and pseudolabels differently:

- BHSD ground truth masks: full segmentation loss weight
- confident pseudolabel masks: reduced segmentation loss weight
- weak positive pseudolabel slices: ignored or heavily downweighted
- classification-negative slices: empty masks with normal or moderately reduced
  weight

Keep metadata columns identifying mask source and confidence:

```text
mask_source = bhsd_gt | pseudolabel | empty_from_classification | ignored_positive
segmentation_weight = float
pseudolabel_confidence = float
```

## Outputs To Save

For each pseudolabeled slice, save:

- calibrated soft `any` mask
- hard `any` mask
- optional subtype soft masks
- optional subtype hard/color mask for visualization
- threshold and temperature used
- ensemble summary features such as max probability, top-k mean, hard area, and
  soft mass
- source model run IDs/checkpoints used for reproducibility

## Open Checks Before Implementation

- Confirm final BHSD CV model quality, especially `volume_dice_any` and
  `volume_hd95_any`.
- Decide whether slice-level or volume-level threshold gives better visual masks.
- Inspect OOF BHSD calibration plots for `any` probabilities over positive and
  negative pixels.
- Decide pseudolabel storage format after estimating disk size for the full RSNA
  training set.

## Deployment And Iterative Training Roadmap

In deployment, the classifier/sequence head should decide whether a slice is
positive. The segmentation branch is then used as the radiologist-facing
localization heatmap for slices that the classifier flags. Segmentation is
therefore an explanatory/localization output, not the primary detector.

Target final model structure:

```text
CT slices -> shared encoder -> per-slice feature maps
                         -> segmentation decoder -> heatmap
                         -> sequence head -> refined slice and series probabilities
```

The first joint classifier-segmenter experiment did not improve classification:
its best slice-level `auc_any` was 0.9831 versus 0.9844 for the original
classification-only model. Classification performance should therefore be
preserved by treating the classification encoder as fixed while optimizing the
localization decoder.

The next experiment asks whether a second mask-pseudolabeling stage improves
segmentation. Use the same 9-channel input, DeepLabV3+ decoder, patient-level
BHSD folds, loss, augmentation, batch size, schedule, epochs, and threshold
sweep for all primary comparisons.

During RSNA pseudolabel decoder pretraining, include every slice from a series
that contains at least one classification-positive slice, but exclude entirely
negative series. This retains negative boundary/context slices from positive
scans while avoiding an easy-negative majority that is not representative of
heatmap use behind the classification gate. The current training split contains
257,979 such slices, including 165,526 negative slices.

Use the complete 6,404-slice BHSD set for validation during this pretraining
stage, tracking slice- and volume-level Dice and HD95 with a threshold sweep.
This intentionally leaks the later BHSD target domain into model selection and
must not be reported as unbiased generalization. The controlled five-fold BHSD
OOF comparison below remains the evidence used to decide whether pseudolabel
pretraining helped.

| Condition | Encoder initialization | Decoder initialization | BHSD encoder policy | Purpose |
| --- | --- | --- | --- | --- |
| Direct BHSD baseline | Classification checkpoint | Random | Frozen | Best one-stage path that cannot perturb classification |
| Pseudolabel pretraining | Classification checkpoint | RSNA pseudolabel-trained | Frozen | Primary test of whether the second pseudolabel stage helps |
| Pseudolabel upper bound | Classification checkpoint | RSNA pseudolabel-trained | Fine-tuned at 0.1x decoder LR | Measures available segmentation gain if encoder drift is allowed |

The original frozen-encoder runs froze encoder gradients but did not keep the
encoder in evaluation mode. Consequently, BatchNorm running statistics changed
during both RSNA pseudolabel pretraining and BHSD training. The corrected
deployment experiment loads the pseudolabel-trained decoder and segmentation
head, overwrites the encoder parameters and buffers from the original
classification checkpoint, and keeps the encoder in evaluation mode throughout
BHSD training. This permits one canonical encoder pass to be shared by the
classifier and all five segmentation decoders at inference.

For the fine-tuned upper bound, use encoder LR `3e-5` and decoder/head LR
`3e-4`. For both frozen conditions, use decoder/head LR `3e-4`.

Compare five-fold BHSD OOF results using `any` hemorrhage as the primary target:

- volume Dice and HD95;
- slice Dice and HD95 with empty/empty slices excluded;
- threshold-swept operating points; and
- qualitative heatmap localization.

The second pseudolabel stage is worthwhile only if pseudolabel pretraining with
a frozen encoder improves the direct frozen-encoder BHSD baseline. The
fine-tuned condition is secondary: an improvement there quantifies a
segmentation/classification tradeoff but does not justify changing the shared
deployment encoder without separately confirming classification performance.

After selecting the segmentation path, freeze the shared encoder, extract
per-slice features, and train the sequence modeling head for refined slice- and
series-level classification. The classifier or sequence head controls whether a
heatmap is displayed; the segmentation decoder supplies localization only.

### Final second-stage segmentation result

The deployable experiment restores the canonical classifier encoder after
loading the pseudolabel-trained model, freezes its parameters, and keeps all
encoder modules in evaluation mode. Every encoder parameter and buffer in all
five final checkpoints was verified tensor-for-tensor against the classifier.

The corrected fold checkpoint monitor values for volume Dice `any` were
`0.6163`, `0.6176`, `0.6109`, `0.6275`, and `0.7072` (mean `0.6359`). A separate
evaluation of the released `last.ckpt` files gave mean volume Dice-any `0.6333`,
slice Dice-any `0.5288`, volume HD95-any `34.85 mm`, and slice HD95-any
`96.17 mm`. Dice used a per-fold threshold sweep from 0.1 to 0.9; the quoted
HD95 values use threshold 0.5, and empty/empty slice pairs are excluded from
the slice metrics.

The prior `0.6875` frozen-gradient volume Dice summary mixed pre-training sanity
validation with trained epochs and must not be used as a checkpoint result.
The old run also allowed BatchNorm state to drift despite frozen gradients. The
corrected frozen-state result is the only result used for the deployable
five-decoder ensemble. It remains a biased estimate: the pseudolabel teacher
ensemble was derived from BHSD folds, creating an indirect information path
back to BHSD validation patients.

Do not run another pseudolabel-generation cycle with the improved student under
the current evaluation design. A further cycle would compound this indirect
leakage and teacher confirmation bias. Reconsider iterative self-training only
after obtaining an untouched external segmentation test set or implementing a
strict fold-specific teacher pipeline.

## Frozen-Feature Sequence Classification

Use the final 9-channel classification-only `last.ckpt` as the frozen feature
extractor. Extract the pooled 1,280-dimensional feature before classifier
dropout. Store the original classifier logits for baseline evaluation and
validation-only blending, but do not expose them to the sequence model during
training because in-sample training logits are optimistically accurate.

The fixed split series-length distribution is:

- median: 33-34 slices;
- 95th percentile: 44 slices;
- 99th percentile: 52 slices; and
- maximum: 60 slices.

Use a padded sequence length of 64. For a future series longer than 64, sample
indices with `round(linspace(0, length - 1, 64))` and nearest-neighbor restore
the resulting predictions to the original slice indices. Padding must be
excluded from the GRU, attention softmax, losses, and metrics.

Sequence architecture:

```text
frozen pooled slice feature + normalized slice position
                         |
             Linear(1280, 512) + LayerNorm
                         |
          2-layer bidirectional GRU (256/direction)
                         |
          +--------------+----------------+
          |                               |
   contextual slice head       class-specific attention pool
      6 slice logits                 6 series logits

projected frozen slice features
                         |
       class-specific MIL attention pool
                         |
                 6 series logits
```

Train on the natural series distribution with ordinary shuffling and no
positive/negative resampling. Use normalized position because physical z
spacing is unavailable. Initial feature augmentation is Gaussian noise with
standard deviation 0.02, elementwise dropout 0.05, and whole-slice feature
dropout 0.02. Keep sequence reversal disabled initially because extraction and
deployment share a reliable orientation/order.

Loss weights:

- contextual slice BCE: 1.0;
- contextual series BCE: 0.5;
- order-independent MIL series BCE: 0.25; and
- class weights `[1, 1, 1, 1, 1, 2]` for all three objectives.

Average slice BCE within each series before averaging across the batch so long
series do not dominate. Report contextual slice, contextual series, MIL series,
baseline slice, and baseline max-pooled series AUROC separately. Select the
primary checkpoint using contextual slice `auc_any`; choose any inference blend
with the untouched baseline logits using validation data only.

### Sequence sweep result

A deterministic 15-condition sweep varied auxiliary loss weights, feature
noise, GRU depth, hidden size, and learning rate. The two strongest
architectures were repeated with seeds 89 and 90 in addition to seed 88. The
test split remained untouched throughout model and blend selection.

The selected architecture keeps the original two-layer bidirectional GRU,
256 hidden units per direction, feature noise 0.02, and learning rate `3e-4`.
Only the objective weights change:

```text
slice BCE  = 1.00
series BCE = 0.25
MIL BCE    = 0.10
```

Across three seeds, this architecture achieved:

| Validation metric | Mean | Seed SD |
| --- | ---: | ---: |
| Contextual slice AUC any | 0.987063 | 0.000134 |
| Blended slice mean AUC | 0.986182 | 0.000015 |
| Blended series AUC any | 0.987150 | 0.000040 |
| Blended series mean AUC | 0.978668 | 0.000040 |

The original classifier baselines are slice AUC any `0.984356`, slice mean AUC
`0.984038`, series max AUC any `0.986715`, and series max mean AUC `0.977403`.
The selected seed-88 contextual slice head improves AUC any to `0.987150`.
Patient-cluster bootstrap comparison against the baseline gives a mean delta of
`+0.002814` with 95% CI `[+0.001764, +0.003984]`.

Standalone contextual subtype performance is unstable for rare epidural
hemorrhage. Use validation-selected per-class logit blending with the frozen
classifier rather than replacing it outright. Class order is epidural,
intraparenchymal, intraventricular, subarachnoid, subdural, any:

```text
slice alpha  = [0.00, 0.75, 0.75, 1.00, 0.75, 1.00]
series alpha = [0.00, 0.50, 0.50, 1.00, 0.50, 0.50]
```

For each class, compute:

```text
final_logit = (1 - alpha) * baseline_logit + alpha * contextual_logit
```

Blend logits, not sigmoid probabilities. The fixed slice coefficients mean:

| Class | Baseline weight | Contextual weight | Interpretation |
| --- | ---: | ---: | --- |
| Epidural | 1.00 | 0.00 | Original classifier only |
| Intraparenchymal | 0.25 | 0.75 | Mostly contextual |
| Intraventricular | 0.25 | 0.75 | Mostly contextual |
| Subarachnoid | 0.00 | 1.00 | Contextual only |
| Subdural | 0.25 | 0.75 | Mostly contextual |
| Any | 0.00 | 1.00 | Contextual only |

For the series blend, first calculate the baseline series probability for each
class as the maximum original classifier probability across valid slices. Clamp
that probability to `[1e-6, 1 - 1e-6]`, convert it back to a logit, and blend it
with the class-specific attention-pooled GRU series logit:

| Class | Baseline-max weight | Contextual-series weight |
| --- | ---: | ---: |
| Epidural | 1.00 | 0.00 |
| Intraparenchymal | 0.50 | 0.50 |
| Intraventricular | 0.50 | 0.50 |
| Subarachnoid | 0.00 | 1.00 |
| Subdural | 0.50 | 0.50 |
| Any | 0.50 | 0.50 |

These coefficients were selected once on validation from
`[0.00, 0.25, 0.50, 0.75, 1.00]` independently for each class and were frozen
before the holdout test evaluation.

The slice blend reaches mean AUC `0.986176`. Its bootstrap mean improvement is
`+0.002145`, with 95% CI `[+0.001559, +0.002795]`. The series blend reaches AUC
any `0.987112` and mean AUC `0.978710`; its mean-AUC improvement is significant,
while its AUC-any interval still includes no change. Retain baseline max pooling
as a defensible series-level fallback until test evaluation.

The MIL head remains useful as an auxiliary training objective and diagnostic,
but the contextual series blend is the preferred series output. The selected
training and blend settings are captured in
`rsna_ich_effv2m_sequence_bigru_selected.py`.

### Complete selected sequence specification

The following settings are authoritative for reproducing the selected run:

| Component | Setting |
| --- | --- |
| Source classifier | Final 9-channel classification-only `last.ckpt` |
| Source input | Three adjacent slices, three CT windows per slice, flattened to 9 channels |
| Extracted feature | 1,280-dimensional global-average-pooled encoder feature before classifier dropout |
| Stored dtype | Features and baseline logits `float16`; labels `uint8` |
| Series order | Numeric slice filename order within patient/study/series |
| Maximum sequence | 64; observed dataset maximum is 60 |
| Longer-series policy | `round(linspace(0, length - 1, 64))`, then nearest-neighbor restore predictions |
| Padding | Zero padded; excluded from GRU packing, attention, loss, and metrics |
| Position | Original slice index divided by `max(original_length - 1, 1)`; no physical spacing |
| Input normalization | LayerNorm over the 1,280-dimensional frozen feature |
| Projection | Linear `1280 -> 512`, GELU, then LayerNorm |
| Feature augmentation | Gaussian noise SD `0.02`, elementwise dropout `0.05`, whole-slice feature dropout `0.02` |
| Sequence reversal | Disabled (`p=0`) |
| Sequence model | Two-layer bidirectional GRU, hidden size 256 per direction |
| GRU dropout | `0.20` between GRU layers |
| Context feature | 512 dimensions (256 forward + 256 backward) |
| Slice head | Dropout `0.20`, then Linear `512 -> 6` |
| Series head | Class-specific attention, attention hidden size 128, then six class-specific linear classifiers |
| MIL head | Class-specific attention over projected pre-GRU features, attention hidden size 128 |
| Labels | Epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any |
| Class weights | `[1, 1, 1, 1, 1, 2]` for slice, series, and MIL BCE |
| Objective weights | Slice `1.00`, contextual series `0.25`, MIL series `0.10` |
| Slice reduction | Mean over valid slices within each series, then mean across series |
| Training distribution | Natural series distribution; no positive/negative resampling |
| Batch size | 128 series/GPU on two GPUs; effective batch size 256 |
| Validation batch | 256 series/GPU |
| Optimizer | AdamW, LR `3e-4`, weight decay `1e-2` |
| Scheduler | Linear warmup for 5% of optimizer steps from 0, cosine decay to `1e-6` |
| Epochs | 20 |
| Precision/strategy | `bf16-mixed`, two-GPU DDP |
| Seed | 88; architecture repeated with seeds 89 and 90 |
| Checkpoint metric | Maximize contextual slice `auc_any` |
| Slice inference | Fixed per-class logit blend shown above |
| Series inference | Fixed per-class blend of baseline max-probability logit and contextual attention logit |
| MIL inference | Diagnostic only; not the preferred final series output |

### One-time held-out test result

After freezing the checkpoint and all blend coefficients above, evaluate once
on the patient-separated holdout test split. Do not use these results for
further tuning.

| Test metric | Baseline | Contextual | Fixed blend |
| --- | ---: | ---: | ---: |
| Slice AUC any | 0.979408 | 0.982529 | 0.982529 |
| Slice mean AUC | 0.975795 | 0.971997 | 0.978236 |
| Series AUC any | 0.979935 | 0.978807 | 0.980084 |
| Series mean AUC | 0.958150 | 0.960182 | 0.960560 |

The patient-cluster bootstrap confirms the primary slice result:

- contextual/blended slice AUC-any delta: `+0.003131`, 95% CI
  `[+0.001868, +0.004403]`;
- blended slice mean-AUC delta: `+0.002453`, 95% CI
  `[+0.001752, +0.003208]`; and
- blended series mean-AUC delta: `+0.002418`, 95% CI
  `[+0.001401, +0.003478]`.

The blended series AUC-any delta is only `+0.000180`, with 95% CI
`[-0.000663, +0.001048]`. Sequence context is therefore clearly beneficial for
slice detection and average subtype discrimination, but it has not established
a series-level `any` improvement over max pooling. Keep the fixed blend for the
selected deployment policy and retain baseline max pooling as a transparent
series-`any` reference/fallback.

### MaxViT backbone and contextual-head comparison

Train `maxvit_tiny_tf_512.in1k` with the same 9-channel inputs, fixed patient
split, loss, augmentations, optimizer, schedule, batch size, and three-epoch
protocol as the EfficientNetV2-M classifier. Its final validation slice
`auc_any` is `0.985481`, compared with `0.984356` for EfficientNetV2-M. An equal
logit blend reaches slice `auc_any = 0.986111` and six-class mean AUC
`0.986667`, providing validation evidence that the two backbones have
complementary errors. These results do not justify another evaluation on the
already observed holdout test.

For contextual comparison, extract MaxViT pooled features from train and
validation only using the final classifier `last.ckpt`. Store 512-dimensional
features and six baseline logits as `float16`; labels remain `uint8`. Do not
extract holdout-test features. Run the same 19-condition BiGRU matrix used for
EfficientNetV2-M, covering auxiliary objective weights, feature noise, one- or
two-layer GRUs, hidden width, learning rate, and seeds 88-90. All unchanged
sequence settings, including maximum length 64, projection width 512, natural
series distribution, 20 epochs, two-GPU DDP, and per-class validation-only
logit blending, remain matched. Select and compare using validation only.

The MaxViT sequence pipeline writes features under
`data/features/rsna_ich_maxvit_tiny_9ch`, checkpoints under
`experiments/rsna_ich_maxvit_tiny_sequence_bigru`, and analysis under
`logs/maxvit_sequence_sweep`. Compare standalone baseline, contextual, and
blended slice and series metrics against their same-backbone baselines and the
selected EfficientNetV2-M BiGRU. Hold off on MaxViT segmentation training until
the classification and contextual results justify its additional deployment
cost.

## Potential Further Work

The patient-separated holdout test has now been evaluated. Changes below may be
developed and selected with training/validation data, but claims of additional
generalization improvement require a new external or prospectively collected
test set.

### High Priority

1. **Probability calibration.** Apply temperature scaling after the fixed
   logit blend. Fit slice and series temperatures separately. Start with one
   temperature for `any` and one shared subtype temperature because epidural is
   too rare for stable independent series calibration. Compare identity versus
   scaling using five patient-level cross-fitting folds within validation and
   report NLL, Brier score, and ECE. Refit the accepted calibrator on all
   validation patients and freeze it before deployment.

2. **Operating-point selection.** After calibration, choose slice and series
   thresholds for the intended workflow. Report sensitivity, specificity,
   positive predictive value, negative predictive value, and the fraction of
   slices that trigger a displayed localization heatmap. Thresholds should
   reflect the higher cost of missing hemorrhage rather than defaulting to 0.5.

3. **Integrated inference model.** Package the frozen encoder, original
   per-slice classification head, selected BiGRU, fixed blend coefficients, and
   DeepLabV3+ localization decoder into one inference path. Verify that cached
   feature inference and integrated inference produce numerically equivalent
   outputs, including edge-slice zero filling, padding masks, series ordering,
   and source-slice mapping.

4. **External validation.** Evaluate classification, calibration, and
   localization on a dataset that was not involved in classifier training,
   pseudolabel generation, calibration, blending, or model selection. Include
   scanner/site and clinically relevant subgroup analyses where metadata
   permits.

### Sequence Modeling

The image encoder used for sequence modeling is already a true frozen-state
encoder: feature extraction loads the original classification checkpoint,
calls `eval()`, and runs under `torch.inference_mode()`. Sequence training uses
only the resulting fixed 1,280-dimensional arrays. Therefore any number of
sequence heads can share the same single encoder pass at deployment.

Run a validation-only architecture/ensemble comparison using the selected loss
weights and all other established settings. Compare three seeds each of the
two-layer BiGRU, a two-layer BiLSTM with the same hidden width, and a two-layer
Transformer encoder with model width 512, eight attention heads, feed-forward
width 1,024, pre-norm, GELU, and dropout 0.2. Average logits when ensembling.
Evaluate each individual head, each three-seed architecture ensemble, and the
nine-head cross-architecture ensemble. Select using only the fixed validation
split; do not inspect the existing heldout test again. Any claimed improvement
requires a new external or prospective test set.

Record validation regressions as well as improvements. For every candidate,
report raw contextual and blended slice and series AUC for all six labels,
`any`, and the six-class mean; deltas versus the original classifier; deltas
versus the selected seed-88 BiGRU; the number of classes that regress; and the
worst per-class delta. Raw contextual regressions must remain visible because
validation-optimized blending includes classifier-only (`alpha=0`) and can
therefore hide a weak contextual class by construction. Review seed consistency
and all regression columns before freezing a candidate. Do not automatically
evaluate the winner on the existing test set.
That test has already been observed for the current model, so strict unbiased
confirmation of a replacement requires new external or prospective data.

1. **Contiguous truncation augmentation.** During training only, truncate about
   30% of series to a contiguous 60-100% of their original length, with at
   least 16 retained slices. Crop features, labels, baseline logits, and source
   indices together while preserving positions normalized to the original
   series. Positive-series crops should retain at least one positive slice so
   the original series label remains valid. Validate on complete series.

2. **Incomplete-series behavior.** Define whether deployment must support
   partial studies. If so, test prefixes, suffixes, and internal contiguous
   subsets explicitly. A one-element sequence is technically accepted by the
   GRU but is outside the intended full-series distribution and should not be
   treated as equivalent to the base per-slice classifier.

3. **Long thin-slice series.** The present maximum is 60 slices, so the 64-step
   limit never truncates current data. For future series longer than 64,
   compare uniform resampling plus nearest-neighbor prediction restoration with
   overlapping sequence windows. Sliding windows may better preserve very
   small hemorrhages that fall between uniformly sampled slices.

4. **Physical position metadata.** If DICOM z-position and slice spacing become
   available, compare physical coordinates and inter-slice distances with the
   current normalized ordinal position. Preserve robust behavior when metadata
   is absent or unreliable.

5. **Series-`any` aggregation.** The contextual blend improves mean series AUC
   but has not significantly improved series `any` over baseline max pooling.
   Future external-data experiments may compare calibrated max, top-k mean,
   log-sum-exp, noisy-OR, and attention pooling. Baseline max remains the
   current reference/fallback.

6. **Three-way series blending.** A baseline + BiGRU + MIL blend is technically
   possible but is low priority. The baseline + BiGRU blend already slightly
   outperformed baseline + MIL, and another set of class-specific coefficients
   would add validation-selection flexibility for likely marginal benefit.

### Labels And Evaluation

1. **Extra-axial diagnostic metric.** Report an auxiliary epidural-or-subdural
   metric to distinguish missed extra-axial hemorrhage from subtype confusion.
   Epidural remains clinically important, but its very low prevalence makes
   standalone fold and seed comparisons unstable.

2. **Patient-cluster uncertainty.** Continue reporting patient-level paired
   bootstrap intervals rather than treating correlated slices as independent.
   For future model comparisons, repeat the strongest architectures across
   several training seeds before choosing among differences smaller than about
   `0.0002` AUC.

3. **Attention review.** Inspect class-specific BiGRU and MIL attention against
   positive slice locations. Attention is not itself a segmentation map, but it
   can reveal endpoint reliance, diffuse weighting, or failure to focus on the
   hemorrhage-containing portion of a series.

### Segmentation And Pseudolabeling

1. **Independent segmentation evaluation.** The current BHSD comparison has an
   indirect information path through the BHSD-derived pseudolabel teachers.
   Confirm the frozen-encoder decoder benefit on untouched pixel-level masks.

2. **Leakage-safe pseudolabel experiment.** If no external masks are available,
   a rigorous but expensive alternative is fold-specific pseudolabeling: for
   BHSD fold `k`, generate pseudolabels only from a teacher that excluded fold
   `k`, pretrain a fold-specific decoder, and evaluate on fold `k`.

3. **Do not iterate pseudolabeling without independent evaluation.** A third
   teacher/student mask cycle could amplify teacher errors and compound the
   current indirect leakage. Reconsider it only after establishing an untouched
   segmentation endpoint.

4. **Localization workflow validation.** Evaluate whether classifier-gated
   heatmaps improve radiologist localization and confidence, including false
   localization on classifier-positive slices. Segmentation Dice alone does not
   establish clinical usefulness for an explanatory overlay.

Implementation preferences:

- Keep the `any` mask as the primary segmentation target; subtype segmentation
  should be auxiliary and lower weight.
- Consider a segmentation-aware sampler in joint RSNA training if positive
  masks are too sparse per batch. Validation should remain on the natural
  distribution.
- Use BHSD OOF/validation Dice, HD95, and qualitative review as the gate for
  each pseudolabel generation loop to avoid reinforcing confident errors.
- Keep sequence modeling classification-only at first. Let the sequence head
  suppress or promote heatmap display, but keep the heatmap spatially tied to
  the per-slice encoder/decoder output.
- Preserve reusable artifacts: encoder checkpoints, decoder checkpoints, OOF
  BHSD predictions, RSNA pseudolabel manifests, thresholds, and extracted
  feature arrays.
