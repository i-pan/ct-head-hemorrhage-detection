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

Planned training sequence:

1. Train a joint classifier-segmenter on RSNA with 9-channel inputs and
   pseudolabel masks. Classification remains the primary objective and checkpoint
   metric; segmentation is a localization regularizer.
2. Fine-tune or evaluate the resulting segmentation branch on BHSD to determine
   whether the joint RSNA training improves segmentation transfer.
3. Use the improved 9-channel segmentation models to regenerate RSNA
   pseudolabels if BHSD OOF/validation metrics and qualitative review improve.
4. Repeat the joint RSNA classifier-segmenter training with the improved
   pseudolabels.
5. Freeze the shared encoder, extract per-slice features, and train a sequence
   modeling head for refined slice-level predictions and series-level
   predictions.
6. Train a final segmentation decoder branch on BHSD with the encoder frozen, so
   heatmap quality can improve without disturbing the classifier/sequence
   representation.

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
