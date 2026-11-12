# Alternate ICH release: exact model recipe and reproduction

The alternate model is the three final epoch-50 **raw** auxiliary classifiers:
ConvNeXt-Tiny, MaxViT-Tiny, and CoAtNet-0. Each has output stride 16, spatial-logit
max pooling, and a binary U-Net decoder. This is distinct from the
[prior MaxViT/BiGRU/DeepLab recipe](ICH_LEGACY_PIPELINE.md).

## Released training recipe

Inputs: three contiguous axial slices, brain/subdural/bone windows, nine channels,
512 × 512, center-slice six-label classification. Any hemorrhage has BCE weight 2;
the other five subtypes have weight 1. ImageNet initialization; all model weights
trainable. Missing neighbors are zero windowed channels.

Fifty epochs, one center slice per series per epoch. Positive series select the
positive or negative bin with probability 0.5 each, falling back to the nonempty
bin. Bins cycle independently through unsampled slices before reshuffling.
AdamW LR 3e-4, weight decay 0.01, 5% linear warmup then cosine to 1e-6, actual
optimizer-step schedule, batch 32/GPU on two GPUs without accumulation. Historical
runs used 299 steps/epoch and 14,950 updates. Each model uses its final checkpoint,
not the validation-selected best. BF16 mixed precision, seed 88.

Augmentations match classification: depth/horizontal/vertical flips at 0.25,
one of no-op, affine, brightness/contrast, blur, noise or coarse dropout. Spatial
mask transforms align with images; coarse dropout also zeros the pseudo-mask.

Fifteen final segmentation teachers generated mean-probability soft masks for
positive training slices (three families × five BHSD folds). Each teacher used
a classifier-initialized encoder, U-Net, 25 positive-slice passes, batch Dice+BCE,
and encoder LR one tenth decoder LR. Known-negative slices have empty masks.
Student real masks are not oversampled or mixed in. The teacher ensemble itself
is not distributed with the student release.

The student decoder uses BCE plus batch Dice, with Dice limited to positive
slices. The spatial any-logit map uses BCE against adaptive-max-pooled teacher
probabilities. Decoder loss and spatial loss each have coefficient 0.5. Pixels
inside HU [0, 100] OR teacher ≥ 0.07 dilated by a radius of 3 pixels have weight 1;
outside pixels have weight 0.25. Weighted losses normalize by the weight sum.

Exact configs:

- `src/skp/configs/rsna_ich_convnext_tiny_s16_auxiliary_roi_w025.py`
- `src/skp/configs/rsna_ich_maxvit_tiny_s16_auxiliary_roi_w025.py`
- `src/skp/configs/rsna_ich_coatnet0_s16_auxiliary_roi_w025.py`

Historical launchers: `scripts/run_ich_convnext_roi_w025.py` and
`scripts/run_ich_backbones_roi_w025.py`. These retain historical paths and counts;
use the portable route below for new work.

## Reproduction commands and historical differences

Follow [input acquisition and the complete staged pipeline](ICH_PIPELINE.md).
`run_ich_recipe.py` supplies classifiers → five-fold teachers → soft masks →
students. Use the same recipe with fresh outputs. New splitting excludes **all
BHSD patients**, including their other studies. Therefore new training does not
reproduce the historical cohort byte-for-byte or promise identical metrics.

Historical students excluded directly matching BHSD studies but included other
BHSD-patient studies; supplemental teacher masks covered their positive slices.
The portable clean route does not need that supplement. The old split also left
8 teacher patients in test and 1 in validation. The published evaluation must
retain this distinction and the fact that the test set was already consumed.

## Inference and measured comparison

Average the three slice logits equally, then normalize log-sum-exp over slices
with T = 0.25, then sigmoid. The measured endpoint is series-level **any hemorrhage**.
All six outputs now have retrospective slice/series evaluation, recorded in
`release/ich_alt/subtype_evaluation.json` and the HF model card. Subtype results
use the same fixed T = 0.25 without subtype-specific tuning. Optional segmentation averages three sigmoid maps without
gating. The binary decoder and coarse maps are for exploratory localization;
no external segmentation validation or ensemble Dice claim is made.

On the post-hoc teacher-disjoint subset (1,410 series, 981 patients), AUC = 0.981499,
Brier = 0.048828, log loss = 0.174175; at 0.5 sensitivity = 0.951846 and specificity = 0.918679.
These are not untouched holdout estimates. At the same three-model logits, max
pooling gives corrected-cohort AUC = 0.981277, Brier = 0.060390, log loss = 0.209873.
The probability-quality benefit of LSE transfers; the AUC difference is inconclusive.
No test-set fitting occurred. The later comparison is exploratory.

## Release mechanics

`release/ich_alt/export.py` exports only the approved final tensors and hashes.
`skp.ich_alt` provides strict member loading, subset downloads, slab preprocessing,
and ensemble prediction. The release includes neither optimizer resume states nor
teacher weights. Fine-tuning begins from a selected unfrozen member with a newly
constructed optimizer; use the full training pipeline for auxiliary objectives.
The author selected Apache-2.0 for the new weights and MIT for project code.
Dataset terms remain separate; see `release/ich_alt/NOTICE.md`.

## Published release

[Alternate HF model](https://huggingface.co/ianpan/ct-head-hemorrhage-detection-alt/tree/faac61d23d08f50b6c999ed2aea416d92890139b)
contains the three final raw checkpoints, installable minimal runtime, hashes and
verification report. See `release/ich_alt/receipt.json` for immutable revisions
and rollback references. Code/source revision is `cbab8c9778530ac7dd0ede90a597e3087987893d`.

All payload files were downloaded at the pinned HF revision and SHA-256 verified.
A remote-ID download and installed-wheel single-member GPU inference passed.
Local real-image checks matched all three original checkpoint outputs exactly;
140 repository tests passed. The wheel was installed into an isolated target
using existing dependencies; a full fresh dependency installation was not tested.
