---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
- medical
- radiology
- ct
- intracranial-hemorrhage
- image-classification
- image-segmentation
- pytorch
- timm
datasets:
- rsna-intracranial-hemorrhage-detection
metrics:
- roc_auc
---

# CT Head Hemorrhage Detection

Research model for slice- and series-level intracranial hemorrhage detection and
slice-level localization on noncontrast head CT. The complete system combines:

1. a 9-channel MaxViT-Tiny slice encoder/classifier;
2. a two-layer bidirectional GRU for contextual slice and series predictions;
3. five DeepLabV3+ decoders sharing that exact encoder state for
   radiologist-facing localization.

This is **research software, not a medical device**. It must not be used to
diagnose, exclude, triage, or manage intracranial hemorrhage without independent
clinical validation and appropriate regulatory review.

## Model Details

| Component | Specification |
| --- | --- |
| Slice encoder | `maxvit_tiny_tf_512.in1k`, ImageNet initialization |
| Slice input | Three adjacent axial slices centered on the predicted slice |
| Channels | Brain, subdural, and bone windows for each slice, flattened to 9 channels |
| Slice outputs | Epidural, intraparenchymal, intraventricular, subarachnoid, subdural, and any hemorrhage |
| Sequence head | Two-layer bidirectional GRU, 256 hidden units/direction, class-specific attention |
| Sequence outputs | Context-refined slice probabilities and attention-pooled series probabilities |
| Localization | Five-fold DeepLabV3+ decoder ensemble sharing the permanently frozen classification encoder |
| Inference execution | One encoder pass per slab, followed by the classifier/BiGRU branches and five lightweight decoder passes |
| Input size | 512 x 512 pixels |

Class order is always:

```text
epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
```

The segmentation output is intended as a localization heatmap after the
classifier has flagged a slice. It is not intended for hemorrhage volume
measurement.

```text
9-channel slab
      |
      v
shared MaxViT-Tiny encoder (one pass)
      |
      +-- multiscale feature maps --> decoder fold 0 --+
      |                            --> decoder fold 1   |
      |                            --> decoder fold 2   +--> mean heatmap
      |                            --> decoder fold 3   |
      |                            --> decoder fold 4 --+
      |
      +-- deepest map --> classifier head/pool --> 512-D pooled feature
                                                   |               |
                                                   v               v
                                           linear slice head   BiGRU sequence head
```

## Input Processing

The model expects three correctly ordered adjacent slices. At the beginning or
end of a series, the missing neighboring slice is replaced by a zero-valued
windowed slice.

1. Read each source PNG without reducing its 16-bit depth.
2. Normalize spatial size to 512 x 512. If either source dimension exceeds
   640 pixels, resize the complete field of view with area interpolation.
   Otherwise, center-crop or zero-pad.
3. Convert stored values to Hounsfield units:

   `HU = stored_value * rescale_slope + rescale_intercept`

4. Apply three linear CT windows independently:

| Window | Center | Width | HU interval |
| --- | ---: | ---: | ---: |
| Brain | 40 | 80 | 0 to 80 |
| Subdural | 80 | 200 | -20 to 180 |
| Bone | 600 | 2800 | -800 to 2000 |

5. Clip each window and scale it to `[0, 1]`.
6. Flatten in **slice-major order**:

   `[previous brain, previous subdural, previous bone, center brain, ...,
   next bone]`

7. The model linearly normalizes `[0, 1]` to `[-1, 1]`.

Slices were ordered by the first numeric token in each slice filename within
each patient/study/series directory. Physical z-spacing was not available to
the sequence model.

## Data And Splits

Classification used the
[RSNA Intracranial Hemorrhage Detection challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
dataset converted to lossless 16-bit PNG. Splits were made at the **patient level** so no patient
appeared in more than one partition. The separate BHSD pixel-annotated subset
was excluded from classification training, validation, and test.

| Partition | Patients | Studies/series | Slices |
| --- | ---: | ---: | ---: |
| Training | 16,910 | 19,102 | 661,118 |
| Validation | 887 | 1,000 | 34,865 |
| Heldout test | 989 | 1,450 | 50,415 |
| Excluded BHSD subset | 191 | Not reported | 6,404 |

The intended 1,000-study test sample was expanded to all studies belonging to
the selected patients, yielding 1,450 heldout studies. This prevents leakage
and avoids discarding studies from those patients.

The holdout was first evaluated for the earlier EfficientNetV2-M release. The
MaxViT architecture, checkpoint, and blend coefficients were subsequently
selected using validation only, then evaluated here to document the replacement
release. The test set is therefore consumed and these MaxViT results are a
post-selection benchmark, not a pristine one-shot estimate. It must not be used
for further tuning.

## Training

### Slice classifier

The classifier was trained for three epochs on two GPUs with batch size 32 per
GPU, `bf16-mixed` precision, AdamW (`lr=3e-4`, `weight_decay=1e-2`), 5% linear
warmup from zero, and cosine decay to `1e-6`. The multilabel BCE class weights
were `[1, 1, 1, 1, 1, 2]`, giving the `any` target twice the weight of each
subtype.

For the 9-channel stem, timm repeated the three ImageNet input-channel kernels
three times and multiplied them by `3/9`, preserving the aggregate activation
scale at initialization.

Spatial flips were independently applied with probability 0.25 for depth,
horizontal, and vertical axes. One transform was then sampled uniformly from:
no-op, affine, brightness/contrast, Gaussian or motion blur, Gaussian noise,
and coarse dropout.

### Sequence model

The frozen classifier produced one 512-dimensional pooled feature per slice.
The sequence model used the natural class distribution, a maximum length of 64,
normalized slice position, a `512 -> 512` projection, and a two-layer
bidirectional GRU with 256 hidden units per direction. Training-time feature
augmentation used Gaussian noise SD 0.02, elementwise dropout 0.05, and
whole-slice feature dropout 0.02.

The sequence model was trained for 20 epochs on two GPUs with effective batch
size 256 series, AdamW (`lr=3e-4`, `weight_decay=1e-2`), and the same warmup and
cosine schedule. Loss weights were slice BCE 1.00, contextual series BCE 0.25,
and auxiliary MIL series BCE 0.10. All objectives used class weights
`[1, 1, 1, 1, 1, 2]`. Seed 88 was selected after a deterministic sweep and the
architecture was repeated with seeds 89 and 90.

For sequences longer than 64, inference samples
`round(linspace(0, length - 1, 64))` and restores predictions to original slice
indices with nearest-neighbor assignment. The observed development-data maximum
was 60 slices.

### Segmentation model

The localization branch was developed on BHSD with five patient-separated
folds. It predicts six multilabel maps for the middle slice of the 3-slice
input. The final model uses a DeepLabV3+ decoder and per-sample DiceFocalLoss.
Training used batch size 32/GPU, AdamW, decoder LR `3e-4`, ten epochs, and a
threshold sweep from 0.1 through 0.9.

The selected path initialized the decoder from an RSNA pseudomask-training stage
and then trained five BHSD folds while freezing the classification encoder.
Encoder parameters and buffers were restored from the canonical classifier,
kept in evaluation mode throughout training, and verified tensor-for-tensor
against the classifier checkpoint after training. The five validation-selected
`best.ckpt` decoder outputs are ensembled by averaging sigmoid probabilities. See
**Segmentation Evaluation Caveat** below.

## Inference Blending

Blend logits, not probabilities. For each class:

```text
final_logit = (1 - alpha) * baseline_logit + alpha * contextual_logit
```

| Class | Slice alpha | Series alpha |
| --- | ---: | ---: |
| Epidural | 0.75 | 0.50 |
| Intraparenchymal | 0.75 | 0.25 |
| Intraventricular | 0.75 | 0.50 |
| Subarachnoid | 0.75 | 1.00 |
| Subdural | 0.75 | 0.50 |
| Any | 1.00 | 0.50 |

For series inference, the baseline is the maximum baseline probability over
valid slices, clipped to `[1e-6, 1 - 1e-6]` and converted back to a logit before
blending with the contextual attention logit. Coefficients were independently
selected from `[0, 0.25, 0.5, 0.75, 1]` on validation and frozen before test.

The auxiliary MIL output is retained as a diagnostic but is not part of the
selected final blend.

## Heldout Test Performance

The test set contains 50,415 slices from 1,450 series belonging to 989 patients.
Positive prevalence was 15.65% by slice and 44.97% by series for `any`
hemorrhage. AUROC is threshold-independent and does not imply calibration or a
clinically acceptable operating point.

### Slice AUROC

| Class | Baseline classifier | Contextual BiGRU | Fixed blend |
| --- | ---: | ---: | ---: |
| Epidural | 0.954060 | 0.923096 | 0.947909 |
| Intraparenchymal | 0.986440 | 0.986407 | 0.987003 |
| Intraventricular | 0.994089 | 0.990694 | 0.993089 |
| Subarachnoid | 0.971642 | 0.973272 | 0.974216 |
| Subdural | 0.968003 | 0.971891 | 0.972305 |
| Any | 0.980826 | 0.982764 | 0.982764 |
| Macro mean | 0.975843 | 0.971354 | 0.976214 |

### Series AUROC

| Class | Baseline max | Contextual attention | Fixed blend |
| --- | ---: | ---: | ---: |
| Epidural | 0.926251 | 0.909021 | 0.926238 |
| Intraparenchymal | 0.968643 | 0.969831 | 0.969060 |
| Intraventricular | 0.987429 | 0.984272 | 0.987287 |
| Subarachnoid | 0.950067 | 0.954478 | 0.954478 |
| Subdural | 0.957891 | 0.960066 | 0.960236 |
| Any | 0.979037 | 0.979871 | 0.980402 |
| Macro mean | 0.961553 | 0.959590 | 0.962950 |

Patient-cluster bootstrap comparisons against the baseline gave:

- fixed-blend slice `any` AUROC delta `+0.001952`, 95% CI
  `[+0.001283, +0.002693]`;
- fixed-blend slice macro-AUROC delta `+0.000292`, 95% CI
  `[-0.001714, +0.001782]`;
- fixed-blend series `any` AUROC delta `+0.001347`, 95% CI
  `[+0.000164, +0.003135]`;
- fixed-blend series macro-AUROC delta `+0.001432`, 95% CI
  `[-0.000152, +0.003001]`.

The data support improved `any` discrimination from the fixed contextual blend.
Macro-average changes are less certain, and the raw contextual head regresses
epidural AUROC substantially. Validation selected a nonzero epidural blend for
this MaxViT release, so that regression remains visible rather than being
silently replaced with classifier-only output.

Epidural hemorrhage was rare: 264 positive test slices and 27 positive test
series. Its per-class estimates and blend coefficient are correspondingly less
stable than those of more prevalent labels.

## Segmentation Evaluation Caveat

The released corrected frozen-state run produced the following fold-averaged
`any` hemorrhage metrics:

| Released `best.ckpt` ensemble members | Value |
| --- | ---: |
| Volume Dice, threshold swept per fold | 0.6403 |
| Volume Dice at threshold 0.5 | 0.6362 |
| Slice Dice at threshold 0.5 | 0.5407 |
| Volume HD95 at threshold 0.5 | 34.37 mm |
| Slice HD95 at threshold 0.5 | 89.02 mm |

The checkpoint monitor's fold values were `0.6055`, `0.6326`, `0.6331`,
`0.6443`, and `0.6858` (mean `0.6403`). Empty/empty slice pairs were excluded
from slice Dice/HD95. Dice thresholds were swept from 0.1 through 0.9 within
each BHSD validation fold, while the fixed-threshold Dice and HD95 values above
are fold means at threshold 0.5.

An earlier internal summary reported volume Dice `0.6875` for the
frozen-gradient experiment. That value mixed pre-training sanity validation
with trained epochs and is not a valid final checkpoint estimate; it is not
used for this release.

**These are not unbiased generalization estimates.** The pseudolabel teacher
ensemble was itself derived from BHSD folds, creating an indirect information
path back to BHSD validation patients. The results support an optimization and
localization prior, not a claim of external segmentation accuracy. No untouched
segmentation test set was available.

## Artifact Layout

```text
config.json
classifier/model.safetensors
sequence/model.safetensors
segmentation/fold0.safetensors
...
segmentation/fold4.safetensors
inference.py
requirements.txt
```

The classifier file contains the full slice classifier and the canonical
encoder state. Each segmentation fold contains only its DeepLabV3+ decoder and
segmentation head; no encoder parameters or BatchNorm buffers are duplicated.
For each slab, integrated series inference normalizes once and runs the encoder
once. The deepest feature map passes through the classifier convolution, pool,
and linear head; the exact same pooled 512-dimensional feature is also passed
to the BiGRU. The cached multiscale feature pyramid fans out to all five resident
decoders. This is numerically equivalent to the independently executed
classifier, sequence head, and corrected frozen-state segmentation checkpoints
while avoiding redundant encoder passes.

The implementation, training configurations, frozen-state regression tests,
and validation-only sequence architecture comparison are available at source
revision
[`098b7ff`](https://github.com/i-pan/ct-head-hemorrhage-detection/tree/098b7ff).

## Slice Inference Example

Install `requirements.txt`, then load three original 16-bit PNG slices. Use
`None` for a missing neighbor at a series boundary.

```python
from inference import HemorrhageSliceModel, preprocess_slab, read_uint16_png

x = preprocess_slab(
    read_uint16_png("previous.png"),
    read_uint16_png("center.png"),
    read_uint16_png("next.png"),
    rescale_slope=1.0,
    rescale_intercept=-1024.0,
)

model = HemorrhageSliceModel()
output = model.predict(x)
class_probabilities = output["classification"]  # shape: (6,)
mask_probabilities = output["segmentation"]      # shape: (6, 512, 512)
```

This is standalone slice inference. For integrated contextual inference, stack
all ordered 9-channel slabs and call `predict_series`:

```python
series_output = model.predict_series(slabs, batch_size=4)

slice_probabilities = series_output["slice_classification"]
series_probabilities = series_output["series_classification"]
mask_probabilities = series_output["segmentation"]
```

`slabs` has shape `(number_of_slices, 9, 512, 512)`. `batch_size` controls how
many slabs are resident on the accelerator at once; each slab is still encoded
once regardless of batch size.

For a DICOM- or NIfTI-derived HU volume, avoid materializing the complete
9-channel slab array in host memory:

```python
series_output = model.predict_hu_volume(
    volume,                 # shape: (D, H, W), physically ordered, in HU
    batch_size=2,           # lower this if accelerator memory is limited
    return_segmentation=True,
)
```

This windows and transfers only one batch at a time. Returning all six
full-resolution segmentation maps still requires approximately 6 MiB of host
memory per source slice; set `return_segmentation=False` when only
classification is needed.

The same pooled features drive the standalone classifier and BiGRU, and the
same feature maps drive all five decoder folds. Series longer than 64 slices
use the training dataset's deterministic fallback by default: uniformly sample
64 positions, run the BiGRU, and restore contextual slice predictions by
nearest-neighbor assignment. The standalone classifier and all five decoders
still run on every source slice. `long_series_mode="full"` is available because
the BiGRU has no architectural length limit, but lengths above 64 were not
represented during development and full-length behavior has not been
validated. The standalone and contextual logits are combined using the fixed
class-specific blends documented above.

### DICOM series

The loader filters scouts and mixed-series files, sorts slices by projected
`ImagePositionPatient`, applies each slice's `RescaleSlope` and
`RescaleIntercept`, and returns a physically ordered HU volume. Missing rescale
metadata is treated as an error rather than silently interpreting stored pixels
as HU.

```python
from inference import preprocess_hu_volume_slab, read_dicom_series

series = read_dicom_series("dicom_series/")
center_index = len(series["image"]) // 2
x = preprocess_hu_volume_slab(series["image"], center_index)
output = model.predict(x)
```

Only single-frame CT series are supported. Multiframe DICOM, localizers, mixed
series, and duplicate slice positions are rejected or filtered. The returned
metadata includes physical spacing, orientation code, sorted source files,
excluded files, and per-slice rescale values.

### NIfTI volume

NIfTI voxel values must represent HU after header scaling. The loader applies
the NIfTI scale proxy, permutes/flips voxel axes to LPS, then returns a
DICOM-like `(inferior-to-superior, anterior-to-posterior, right-to-left)` volume
in `(D, H, W)` order.

```python
from inference import preprocess_hu_volume_slab, read_nifti_volume

volume = read_nifti_volume("head_ct.nii.gz")
center_index = len(volume) // 2
x = preprocess_hu_volume_slab(volume, center_index)
output = model.predict(x)
```

This is nearest-axis permutation/flip reorientation. It does not resample an
arbitrarily oblique acquisition into a true axial series. Review orientation
and HU values before inference when converting data from another system.

## Demo Scope

The linked static Space shows standalone slice-classifier probabilities and
segmentation heatmaps precomputed from heldout 3-slice slabs. It does not run the
BiGRU because a single 3-slice slab does not provide a complete CT series.
Therefore, Space classification corresponds to the **baseline** column, not the
contextual or fixed-blend column.

The included heldout examples are deidentified, subtype-isolated slices selected
for clear qualitative demonstration. They are not a random sample and must not
be interpreted as another performance evaluation.

## Limitations

- Retrospective development and testing used one public challenge dataset.
- No external, prospective, site-stratified, scanner-stratified, demographic,
  or clinical workflow validation was performed.
- Probabilities were not temperature-scaled and should not be interpreted as
  calibrated risks.
- No clinical sensitivity/specificity threshold has been selected.
- Series ordering uses filenames rather than physical z-coordinate or spacing.
- The model expects noncontrast axial head CT with the specified preprocessing;
  behavior on reformats, contrast CT, postoperative anatomy, severe artifact,
  pediatric patients, or other domains is unknown.
- The segmentation branch is affected by indirect BHSD leakage described above
  and should be interpreted only as a qualitative localization aid.
- Subtype labels can overlap, and epidural performance is especially uncertain
  because of low prevalence.
- The model may produce confident false negatives or false positives. A heatmap
  is not evidence that a prediction is correct.

## Ethical And Clinical Considerations

Outputs must be reviewed by qualified clinicians in the context of the complete
exam and clinical history. Do not use this model to delay image interpretation,
replace a radiologist, or exclude hemorrhage. Any deployment requires local data
governance, privacy review, cybersecurity controls, monitoring for domain drift,
failure analysis, calibration, threshold selection, and prospective validation.

## Citation

If this artifact is useful, cite the model repository and the original RSNA
Intracranial Hemorrhage Detection challenge. The software is released under the
MIT License; underlying datasets retain their own terms of use.
