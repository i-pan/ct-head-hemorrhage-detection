---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- medical
- ct
- intracranial-hemorrhage
- image-segmentation
- timm
metrics:
- roc_auc
---

# CT Head Hemorrhage Detection — Alternate Ensemble

Three final stride-16 classifiers (ConvNeXt-Tiny, MaxViT-Tiny, CoAtNet-0), each
with a binary localization decoder. Classification is primary; maps are exploratory.

**[Source](https://github.com/i-pan/ct-head-hemorrhage-detection/tree/a557df791f139862e266dd696405bda1dfcc247c)
· [Specific training reproduction](https://github.com/i-pan/ct-head-hemorrhage-detection/blob/a557df791f139862e266dd696405bda1dfcc247c/docs/ICH_ALT_RELEASE.md)
· [Training procedure](#training-procedure)
· [Original MaxViT/BiGRU model](https://huggingface.co/ianpan/ct-head-hemorrhage-detection)**

The original uses contextual BiGRU classification and five frozen-encoder
DeepLabV3+ decoders. This alternate uses mean slice logits and normalized LSE
T = 0.25, without a sequence model. Neither is established as clinically superior.
Research use only; not validated for diagnosis, triage, or treatment decisions.

## Installation and inference

Python 3.11; tested with PyTorch 2.6 and CUDA 12.4. Install the bundled wheel and dependencies:

```bash
pip install -r requirements.txt
pip install skp_ich_alt_runtime-0.1.0-py3-none-any.whl
```

```python
import numpy as np
from skp.ich_alt import ICHEnsemble, preprocess_slab

# Three correctly ordered axial stored-pixel images; None for missing neighbors.
x = preprocess_slab(previous, center, following,
                    rescale_slope=1.0, rescale_intercept=-1024.0)
model = ICHEnsemble("path/to/release", device="cuda:0")
# Supply ALL ordered slabs for the series-level endpoint.
out = model.predict_series(np.stack(ordered_slabs), batch_size=4,
                           return_segmentation=True)
```

For HU arrays use slope 1 and intercept 0. No DICOM parser is implied: apply correct
per-slice rescaling and orientation upstream. Do not interpret a single-slab
result as a full-series prediction. Input is N × 9 × 512 × 512 in [0, 1], slice-major
brain/subdural/bone windows (40/80, 80/200, 600/2800 center/width). Missing neighbors
are zeros after windowing. Spatial processing resizes the field of view if either
dimension exceeds 640; otherwise center-crops/pads stored pixels before rescaling.
Maps are in the processed 512 × 512 grid, not restored native patient coordinates.

Outputs:

- `slice_logits`, `slice_classification`: N × 6, class order epidural,
  intraparenchymal, intraventricular, subarachnoid, subdural, any.
- `series_classification`: six scores, with retrospective subtype evaluation below.
  Independent external performance and calibration remain unvalidated.
- Optional `segmentation`: N × 1 × 512 × 512 mean sigmoid probabilities, ungated.
- Optional `any_logit_map`: N × 1 × 32 × 32 mean coarse logits, not calibrated masks.

Classification averages member logits before pooling. Localization averages
probabilities; it has no classifier or ground-truth gate. No teacher models are
required at inference. All three architectures are used by default. Subsets select
architectures, not folds; full-ensemble metrics do not apply to subsets.

Retrieve the original immutable weight release (weights are unchanged by this evaluation update):

```python
from huggingface_hub import snapshot_download
folder = snapshot_download("ianpan/ct-head-hemorrhage-detection-alt", revision="af3fea32c9093495e39beccb727136e33afed767")
```

Single-member download avoids downloading the other weights:

```python
from skp.ich_alt import download_release, load_member
folder = download_release("af3fea32c9093495e39beccb727136e33afed767", families=("convnext_tiny",))
member = load_member(folder, "convnext_tiny", device="cuda:0")
member.train()  # parameters are unfrozen; construct a new optimizer
# For classification fine-tuning:
logits = member({"x": batch_in_0_1})["logits"]
# Six multilabel BCE targets, center slice; any class weight 2, other weights 1.
```

For decoder/spatial fine-tuning, use the linked training pipeline: it supplies
aligned soft segmentation targets, positive-slice indicators and ROI weights,
and installs the classification criterion. No optimizer/scheduler resume state is
included. The original classifier preprocessing and heads remain unchanged.

## Performance and limitations

Final raw epoch 50 weights, equal mean slice logits, normalized LSE T = 0.25, no
learned calibration or test fitting. Test set was already consumed by earlier
work. Eight BHSD teacher patients overlapped historical test membership; one
also overlapped validation. The corrected result is a **post-hoc teacher-disjoint
audit**, not a newly untouched holdout.

| Series any endpoint | Teacher-disjoint test subset |
| --- | ---: |
| Series | 1,410 |
| Patients | 981 |
| AUROC | 0.981499 |
| Brier score | 0.048828 |
| Log loss | 0.174175 |
| Sensitivity at 0.5 | 0.951846 |
| Specificity at 0.5 | 0.918679 |

An exploratory comparison on the corrected cohort found max pooling AUC 0.981277,
Brier 0.060390 and log loss 0.209873: LSE improves probability quality with essentially
unchanged discrimination. No temperature was fitted on test. External validation,
clinical utility, small-bleed subgroup performance and deployment latency are
not established here. No full-ensemble segmentation performance is claimed;
BHSD localization is affected by teacher reuse. Maps are not volume measurements.

### Subtype evaluation

Retrospective evaluation of the same frozen logits, with T = 0.25 applied to every
class. No subtype-specific fitting, temperature selection, or checkpoint selection.
Slice probabilities are sigmoid of the mean member logits; series targets are
positive if any constituent slice is positive. Labels are multilabel, not mutually
exclusive. Macro AUROC below averages all six outputs, including `any`.

**Teacher-disjoint subset: 48,972 slices, 1,410 series, 981 patients.**

| Label | Positive slices | Positive series | Slice AUROC | Series AUROC | Series Brier | Series log loss |
|---|---:|---:|---:|---:|---:|---:|
| epidural | 258 | 25 | 0.956383 | 0.933978 | 0.016076 | 0.061626 |
| intraparenchymal | 2,693 | 393 | 0.988461 | 0.973466 | 0.051395 | 0.182217 |
| intraventricular | 2,067 | 295 | 0.993839 | 0.987544 | 0.030723 | 0.112687 |
| subarachnoid | 2,549 | 280 | 0.970258 | 0.953025 | 0.063877 | 0.215483 |
| subdural | 3,303 | 261 | 0.970386 | 0.957264 | 0.060484 | 0.196364 |
| any | 7,601 | 623 | 0.981303 | 0.981499 | 0.048828 | 0.174175 |
| Macro AUROC (six outputs) | — | — | 0.976772 | 0.964463 | — | — |

Epidural has only 25 positive series in the teacher-disjoint subset, so its
estimate is less stable. These point estimates have no new confidence intervals
and do not establish external subtype calibration or clinical utility. All
previous test-consumption and teacher-overlap caveats still apply. Aggregate
metrics and source-prediction hashes are in [subtype_evaluation.json](subtype_evaluation.json).

## Training procedure

RSNA ICH classification, ImageNet initialization, 9-channel 512 × 512 inputs. Fifty
sampled-series epochs: one center per series, positive series independently
cycle positive/negative bins with 0.5 selection probability. Two GPUs,
batch 32/GPU, no accumulation, BF16, seed 88, AdamW LR 3e-4 / weight decay 0.01,
5% warmup then cosine to 1e-6; 14,950 historical optimizer steps. Six weighted BCE
targets (any weight 2). All encoder/decoder parameters trainable. Final raw states
are used, not best, EMA or SWA. Augmentations include flips, affine,
brightness/contrast, blur, noise and dropout, aligned with soft masks.

Fifteen final teachers (three families × five BHSD folds) produced mean soft masks
on positive slices; negatives have zero masks. Teachers use classifier encoders
with tenfold lower LR, U-Net, 25 positive passes and batch Dice + BCE. Student losses
add 0.5 × (decoder BCE + positive-only batch Dice) and 0.5 × spatial-map BCE. HU [0, 100]
OR teacher ≥ 0.07 dilated 3 px receives weight 1; other pixels weight 0.25. Coarse targets
use max pooling. No real-mask oversampling. Teacher weights are not in the release.

The [reproduction guide](https://github.com/i-pan/ct-head-hemorrhage-detection/blob/a557df791f139862e266dd696405bda1dfcc247c/docs/ICH_ALT_RELEASE.md)
links exact configs, historical launchers, and the complete acquisition→teacher→
pseudolabel→student route. New reproduction excludes all BHSD patients and derives
step counts from the resulting cohort. Historical runs excluded matched studies
and used supplemental masks for other studies of BHSD patients; consequently the
new clean recipe is not a promise of identical historical metrics.

## Licensing and provenance

See NOTICE.md. Code MIT; upstream pretrained backbones declare Apache-2.0.
Weights are licensed by the author under Apache-2.0. Training datasets retain
their own terms, including conflicting BHSD license statements; no dataset
redistribution or third-party rights are granted by this release. No patient data, teacher checkpoints, training caches or optimizer
states are included. `release.json` records source revision, checkpoint hashes,
state selection, exact file sizes and SHA-256. Contact/report issues through the
linked GitHub repository. HF hosting does not imply hosted inference support.
