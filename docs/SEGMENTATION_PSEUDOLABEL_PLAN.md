# Segmentation teachers and auxiliary classification supervision

Discussion recorded 2026-11-10. Classification performance remains the primary
objective, particularly small-hemorrhage sensitivity without excessive false
positives. Train segmentation teachers on BHSD, then explore their pseudo-masks
as auxiliary supervision for classification on the larger non-BHSD dataset.
The sections below distinguish agreed decisions, measured results, and proposals.

## Agreed data boundaries

- BHSD supplies manual masks for training segmentation teachers.
- Exclude BHSD from direct downstream student training, as in the original
  classification experiments. Do not mix or oversample real BHSD masks into
  student batches. BHSD still influences the student indirectly through teachers.
- Most slices lack segmentation masks, but slice-level classification labels are
  available. Known-negative slices have valid empty-mask targets, subject to
  slice-label reliability. The relevant annotation deficit is chiefly positive
  slices without segmentation masks.
- Classification validation/test studies must stay out of student training.
- BHSD teacher assessment uses patient-separated folds. Any teacher-ROI audit
  on BHSD must use out-of-fold predictions, never a model trained on that study.

## Teacher training decisions

The initial model is ConvNeXt-Tiny initialized strictly from the final hemorrhage
classifier trained with stride 16 and spatial logit max pooling. Preserve its
nine-channel input, CT windows, normalization, and 512 x 512 resolution. A U-Net
predicts a binary any-hemorrhage mask for the center slice; ignore subtypes in
this segmentation loss. Its skips accommodate the encoder's repeated stride 16
resolution. Encoder peak LR is 3e-5; decoder peak LR is 3e-4, retaining the 10 x ratio.

Use classification-like augmentation with aligned image/mask geometry; omit
cutout that would obscure a bleed while preserving its segmentation target.
Loss starts as equally weighted soft Dice and BCE. Planned sampling arms are
100% positive,75/25 positive/negative, and 50/50, with slices cycling without
replacement within their separate bins. Planned Dice arms aggregate per slice
or over the batch. These broader ablations are deferred pending schedule results.

Run five-fold CV from the outset. The initial schedule study uses positives only,
per-slice Dice, batch 32 on a single GPU per fold, and independent 10/25/50-pass
budgets. One pass visits every positive training slice once, retaining the final
partial batch. Two independent folds run concurrently. Warmup is 5% of actual
optimizer updates, followed by cosine decay; decoder endpoint 1e-6, encoder 1e-7.
Each budget starts from the same classifier initialization with its own schedule.

Primary metric and checkpoint ranking: mean hard Dice on ground-truth-positive
slices at threshold 0.5. Also sweep 0.1–0.9 in 0.1 steps, recording best positive-slice
Dice and its threshold. Secondary: mean per-volume Dice at 0.5 and at that same
selected threshold. Volume metrics include predictions on negative slices.
Save both best and most-recent checkpoints/predictions. Lead comparisons with
final checkpoints; best-checkpoint results are secondary context.

**Schedule review decision:** use 25-pass results to decide whether 50 is warranted.
If 25 shows no meaningful final-checkpoint improvement over 10, or deterioration
with falling training loss, skip 50. Continued validation improvement supports
trying 50. This is a decision rule, not a claim that 25 is already too long.
**Updated decision:** Ian cancelled all 50-pass runs. Fold 0 had just started
and was terminated; folds 1–4 were cancelled before launch. The remaining
25-pass fold 4 continues unchanged. The original controller is stopped to prevent
its in-memory queue launching further jobs. The observer service
`ich-bhsd-finish25-20261110` updates completion and retires that controller only
after preserved training exits. Queue records distinguish cancellation from
completed training; partial artifacts are retained.

Full implementation and artifact locations:
[BHSD schedule study](BHSD_CONVNEXT_SCHEDULE_STUDY.md).

## Pseudo-mask construction proposals

No downstream pseudo-label recipe has been selected or launched.

For non-BHSD positive training slices, consider averaging probability maps from
five final teacher checkpoints and retaining inter-teacher disagreement. Keep
unmodified soft predictions available so target thresholds/sharpening can be
changed without repeating inference. Known-negative slices receive empty masks.
For BHSD audits use only each study's held-out teacher, not the five-fold ensemble
that would include teachers trained on the study.

Candidate ablations, changing one factor at a time:

| Factor | Reference proposal | Alternative |
|---|---|---|
| Target values | Unmodified mean probabilities | Temperature sharpening, initially T=0.5 |
| Confidence | Uniform weighting | Gradual disagreement weighting with a nonzero floor |
| Uncertainty handling | Keep soft targets | Ignore highly uncertain pixels |
| Loss | Soft-target BCE | BCE plus per-slice soft Dice |
| Pixel balance | Mean BCE | Region-balanced BCE |

Sharpening q uses sigmoid(logit(q)/T), with numerical clipping if implemented.
Sharpening around 0.5 suppresses weak positive targets as well as hardening strong
ones. It does not create confidence. Teacher disagreement differs from entropy
of the mean prediction; teacher agreement does not establish correctness.
Hard exclusion may preferentially remove subtle bleeds and uncertain boundaries.

Proposed safeguard: when a known-positive slice has no credible teacher
foreground, keep classification supervision and omit or reduce auxiliary
segmentation supervision. Do not silently turn it into an empty target or force
foreground by selecting top-scoring pixels. The credibility criterion remains
undecided and must not systematically discard small bleeds unnoticed.

Consider series-balanced positive sampling so large/long positive studies do not
dominate. Positive/negative sampling and auxiliary-loss weights remain controlled
experimental choices. Real-mask oversampling was explicitly rejected.

## Background imbalance and region-balanced BCE

Mean pixel BCE can be dominated by background even on positive slices. Tiny soft
probabilities over a large image can also accumulate substantial apparent
foreground mass. Avoid unchecked inverse-frequency weights derived directly
from summed soft probabilities.

One proposal is L = alpha * mean(BCE inside ROI) + (1-alpha) *
mean(BCE outside ROI), initially testing alpha 0.5. Targets remain soft in both
regions; ROI membership controls weight, not target value. Empty regions need
explicit handling if implemented. Per-slice soft Dice is a complementary
foreground-sensitive term. A region containing mostly normal tissue does not
fully solve foreground/background imbalance.

### Teacher-defined ROI proposal

Threshold teacher probabilities liberally (illustrative 0.1–0.2, not selected),
optionally dilate, and allocate additional loss weight to that candidate region.
This emphasizes predicted bleeds, boundaries and nearby background. Retain
soft-target supervision outside the region.

Concern: the teacher determines both the target and where supervision is
concentrated. A low threshold/dilation can recover uncertain boundaries but
cannot reliably recover a disconnected bleed the teacher misses. Ian considers
this worth testing, contingent on a ground-truth coverage audit.

### HU-defined ROI proposal

Define a region from original center-slice HU before CT windowing/intensity
augmentation. Apply the same geometric transformation to the weighting region
as to images and targets. Keep supervision outside the band. HU supplies a
teacher-independent weighting region, but teacher errors still enter through
pseudo-targets. Do not use an HU band as a hard definition of hemorrhage.

## Measured BHSD HU coverage

Annotation-only audit:192 studies,191 patients,6404 slices,2368 positive slices.
Original reoriented NIfTI HU and manual masks; no teacher predictions. Bands are
inclusive. Region-size denominators are full 512 x 512 images, with extracranial
tissue and air; there is no brain mask. Pooled hemorrhage-pixel coverage:

| HU band | Hemorrhage covered | All-image area | Positive-image area |
|---|---:|---:|---:|
| 0–100 |99.68%|18.67%|24.73%|
| 20–100 |94.65%|15.97%|21.61%|
| 30–100 |87.30%|10.95%|14.43%|
| 35–100 |80.70%|7.48%|9.37%|
| 40–100 |71.65%|5.03%|5.90%|
| 50–100 |49.14%|2.58%|2.83%|

Small-bleed coverage, macro-averaged within each group:

| HU band | Components <=0.1 mL (n 555) | Studies <=1 mL total bleed (n 21) |
|---|---:|---:|
| 0–100 |98.19%|100.00%|
| 20–100 |96.25%|95.91%|
| 30–100 |86.61%|77.27%|
| 40–100 |60.20%|49.96%|

Components are 26-connected binary 3D regions, a proxy rather than adjudicated
individual lesions. At 40–100 HU,18.9% of the smallest components have zero
coverage; at 20–100 this is 2.7%. Subdural pixel coverage is 79.3% at 20–100 versus
99.0% at 0–100. Raising the upper limit from 100 to 120 with lower limit 20 adds
about 0.1 percentage point of overall hemorrhage coverage.

Interpretation:0–100 is a proposed first weighting band, preserving nearly all
annotated hemorrhage for a modest area increase over 20–100. It is not yet a
selected training setting. Even on positive slices, about 96.5% of pixels within
this region are background. Whole-BHSD results are descriptive and should not be
presented as independent held-out validation of a band selected using them.

Reproduction: [audit script](../scripts/analyze_bhsd_hu_coverage.py).
Local artifacts: [band results](../diagnostics/bhsd_hu_coverage_20261110/bands.csv),
[size strata](../diagnostics/bhsd_hu_coverage_20261110/size_strata.csv),
[subtypes](../diagnostics/bhsd_hu_coverage_20261110/subtypes.csv), and
[provenance](../diagnostics/bhsd_hu_coverage_20261110/provenance.json).

## Agreed follow-up teacher ROI audit

After final teacher predictions are available, repeat the annotation coverage
audit using out-of-fold BHSD predictions. Sweep probability thresholds and
modest dilation, measuring hemorrhage coverage, ROI area on all/positive slices,
small-component coverage, complete misses, and secondary subtype breakdowns.
Compare teacher versus HU regions at matched ROI area. Also examine their union.
Dilation radius and threshold grid remain to be specified. Audit final checkpoints;
do not substitute best checkpoints silently. No teacher ROI audit has run yet.

## Proposed auxiliary classification experiments

Ian proposes retraining classification with a segmentation decoder branch and
possibly direct supervision of the spatial classification-logit map. These
branches have different roles: the decoder encourages detailed encoder features;
map supervision directly constrains the activations used by max pooling and may
reduce isolated false-positive activations. These benefits remain hypotheses.

Proposed controlled arms: classification only; plus decoder; plus map loss;
plus both. Retain six slice-level classification targets. Binary segmentation
would supervise the any-hemorrhage channel only. Use the same initialization,
sampling, update budget, classification loss and evaluation cohort across arms.
A general objective is classification loss + lambda_decoder * decoder loss +
lambda_map * map loss. Auxiliary weights are undecided; start modestly and check
for competition with classification gradients.

For 32 x 32 map targets, test max pooling teacher probabilities over corresponding
16 x 16 image regions. This represents presence somewhere in a cell and avoids
average-pooling away a tiny bleed. It can also propagate isolated teacher false
positives. Spatial cells have receptive fields larger than their nominal blocks,
so supervision is coarse localization, not pixel-exact boundaries. Downsampling
method, confidence handling and map-loss balance remain proposals.

Known-negative slices can supervise every any-hemorrhage map cell as negative.
Known-positive teacher misses should not override the reliable positive slice
label. The decoder can be removed at inference; direct map supervision adds no
inference branch. Judge success on final-checkpoint series AUC and small-bleed
sensitivity at matched specificity, with calibration and segmentation quality
as secondary outcomes. Improved Dice alone does not establish classification gain.

## Three-family soft-label export (2026-11-10)

Launched `scripts/run_ich_teacher_pseudolabel_export.py` on both GPUs under user
systemd unit `ich-teacher-pseudolabel-export-20261110`. Uses all 15 final
25-pass positive-only batch-Dice teachers: ConvNeXt-Tiny, MaxViT-Tiny and
CoAtNet-0, five folds each. EffNet is excluded. Outputs live at
`data/pseudolabels/rsna_three_family_final_20261110`; logs and worker status at
`logs/ich_teacher_pseudolabel_export_20261110`. Completion is established by
`completion.json`, not by this launch record.

Targets are 91,744 known-positive slices without manual masks, from 7,579
training series / 6,690 patients. The original classification exclusion was
study-level: an additional 709 positive slices in other studies from 29
BHSD-related patients are explicitly excluded here. Validation/test patients
are excluded. The manifest retains original adjacent-slice context and original
classification preprocessing (nine channels, 512 x 512).

Only the equal-weight mean probability and population standard deviation
across the 15 teachers are stored, as compressed float16 512 x 512 arrays.
Individual teacher predictions are not saved. Uncompressed size is about
45 GiB per field, 90 GiB total; the 256-slice preflight compressed to 151 MiB
for both fields together (whole-cohort compression may differ). No sharpening
or hard threshold is applied. Low-score predictions on known-positive slices
remain soft predictions, not confirmed-empty targets. Negatives have implicit
empty masks and are not exported. Shards are atomic and resumable, with
checkpoint/source hashes and export signatures recorded in provenance.json.

Preflight passed all 15 checkpoint loads, numerical mean/std checks, rank
partition coverage, and exact saved-array roundtrip. Measured 256 slices in
41 seconds on one GPU; initial two-GPU estimate approximately two hours.

## BHSD OOF uncertainty audit (2026-11-10)

Completed CPU-only analysis with `scripts/analyze_bhsd_uncertainty.py`, outputs
`diagnostics/bhsd_uncertainty_20261110`. Uses the three final held-out teachers
per case (one per family), not all 15: the other folds trained on these patients.
Analyzes all 2,368 positive slices; 471 have <=262 annotated positive pixels
(<=0.1% of image area). This small-area definition is slice-based, not a
physical lesion-volume or connected-component definition.

Compared population standard deviation, entropy of mean probability, and
mutual information = entropy(mean) minus mean(entropy). Metrics use pooled
pixel counts and 2,048-bin uncertainty histograms with fractional ties;
fold-level results are also saved. Retention is ranked independently within
each named stratum. Foreground/background ground-truth strata are diagnostic,
not deployable selection rules. Thresholds 0.5 and 0.6, retention 100/95/90/75/50%.

Within predicted foreground at threshold 0.5, retaining the least-uncertain
90% gives false-positive fractions 23.28% (std), 21.92% (entropy), 23.39% (MI),
versus 26.57% unfiltered. Corresponding retained true-positive pixels are
94.04%, 95.70%, 93.90%; small-slice true-positive retention 87.38%, 88.76%,
87.30%. Entropy has lower retained false-positive fraction in all five folds.
At threshold 0.6 the same comparison favors entropy: 20.59%, 19.98%, 20.72%.
Within predicted foreground, entropy ranking is simply increasing confidence
in hemorrhage probability; this is equivalent to raising the probability cutoff,
not independent information from ensemble diversity.

Global masking is concerning: removing the most-uncertain 5% of all positive-
slice pixels removes 28.49% (std), 24.22% (entropy), 30.83% (MI) of annotated
hemorrhage pixels. Small-slice hemorrhage retention is only 43.96%, 50.23%,
41.49%. Error rates alone reward removing difficult true-positive regions.
These descriptive results support testing gentle region-specific weighting;
they do not demonstrate that uncertainty weighting improves student training.
No compelling reason here to replace std storage with MI. Current export
unchanged; entropy is recoverable from saved means. Three-teacher OOF uncertainty
does not exactly reproduce the 15-teacher export distribution.

## Authorized auxiliary classification runs (2026-11-10)

Ian requested monitoring the export to completion and then relaunching all four
stride-16 spatial-logit-pooling classifiers with both auxiliary branches.
Final requested weights: **0.5 each** (supersedes the initially proposed 0.1
and briefly discussed 1.0). Objective is weighted six-class classification BCE
+ 0.5 * (binary decoder BCE + batch Dice) + 0.5 * binary spatial-map BCE.
Only the any-hemorrhage logit channel receives map supervision. Targets are
raw soft three-family ensemble means; known-negative masks are empty. No
temperature sharpening, uncertainty masking, or additional thresholding.
Spatial targets are adaptive-max-pooled to 32 x 32 to retain small positives.
Dice is reduced over each actual microbatch, not the accumulated effective batch.

ConvNeXt-Tiny, MaxViT-Tiny, EffNetV2-M and CoAtNet-0 use the original ImageNet
pretrained initialization and a new five-block U-Net decoder (256/128/64/32/16).
They are new runs, not resumptions/fine-tuning of the final classifier checkpoints.
Input remains nine channels at 512 x 512; output stride 16 and six classification
targets are retained. Classification augmentations are preserved with masks
transformed alongside images, bilinear interpolation for soft masks, and empty
mask fill in cutout regions. Depth flips preserve the center-slice target.
Decoder inference is optional via return_segmentation=True; ordinary validation
and classification export bypass the decoder.

Training excludes all BHSD-related patients: 658,995 slices, 91,744 positives,
19,041 series. This is a stricter cohort than the earlier study-level exclusion,
so comparisons to old baselines are not a perfectly isolated loss ablation.
The odd number of series cannot split exactly over two DDP ranks without repeats.
Run one model per GPU, two concurrently, with common microbatch selected by
GPU preflight (32, else 16, else 8) and accumulation (2, 4, 8) for effective
batch 64. Each epoch visits every series once with independent positive/negative
without-replacement bins and positive-bin probability 0.5. All variants have
298 optimizer steps/epoch, 14,900 steps over 50 epochs, 745 warmup steps,
AdamW LR 3e-4 and cosine floor 1e-6. Runtime schedule assertions remain enabled.
Validation every five epochs, primary any-hemorrhage series max AUC, unchanged
classification loss/metrics, best AND final prediction export. Auxiliary loss
components are logged separately during training; validation has no pseudo masks.

Handoff script: scripts/run_ich_auxiliary_after_export.py. Status/log directory:
logs/ich_auxiliary_20261110. User systemd unit: ich-auxiliary-after-export-20261110.
Requires verified export completion and exporter exit, prepares a verified
45 GiB float16 memory-mapped probability cache, then runs GPU memory/gradient
preflights for every family and checks real data/target coverage. It refuses
duplicate run directories and stops launching queued jobs on failure.
CPU architecture/gradient/inference checks passed for all four families; 25
tests passed covering soft-mask alignment, small-target pooling, schedule,
sampling and handoff behavior. GPU/data preflights are deferred until export
completion so current inference remains uninterrupted. Each family uses a
separate local MLflow database to avoid concurrent SQLite writers.

### Subsequent corrections and current hold (2026-11-11)

The export completed: 91,744 slices, 359 verified shards. The initial auxiliary
launch failed before training because single-GPU auto strategy retained the
trainer's default SyncBatchNorm. After correcting that flag, Ian requested the
original classifier settings. The retry service was stopped during initialization;
no automatic restart is armed. Status is recorded in logs/ich_auxiliary_20261110.

The original saved classifier contract is two GPUs, DDP + SyncBatchNorm,
batch 32 per GPU, no accumulation, 19,102 series / 661,118 training slices,
50 epochs, 299 optimizer steps/epoch, 14,950 total steps and 748 warmup steps.
Restore the original cohort, six workers and other classifier hyperparameters;
retain only the two auxiliary branches and their 0.5 loss weights as intended
model changes. This supersedes the single-GPU/stricter-cohort plan above.

Ian explicitly accepts the BHSD-related overlap and wants those slices included
in auxiliary supervision too. The 709 additional positive slices still need
predictions from the same 15-teacher ensemble before launch; they must not receive
empty targets or be excluded from auxiliary losses. Original BHSD study exclusion
in the classifier remains the baseline; this refers to other included studies
from BHSD-related patients. Ian then requested exploring low-threshold teacher
ROIs first, so training stays stopped pending that analysis/choice. The prepared
training code still requires completing this supplemental-mask integration
before it is ready to launch.

## Low-threshold teacher ROI audit (2026-11-11)

Completed `scripts/analyze_bhsd_teacher_roi.py --initial` and the default refined
sweep. Artifacts: diagnostics/bhsd_teacher_roi_20261111 and
diagnostics/bhsd_teacher_roi_refined_20261111; the latter contains
combined_summary.csv. Evaluated 192 series, all 6,404 slices / 2,368 positive
slices, using saved mean probabilities of the three final held-out teachers per
case. Teacher thresholds .001/.005/.01/.02/.05/.1/.2 and
.055/.06/.065/.07/.075/.08/.09, each with elliptical in-plane dilation radius
0/3/5 pixels, plus union with inclusive 0–100 HU. This is a descriptive ROI
coverage study, not a student training experiment or independent selection test.

Low thresholds are ineffective: <=.01 selects the whole image, and .05 selects
99.88% of positive-slice image area. Negative-slice spot checks show ensemble
median probabilities .057–.064 across folds. Verified ensemble maps exactly equal
the float16-stored mean of individual final OOF predictions on these samples.
The HU reference reproduces the independent annotation-only audit exactly.
Histogram threshold inclusivity, grouped coverage, dilation equivalence and
coverage/area monotonicity checks passed.

Representative results (area on positive slices; coverage pooled over annotated
hemorrhage pixels; small-component coverage macro-averaged over 555 annotated
26-connected 3D components <=0.1 mL):

| ROI | Area | Hemorrhage coverage | Small-component coverage |
| --- | ---: | ---: | ---: |
| Teacher >=.06, no dilation | 76.33% | 99.11% | 95.62% |
| Teacher >=.065, no dilation | 23.21% | 93.73% | 77.62% |
| Teacher >=.07, radius 3 | 7.09% | 93.98% | 75.69% |
| HU 0–100 | 24.73% | 99.68% | 98.19% |
| HU union teacher >=.07, radius 3 | 29.45% | 99.87% | 99.23% |

At .065 without dilation, teacher ROI completely misses 20 positive slices and
84 small components; .07/radius3 misses 29 slices and 120 small components.
HU alone misses no positive slices and nine small components; the .07/radius3
union misses no positive slices and three small components. At .065 the ROI
area varies from 7.4% to 39.0% across folds, indicating threshold sensitivity to
teacher probability offsets. Physical component labels are not adjudicated lesions.

Recommendation for discussion: avoid teacher-only hard loss masking. HU plus
low-threshold teacher union is a more coverage-preserving candidate; teacher-only
regions could instead define emphasis with nonzero outside weight. No ROI has
yet been applied to training, and preserving annotated tissue in an ROI does not
correct teacher segmentation errors. Production uses 15 teachers rather than
three OOF teachers, so exact threshold behavior may differ. Training remains
stopped pending choice and completion of the 709 supplemental masks.

## Approved ConvNeXt unrestricted versus HU/teacher-union runs (2026-11-11)

Ian approved two ConvNeXt-Tiny stride-16 runs only, replacing the pending four-
family queue: unrestricted auxiliary supervision versus HU/teacher-union region
restriction. The original classifier cohort and settings are retained exactly:
661,118 training slices, 92,453 positives, 19,102 series, 50 epochs, two-GPU DDP
with SyncBatchNorm, batch 32/rank, accumulation 1, seed 88, ImageNet initialization,
AdamW 3e-4 / WD .01, 14,950 updates with 748 warmup updates and cosine floor 1e-6,
same six weighted classification targets, max spatial-logit pooling, validation
every five epochs, primary any-hemorrhage series max AUC. Config comparison to
the saved original config.pkl passed, as did exact sample image/label augmentation
parity with the original dataset. Auxiliary weights remain 0.5 each.

All 709 previously excluded BHSD-related training-positive masks are now exported
with the same 15 final teachers and verified in
data/pseudolabels/rsna_three_family_bhsd_related_20261111. They are included in
auxiliary training; there are no classification-only positives. The main export
and supplemental cache together cover all original training positives.

Unrestricted objective: classification BCE + 0.5*(decoder BCE + batch Dice)
+ 0.5*spatial any-logit BCE. Restricted objective uses the same formula, with
both auxiliary losses restricted to HU in [0,100] OR teacher probability >=.07
dilated by an elliptical kernel of radius 3 pixels. ROI is constructed before
augmentation; masks/ROI follow spatial augmentation and flips together. Soft
mask interpolation is linear; ROI is re-binarized at .5 after transformation.
Cutout removes ROI supervision in erased regions for the restricted arm;
unrestricted supervision includes zero-filled target cutout regions. HU is
recovered through thresholds within the unclipped subdural window of the original
center-slice processing, with inclusive-boundary tests. Known-negative slices
have empty targets and use HU-only ROI. BCE is normalized over included pixels;
batch Dice sums only included pixels. Empty ROI contributes zero auxiliary loss.
Map targets use max pooling of ROI-masked soft targets; map eligibility uses
max pooling of ROI, so a cell is supervised if any corresponding pixel is eligible.
Classification supervision always remains unrestricted. No sharpening or
uncertainty weighting is applied.

Queue: scripts/run_ich_convnext_roi_ablation.py; logs/ich_convnext_roi_20261111;
user systemd unit ich-convnext-roi-ablation-20261111. Sequential arms:
rsna_ich_convnext_tiny_s16_auxiliary (unrestricted), then
rsna_ich_convnext_tiny_s16_auxiliary_roi. Both use two GPUs. The launcher verifies
the complete caches, actual data, and GPU forward/backward geometry before
training. It stops on failure. Twenty-nine tests passed, including actual
trainer CLI validation, ROI gradients, HU thresholds and pointing geometry.

After each run: save best AND final classification validation predictions, then
evaluate the final checkpoint on all 6,404 BHSD slices / 192 series using
scripts/evaluate_ich_auxiliary_bhsd.py. Save full-resolution decoder probabilities,
32x32 any-logit maps, class logits, threshold counts and per-slice pointing data
under the run's bhsd_final directory. Report positive-slice Dice at .5 and best
of .1–.9 (with threshold), raw volume Dice and known-negative-gated volume Dice,
and negative-slice false positives. Pointing primary = maximum of bilinearly
upsampled any-logit map falls inside manual mask; also report coarse-cell mask
overlap and distance within 5 mm, plus small-positive-slice area <=10 mm2.
Ties take first raster pixel. No ROI restriction/gating is applied to primary
decoder or pointing predictions. Ian explicitly accepts leakage: these are
exploratory segmentation/localization diagnostics, not formal held-out estimates.

### Memory failure, testing, and explicit training hold

The unrestricted run was killed by systemd-oomd during its first validation.
The recoverable checkpoint is epoch index 3 / optimizer and scheduler step 1196
(four completed passes). Diagnostic loader probes found 17.7 GiB maximum
observed per-rank process-tree PSS with pinning versus 7.2 GiB without pinning;
40 validation batches took 22.2 versus 25.5 seconds. These are bounded inference
probes, not complete training-cycle measurements. Results are saved under
diagnostics/ich_auxiliary_memory_20261111.

Ian initially approved the no-pinning validation/recovery recommendation, then
explicitly instructed **do not resume training until told to do so**. The automatic
handoff service was stopped during preflight, before any training resumed.
Subsequent clarification authorized continuing tests only. Standalone service
ich-validation-only-nopin-20261111 performs full two-rank validation of the
existing checkpoint via scripts/validate_ich_auxiliary_recovery.py. It contains
no training or automatic handoff. Both training arms remain on hold regardless
of validation outcome. Do not invoke scripts/resume_ich_convnext_roi_nopin.py
without new explicit authorization.

Ian subsequently proposed positive-slice-only decoder Dice. Implemented batch
Dice over slices whose real any-hemorrhage label is positive, preserving the ROI
restriction where applicable. Negative slices still supervise decoder BCE and
spatial-map BCE; all-negative batches have zero Dice loss. Soft teacher scores
do not determine slice eligibility. Auxiliary weights remain 0.5. Memory pinning
is now explicitly disabled in the shared auxiliary config. Tests cover exclusion
of negative-slice Dice gradients, retention of their BCE gradients, and empty
positive batches. Neither training run was started. The old checkpoint's four
epochs used the earlier loss; clean ablation comparisons should restart both
arms consistently rather than silently change objective mid-run.

Ian explicitly authorized restarting both arms fresh with the positive-only
Dice recipe. New controller: scripts/run_ich_convnext_roi_positive_dice.py;
service ich-convnext-roi-positive-dice-20261111; logs at
logs/ich_convnext_roi_positive_dice_20261111. New run IDs contain aux05_posdice;
the interrupted checkpoint is preserved and is not resumed. Pinning is disabled,
both auxiliary weights are 0.5, and original 50-epoch DDP settings are retained.
Unrestricted trains first, then HU/teacher union, with the existing best/final
classification exports and exploratory final-checkpoint BHSD evaluations.
Thirty targeted tests passed before launch. This authorization supersedes the
training hold for these two fresh runs only.

### 2026-11-11: soft ROI weight 0.25 ablation
User authorized one fresh ConvNeXt-Tiny S16 run with outside-ROI weight 0.25.
Config: `rsna_ich_convnext_tiny_s16_auxiliary_roi_w025`; service:
`ich-convnext-roi-w025-20261111`; logs: `logs/ich_convnext_roi_w025_20261111`.
Same original classification initialization, cohort, series sampling, 50 epochs,
batch 32/GPU on two GPUs, optimizer and LR schedule as completed ROI/unrestricted arms.
Pixel weights are 1 inside HU-teacher union and 0.25 outside. Decoder BCE is
normalized by sum of weights; positive-only batch Dice uses the same weights.
Spatial BCE uses max-pooled pixel weights and unattenuated max-pooled teacher
probabilities. Original hard-ROI behavior remains unchanged with default weight 0;
weight 1 reproduces unrestricted loss. Both auxiliary coefficients remain 0.5.
31 targeted tests passed, including outside-background gradients, target preservation,
normalization and unrestricted endpoint parity. Controller performs data/GPU preflight,
then training, best/final classification exports, and final BHSD evaluation.

### 2026-11-11: matched backbone ablation of weight-0.25 recipe
User selected completed ConvNeXt-Tiny weight-0.25 recipe and authorized three
fresh runs: MaxViT-Tiny, CoAtNet-0, EffNetV2-M, in that order. Controller:
`scripts/run_ich_backbones_roi_w025.py`; service `ich-backbones-roi-w025-20261111`.
Configs inherit their previous stride-16 spatial-logit backbone configurations,
then apply HU-teacher union and outside weight 0.25. Full configuration comparison
against ConvNeXt found only backbone identity and required backbone adapter/image-size
flags differ. Sampling, 50 epochs, seed 88, batch 32/GPU, two GPUs, no accumulation,
optimizer/LR schedule, preprocessing, augmentations, positive-only weighted Dice,
BCE weight normalization, both auxiliary coefficients 0.5, pin_memory=False,
and evaluation cohort/cadence remain identical. Decoder channel adaptation is
required by backbone features. Each family uses its original ImageNet pretrained
weights; pretrained checkpoints differ by family, as in earlier backbone ablations.
30 targeted tests passed. Controller runs full data preflight plus batch-32 GPU
forward/backward/optimizer preflight for all three families before training.
Each run exports best and final classification predictions and final BHSD masks/maps.
ConvNeXt is reused, not retrained. No additional ablations are queued.
