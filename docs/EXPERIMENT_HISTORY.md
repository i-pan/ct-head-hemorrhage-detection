# ICH experiment history

See [DECISION_LOG.md](DECISION_LOG.md) for rationale. Generated predictions,
checkpoints, diagnostics, dashboard image assets and training logs remain local.

## 2026-11: series-sampling classification

- MaxViT-Tiny, 512px, nine channels (three adjacent slices x three CT windows).
  Each training epoch visits every series once, drawing one labeled center slice.
  Mixed series choose positive/negative bins with probability 0.5 each, cycling
  independently without replacement. Six slice targets use weighted BCE.
- The initial queue selecting slice AUC was superseded by a fresh queue selecting
  any-hemorrhage series AUC from maximum slice predictions.
- Completed 50-epoch reference: best series AUC 0.9899413 at epoch 50.
  Completed 75-epoch run: best 0.9898380 at epoch 35. These use a fixed validation
  set, not OOF predictions. The 100-epoch run was stopped at user request.
- At five false positives among 589 validation negatives, the 50-epoch reference
  detects 31/50, 87/106, 142/157 and 97/98 positive studies in the >0–10%,
  >10–25%, >25–50% and >50% positive-slice-fraction groups respectively.
  Threshold/checkpoint selection used this validation set; results are descriptive.

## Backbone ablations

The 50-epoch queue tests EfficientNetV2-M, ConvNeXt-Tiny and CoAtNet-0;
ResNet-50 was removed before training. The completed MaxViT run is the reference.
Separate dependent queues append CSN-R101 and NaFlexViT-Base. Check live queue
records/services for current state, rather than treating this document as live status.

Shared optimizer schedule: 299 steps/epoch, 14,950 total, 748 warmup. Batch 32
per GPU on two GPUs; validation every five epochs and best-series-AUC export.
CSN uses explicit [3 windows, 3 slices, 512, 512] inputs and Kinetics pretraining.
NaFlexViT keeps nine flattened channels and ImageNet-1k weights. Their GPU
preflights run before training; failure does not silently change the protocol.

## Coronal dashboard

Local Tailscale dashboard source and reproduction instructions live under
`sites/coronal-lab/`. Generated study assets, downloaded weights and fitted models
are excluded from Git. No neural reconstruction training is restarted by these
classification queues.

## 2026-11-10 BHSD segmentation and HU audit

The ConvNeXt-Tiny stride 16 binary segmentation teacher schedule study was launched
with five folds and independent 10/25/50-positive-pass budgets; later discussion
makes 50 conditional on reviewing 25. All 50-pass runs were subsequently cancelled by the user: fold 0 was stopped
after launch; folds 1–4 were cancelled before launch. Remaining 25-pass training
was preserved.
See [schedule protocol](BHSD_CONVNEXT_SCHEDULE_STUDY.md) and
[discussion and decisions](SEGMENTATION_PSEUDOLABEL_PLAN.md). Use the live queue
for current status rather than treating this entry as a completion record.

The manual-mask HU audit completed on 192 BHSD studies.0–100 HU covers 99.68% of
annotated hemorrhage pixels while occupying 18.67% of all image pixels;40–100
covers 71.65% and disproportionately loses small bleeds. Local artifacts are in
`diagnostics/bhsd_hu_coverage_20261110/`; reproduction script:
`scripts/analyze_bhsd_hu_coverage.py`. No teacher predictions defined these bands
or supplied the annotation coverage measurements.

## 2026-11-10 BHSD sampling and Dice reduction ablations

Launched the five remaining sampling/reduction configurations across all five
folds, using the completed positive-only per-slice25-pass runs as reference.
Budgets, validation cohorts/intervals, LR schedules and metrics match the reference;
reported validation loss retains the common per-slice formulation across arms.
See [ablation protocol and locations](BHSD_CONVNEXT_SCHEDULE_STUDY.md#sampling-and-dice-reduction-ablations-launched-2026-11-10).

## 2026-11-10 additional teacher backbones

All25 sampling/Dice ablations completed. Final-checkpoint negative-slice gating
analysis favored positive-only batch Dice for the downstream pseudo-label use
case; Ian selected that recipe. Launched15 backbone-only teacher runs for
MaxViT-Tiny, EffNetV2-M and CoAtNet-0, five folds each, using each family's final
stride16 spatial-logit classifier initialization. These augment the five existing
ConvNeXt teachers into a planned20-model collection. See the
[protocol and registry locations](BHSD_CONVNEXT_SCHEDULE_STUDY.md#selected-recipe-and-additional-backbone-teachers).

## 2026-11-10 final teacher ensemble evaluation

Evaluated all20 final teachers through their held-out BHSD folds. Each study uses
one held-out model per family; no in-fold teacher predictions enter this audit.
Equal arithmetic means of probabilities compare ConvNeXt+MaxViT+CoAtNet (three)
against those families plus EffNet (four). All individual models were reevaluated
from the same float16 probability exports for consistency.

At threshold0.5, the three-family ensemble has positive-slice Dice0.63767,
ungated volume Dice0.63714, and known-negative-gated volume Dice0.69333. The
four-family values are0.63850,0.63976,0.69266. Strongest individual gated Dice
at0.5 is MaxViT0.67467. Paired patient bootstrap gives three-minus-MaxViT gated
volume delta+0.01870 (95%CI+0.00667 to+0.03141), while three-minus-four is+0.00066
(CI-0.00120 to+0.00262). Intervals are descriptive, conditional on these models,
and not adjusted for multiple comparisons or model/threshold selection.

Threshold0.6 maximizes three-family mean positive-slice and gated volume Dice
on this sweep:0.63868 and0.69866. Four-family gated volume Dice at0.6 is0.69658;
the paired difference CI crosses zero. Threshold tuning uses these same OOF
validation cases, so it is exploratory rather than independently validated.

Recommendation: the three-family ensemble is the simpler leading candidate;
EffNet adds no clear gain here. This is not an artifact-deletion decision or an
automatic change to the20-teacher registry. For future non-BHSD pseudo-labeling,
three families would mean15 teachers; this BHSD audit uses only3 per case to
preserve out-of-fold evaluation.

Script: `scripts/analyze_bhsd_teacher_ensemble.py`.
Results/provenance: `diagnostics/bhsd_teacher_ensemble_20261110/`.
Saved per-fold three/four-family soft probability maps, threshold overlap counts,
fold/series metrics, summary and paired patient-bootstrap comparisons. No model
inference or retraining was performed.

### 2026-11-12: locked three-model auxiliary ensemble test evaluation
User explicitly authorized test-set evaluation of final ConvNeXt-Tiny, MaxViT-Tiny,
and CoAtNet-0 weight-0.25 models. Equal mean slice any logits, then normalized
LSE T=0.25 across slices, sigmoid; no calibration fit or test-set tuning.
Artifacts: `diagnostics/auxiliary_w025_test_lse025_20261112/` (protocol, per-model
checkpoint provenance, ordered slice predictions/manifests, series predictions,
script, results). Test patients verified disjoint from train/validation; BHSD
exclusion preserved. 50,415 slices, 1,450 series, 989 patients, 652 positive series.
Series AUC 0.9809935, Brier 0.0497749, log loss 0.1764770, 10-bin ECE 0.0228674.
At fixed probability 0.5: TP618 FN34 TN733 FP65, sensitivity0.947853,
specificity0.918546. All checkpoint hashes verified against final-export metadata;
all three prediction arrays aligned by exact slice path, target and group index.

### Teacher-patient audit and max-pooling comparison
The historical exclusion above was study-based, not complete teacher-patient
isolation:8 teacher patients (40 test series) and1 validation patient overlapped.
The post-hoc teacher-disjoint test audit retains1410 series/981 patients.
See `docs/ICH_ALT_RELEASE.md` for corrected metrics and the fixed max-versus-LSE
comparison. The corrected reproduction pipeline excludes all teacher patients;
these historical weights were not retrained with that corrected cohort.
