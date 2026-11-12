# Pipeline review remediation, 2026-11-12

Supported paths: current ConvNeXt/MaxViT/CoAtNet auxiliary ROI0.25 + LSE0.25,
and prior published MaxViT/BiGRU/frozen DeepLab training recipe. See ICH_PIPELINE.md
and ICH_LEGACY_PIPELINE.md for explicit commands.

Closed gaps: public prepared RSNA source identified; released BHSD NIfTI has a
PNG/manifest producer; dataset roots/manifests and checkpoints configurable;
new splits default to whole-teacher-patient exclusion; teacher preparation,
training, pseudolabel generation, calibration, and test evaluation check overlap;
selected new route no longer depends on 299/14950/92453/91744/709 historical counts;
source prediction/calibration/evaluation entrypoints are outside ignored diagnostics.
Historical launchers and artifacts remain intact as evidence, not the new-user route.

Corrected historical test summary (no refit): exclude 40 series from8 BHSD patients.
1,410 series,981 patients,623 positives; AUC0.9814991199, Brier0.0488279009,
log loss0.1741750733; threshold.5 gives TP593 FN30 TN723 FP64.
Original 1,450-series AUC0.9809935114 remains historical. This is a post-hoc cohort
correction; earlier calibration selection included one BHSD patient. No claim of
new untouched test selection is made. The old release test was already consumed.

Verification: 51 targeted tests passed, including synthetic BHSD NIfTI/image/mask
preparation, overwrite refusal, patient-exclusion and teacher-overlap guards,
positive-only weighted losses, sampler/LR budget, sequence and frozen-encoder
contracts. CLI plans/imports and static undefined-name checks passed. Independent
read-only reviewer identified path-helper shadowing and a historical step assertion;
both were fixed, and guard execution tests added. Reviewer found no further
confirmed handoff blocker; exact three-model mapping validation was also added.
No full ETL/download/train/inference rerun was performed. Existing training data,
checkpoints, and predictions were not modified. New results will vary with corrected
splits and normal seed/GPU effects. Full pipeline runtime remains unexecuted.
