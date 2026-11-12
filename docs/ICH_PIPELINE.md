# ICH training and evaluation pipeline

Two supported paths share input preparation: the current three-backbone auxiliary
recipe below and the [prior published MaxViT/BiGRU/DeepLab recipe](ICH_LEGACY_PIPELINE.md).
Run commands from the repository root. Historical dated launchers are experiment
records; new work uses `run_ich_recipe.py` or `run_ich_legacy_recipe.py`.

## Obtain inputs and install

```
uv sync --group train --group dev --group medical
```

The prepared starting dataset is
[ianpan/rsna-intracranial-hemorrhage-16bit-png](https://huggingface.co/datasets/ianpan/rsna-intracranial-hemorrhage-16bit-png/tree/main).
Download `images.tar`, `slice_labels.csv`, and `rescale_values.csv`. Extract the
archive into a new directory, preserving `images/<patient>/<study>/<series>/*.png`.
Record the download revision/checksums. The prepared dataset card identifies the
original RSNA Kaggle challenge; follow its access/usage terms as well as the
prepared dataset terms. No raw DICOM conversion is needed for this route.

Obtain `label_192.zip` through the [official BHSD project](https://github.com/White65534/BHSD),
which links to [the BHSD dataset](https://huggingface.co/datasets/Wendy-Fly/BHSD).
Extract to a directory containing `images/` and `labels/`. Some releases omit the
`.nii.gz` suffix; the preparation script creates correctly named local symlinks.
The upstream GitHub and Hugging Face license descriptions differ: consult the
upstream terms applicable to your use; this repository does not relicense images.
Do not redistribute prepared data under the code license.

Example local setup (replace `/data/...` with your own paths):

```bash
export ICH_RSNA_ROOT=/data/rsna16
export ICH_BHSD_ROOT=/data/bhsd_prepared
export ICH_BHSD_MANIFEST="$PWD/data/bhsd_clean/bhsd_segmentation_slices.csv"
export ICH_BHSD_HU_CACHE="$PWD/data/bhsd_clean/hu_volumes"
export ICH_SPLITS="$PWD/data/folds/ich_patient_safe.csv"
uv run python scripts/prepare_bhsd_inputs.py --source /data/bhsd_raw/label_192 --output "$ICH_BHSD_ROOT"
uv run python scripts/prepare_bhsd_segmentation.py --bhsd-root "$ICH_BHSD_ROOT" --output-dir data/bhsd_clean --examples-dir diagnostics/bhsd_clean_orientation
uv run python scripts/create_rsna_ich_splits.py --rsna-root "$ICH_RSNA_ROOT" --bhsd-root "$ICH_BHSD_ROOT" --bhsd-scope patient --val-studies 1000 --test-studies 1000 --output "$ICH_SPLITS" --summary-output data/folds/ich_patient_safe.summary.json
```

Preparation converts released NIfTI into three-window PNGs and its own slice
manifest, then uses the established orientation verifier/mask-volume producer.
Labels 1–5 map to epidural, intraparenchymal, intraventricular, subarachnoid,
subdural; `any` is their union. Teacher inputs remain original HU volumes with
brain/subdural/bone windows. The helper's derived neighboring-slice metadata is
for the legacy route; current teachers use zero missing-neighbor HU windows.

New splits exclude *all studies of BHSD patients*, then group remaining patients
into train/val/test. The historical `--bhsd-scope study` is for explicit archival
replay only. Split generation refuses overwrite by default. Existing historical
splits/results are not rewritten. Teacher training and pseudolabel preparation
reject any teacher-patient overlap with validation/test.

## Current selected recipe

Use a fresh work directory. Each command first prints its effective commands;
without `--execute` it is a plan only. Do not run stages concurrently.

```bash
uv run python scripts/run_ich_recipe.py classifiers --work experiments/ich_clean --execute
uv run python scripts/run_ich_recipe.py teachers --work experiments/ich_clean --execute
uv run python scripts/run_ich_recipe.py pseudolabels --work experiments/ich_clean --execute
uv run python scripts/run_ich_recipe.py students --work experiments/ich_clean --execute
```

1. Classifiers: ConvNeXt-Tiny, MaxViT-Tiny, CoAtNet-0, ImageNet initialized,
   output stride 16, spatial logit max pooling, 9 channels at 512², 50 sampled-series
   epochs. One center slice per series, positive/negative bins independently
   shuffled without replacement, positive probability .5. Batch32/GPU on two GPUs;
   299 steps/epoch was historical, **new steps derive from actual series count**.
   AdamW, LR3e-4→1e-6, 5% warmup; six weighted BCE outputs, any weight2.
2. Teachers: selected classifier-initialized encoder, U-Net decoder, 25 positive
   passes, positive-only batch Dice+BCE, encoder LR one tenth decoder, 5 BHSD folds
   per family. These teacher jobs are sequential single-GPU jobs, matching their
   original recipe. This does not change the two-GPU student protocol.
3. Pseudolabels: all 15 final teachers, mean sigmoid center-slice masks, compressed
   shards then float16 random-access cache. New patient-safe splits need no
   BHSD-related supplemental masks. Original supplemental/historical workflows
   remain for archived results, not the clean route.
4. Students: fresh ImageNet initialization, same classification recipe plus binary
   U-Net decoder and spatial any-map BCE. HU[0,100] OR teacher≥.07 dilated3px defines
   weight1; elsewhere weight.25. BCE uses weight-sum normalization; batch Dice on
   real positive slices only; both auxiliary coefficients .5. Student runs are
   sequential, **both GPUs per run**, no accumulation. Decoder skip-channel
   connections adapt to each backbone. Final evaluation occurs at epoch50.

The generated `checkpoints.json` declares classifier, teacher and student roots.
`ICH_CHECKPOINT_REGISTRY` lets the original implementations resolve those outputs;
no historical checkpoint is needed. `ICH_PSEUDOLABELS` and an empty
`ICH_PSEUDOLABEL_SUPPLEMENT` configure the generated cache. The stage runner sets
these variables in its subprocesses. Checkpoint registry/outputs must be kept
alongside the source revision and data manifests. Existing outputs cause training
to stop rather than silently skip interrupted jobs; inspect and resume explicitly.

## Prediction and evaluation

The student stage produces best/final validation archives with hashes and final
BHSD evaluation (Dice at .5 and swept .1–.9). BHSD segmentation remains exploratory:
teacher reuse makes it unsuitable as an untouched generalization test.

The runner generates `experiments/ich_clean/student_runs.json`, mapping
**convnext_tiny**, **maxvit**, **coatnet0** to the student run directories.
Calibration analysis is validation-only; it uses patient-grouped folds and excludes
BHSD rows. The fixed test evaluator performs no fitting:

```bash
uv run python scripts/analyze_ich_lse_pooling.py --runs experiments/ich_clean/student_runs.json --splits "$ICH_SPLITS" --teacher-manifest "$ICH_BHSD_MANIFEST" --output diagnostics/clean_lse_validation
uv run python scripts/evaluate_ich_lse_test.py --runs experiments/ich_clean/student_runs.json --teacher-manifest "$ICH_BHSD_MANIFEST" --output diagnostics/clean_lse_test
```

The test rule is equal mean of three slice any logits, then
`T * (logsumexp(logits/T) - log(number_of_slices))`, T=.25, then sigmoid.
Final raw weights only; no learned calibration. It validates checkpoint hashes,
array alignment, student/teacher patient isolation, and reports AUC/Brier/log loss,
ECE and sensitivity/specificity at fixed .5. It requires two GPUs for inference
shards of one model at a time. Output directories must be new.

Historical test predictions can be audited without new inference:

```bash
uv run python scripts/audit_ich_holdout.py --predictions PATH/test_series_predictions.csv --teacher-manifest "$ICH_BHSD_MANIFEST" --output diagnostics/teacher_disjoint_audit
```

The November12 historical audit removes 40 series from 8 BHSD patients: 1,410
series/981 patients remain; AUC0.981499, Brier0.048828, log loss0.174175,
sensitivity0.951846/specificity0.918679 at .5. Original AUC0.980994 is retained.
This is a post-hoc cohort correction, not a new untouched holdout; earlier
validation selection also included one teacher patient. New training uses the
patient-safe route instead. Randomness/hardware may change new results normally.
