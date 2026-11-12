# Prior released ICH recipe

This is the distinct earlier route corresponding to
[ianpan/ct-head-hemorrhage-detection](https://huggingface.co/ianpan/ct-head-hemorrhage-detection).
It is retained alongside the newer auxiliary ROI/LSE recipe; it is not replaced
by a stride-16 student ensemble. The released artifacts remain unchanged.

Follow [input acquisition and patient-safe preparation](ICH_PIPELINE.md) first.
Use a new work directory and keep the same ICH_RSNA_ROOT, ICH_SPLITS and
ICH_BHSD_MANIFEST environment variables for all stages.

```bash
uv run python scripts/run_ich_legacy_recipe.py classifier --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py features --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py sequence --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py teachers --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py pseudolabels --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py decoder-pretrain --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py decoders --work experiments/ich_legacy_clean --execute
uv run python scripts/run_ich_legacy_recipe.py package --work experiments/ich_legacy_clean --execute
```

Without `--execute`, each command only displays its explicit recipe. Existing
training and partial pseudolabel directories are not deleted or treated as
completed; investigate interruption before resuming. The old dated
`run_maxvit_segmentation_pipeline.py` remains historical and is not the recommended
entrypoint. No remote upload occurs in `package`.

| Stage | Implementation | Producer → consumer |
|---|---|---|
| Classifier | `rsna_ich_maxvit_tiny_fixedval_9ch` | Three all-slice epochs; final classifier → feature extraction and teacher initialization |
| Features | `extract_rsna_ich_features.py` | Frozen 512-D features and logits for train/val → sequence dataset |
| Sequence | `rsna_ich_maxvit_tiny_sequence_bigru_selected` | 20-epoch two-layer BiGRU, loss weights 1/.25/.1; best slice-AUC weights → fixed blend |
| Teachers | `bhsd_maxvit_tiny_seg_9ch` | Five patient-fold U-Net teachers → OOF threshold selection and RSNA masks |
| Masks | `generate_bhsd_9ch_pseudolabels.py` | OOF thresholds, train-only pseudo-masks, merged manifest → decoder pretrain |
| Decoder pretrain | `rsna_ich_joint_maxvit_tiny_9ch_deeplab_frozen` | Frozen encoder/classifier; three-epoch DeepLab training → final folds |
| Final localization | `bhsd_maxvit_tiny_seg_9ch_deeplab_pseudolabel_frozen` | Ten epochs, five folds, canonical encoder restored after pretrained model load; best volume-Dice decoders → package |
| Packaging/evaluation | `package_ich_legacy.py`, `evaluate_ich_legacy.py` | One canonical encoder plus BiGRU and five decoder heads → slice/series predictions and AUC |

Classifier input remains three adjacent slices × three windows at 512², with
zero windowed missing neighbors. This is the original pooled-feature classifier,
not spatial-logit pooling. Losses and augmentation settings live in the actual
configs. Student and final decoder training commands use both GPUs together.
Feature generation uses both GPUs; pseudolabel generation is a single-GPU export.
The wrapper supplies checkpoint and cache paths through ICH_LEGACY_* environment
variables, preserving existing historical defaults when those variables are absent.

The package exporter verifies every encoder parameter and buffer equals the
canonical classifier before retaining only decoder weights. It copies the existing
inference architecture and fixed class-specific logit blends, but writes a new
README rather than attributing historical metrics to newly trained weights.
Best checkpoints are intentional for this earlier sequence/decoder recipe; the
new auxiliary recipe instead uses final student checkpoints.

Validation and explicitly requested heldout evaluation:

```bash
uv run python scripts/evaluate_ich_legacy.py --model-dir experiments/ich_legacy_clean/model --split val --teacher-manifest "$ICH_BHSD_MANIFEST" --output diagnostics/legacy_clean_val
# After freezing the recipe; this consumes the holdout:
uv run python scripts/evaluate_ich_legacy.py --model-dir experiments/ich_legacy_clean/model --split test --teacher-manifest "$ICH_BHSD_MANIFEST" --output diagnostics/legacy_clean_test
```

For localization, each final training fold logs slice/volume Dice and HD95;
`HemorrhageSliceModel(local_dir=...).predict_series(..., return_segmentation=True)`
returns probability maps for display. BHSD localization still reuses teacher
information and must be reported as descriptive. Classification splits now exclude
all teacher patients, so membership and numerical results differ from the historical
release. Fixed historical blend coefficients are reused, not selected on test.
Seed/hardware variation is normal; identical historical metrics are not promised.

The prior model-card test was already consumed during earlier releases. It is not
a pristine holdout for ongoing selection. The new cohort audit and corrected
patient-split policy are documented in ICH_PIPELINE.md.
