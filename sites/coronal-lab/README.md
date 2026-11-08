# Coronal reconstruction lab

Local static dashboard showing four real BHSD / RSNA overlapping CT studies,
patient-coordinate coronal reconstruction, and fitted experimental 3D neural priors.

## Access and service

- Tailscale: http://100.68.163.27:8773
- Loopback: http://127.0.0.1:8773
- User service: `ich-coronal-dashboard.service`
- Status: `systemctl --user status ich-coronal-dashboard.service`
- Restart: `systemctl --user restart ich-coronal-dashboard.service`
- Stop: `systemctl --user stop ich-coronal-dashboard.service`

The server binds only loopback and the host's Tailscale IPv4 address. It serves
`dist/`, without directory listings. No cloud deployment or external inference
is used. A Sites registration was created before the local-only preference
arrived; it remains unpublished, with no source or study assets uploaded.
The `.openai/hosting.json` records that unused registration to avoid duplicating it.

Scroll over any coronal panel to move all three views together. Arrow keys,
the position slider, and cine playback also navigate. The acquired-plane view
scrolls independently; clicking it moves the coronal plane. On small screens,
use “Show acquired-slice navigator”. The third panel switches between cubic,
a single neural prior, the two-seed ensemble, seed disagreement, and absolute
ensemble-versus-linear differences. Export saves the current comparison with
method labels, provenance, and limitations.

## Reproduce assets

From `/home/ian/projects/ich` using its existing Python environment:

```bash
.venv/bin/python sites/coronal-lab/scripts/prepare.py prepare
.venv/bin/python sites/coronal-lab/scripts/prepare.py train --seed 88 --device cuda:0 --steps 350
.venv/bin/python sites/coronal-lab/scripts/prepare.py train --seed 123 --device cuda:1 --steps 350
.venv/bin/python sites/coronal-lab/scripts/prepare.py finish
```

Requires NumPy, pandas, SciPy, nibabel, PyTorch, and the existing local BHSD and
RSNA dataset paths in `prepare.py`. Two seeds may run independently on separate
GPUs. CUDA kernels are not configured for bitwise deterministic reproduction.
Native arrays, full affines, generated arrays, fit traces, and consistency
statistics are in ignored `work/full512/`. The displayed assets and metadata are in
ignored `dist/assets512/`. Original pilot outputs remain in `work/` and `dist/assets/`. Serving does not require a GPU or Python ML packages.

The selected images are BHSD NIfTIs; the script retains their full affines,
canonicalizes to RAS and retains every original 512 × 512 axial sample with
no in-plane reduction. Patient-coordinate reconstruction uses native X/Y spacing
(about 0.488 mm) and 1.25 mm superior–inferior spacing. The coronal width is 512;
height follows physical coverage, including tilt. The viewer and PNG exports
use the physical aspect ratio, accounting for the non-square coronal pixels.
`spacing_xyz` and the target affine specify spacing explicitly.
Binary assets are gzip-compressed, little-endian int16, row-major Z,Y,X.
Display spacing is not native spatial resolution. Coverage is the distance
between first and last slice centers, not the physical slice-edge extent.

## Experimental AI

The custom generator has three 3D convolution/LeakyReLU/upsampling blocks and
a sigmoid output (42,417 parameters). Each study and seed is independently
fitted from fixed random noise using only observed data. Adam uses lr 0.003
for 350 steps with a tissue-weighted MSE and a small through-plane TV penalty.
It is inspired by Deep Image Prior, not an exact reproduction of the published
architecture and not a pretrained diffusion/GAN model.

The generator fits a clipped −100 to 200 HU representation. Its source-plane
residual against **full-range** native intensities is linearly interpolated and
added back to the output, preserving full-range source samples under a
point-sampling approximation. Projection does not model slice averaging or
scanner kernels. Output is sampled at four times the native slice density
before world-coordinate resampling. Two seeds (88, 123) provide an ensemble
mean and an absolute disagreement map; neither is calibrated uncertainty.

The original custom-prior experiment used no missing-slice ground truth,
withheld-slice evaluation, or clinical validation. Study selection is
illustrative and entirely positive. The later pretrained comparison below
adds a withheld-slice benchmark for QDM, Real-ESRGAN, and Real-ESRNet.

References: [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior/)
and [DynamicDPS (MRI research)](https://papers.miccai.org/miccai-2025/0899-Paper4526.html).

## Verification

`scripts/validate_assets.py` checks data sizes, finite geometry, source-sample
consistency, and affine round trips. `scripts/browser-check.cjs` checks all
studies, linked wheel navigation, boundaries, method selection, windowing,
plane guides, playback, PNG export, and mobile layout using the already
installed Playwright in the neighboring `eden-report-review` project.

## Pretrained comparison (October 24, 2026)

Three official checkpoints are now added:

| Model | Training domain | Role |
| --- | --- | --- |
| QDM-L medical ×4 + medical VAE | HaN-Seg / SegRap2023 head-and-neck CT | CT-specific latent diffusion, 15 sampling steps |
| Real-ESRGAN ×4plus | Natural images | Adversarial RRDB generator; out-of-domain control |
| Real-ESRNet ×4plus | Natural images | Non-adversarial pixel-loss RRDB counterpart |

QDM has 691,941,580 diffusion-model parameters (plus its VAE); each Real-ESR
network has 16,697,987. No checkpoint is fine-tuned locally. The code verifies
all inference weights with strict loading; only training-only loss-module
keys from the VAE checkpoint are excluded. Checkpoints are loaded using
`weights_only=True`. SHA-256 hashes and upstream Git revisions are recorded
in the download script and the downloadable dashboard metadata.

Install supplemental packages in a checkout-local directory, leaving the
project's training dependencies unchanged:

```bash
uv pip install --python .venv/bin/python --target sites/coronal-lab/work/deps \
  --no-deps -r sites/coronal-lab/scripts/pretrained-requirements.txt
.venv/bin/python sites/coronal-lab/scripts/fetch_pretrained.py
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py baselines
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py run --method esrgan --device cuda:1 --batch-size 8
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py run --method esrnet --device cuda:1 --batch-size 8
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py run --method qdm --device cuda:0 --batch-size 8 --cases case-01 case-02
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py run --method qdm --device cuda:1 --batch-size 8 --cases case-03 case-04
.venv/bin/python sites/coronal-lab/scripts/run_pretrained.py finish
.venv/bin/python sites/coronal-lab/scripts/validate_pretrained.py
```

The two QDM commands can run concurrently on the two GPUs. Assets are published
to the local dashboard by the `finish` stage only after every study/model run
has completed. No external upload or hosted deployment is involved.

### Transfer adapter and limitations

Each model actually processes **every one of the 512 native-grid coronal
planes**, independently. There are no linearly interpolated stand-ins along
AP. Input is clipped to −100…200 HU, normalized to [0,1], and repeated across
RGB channels. QDM internally receives [-1,1] as in the original sampler.
Output RGB channels are averaged and mapped back to HU-equivalent intensity
units. The published QDM inference code and paper do not specify an original
HU window, so our selected window is an additional transfer assumption.

These checkpoints perform two-dimensional isotropic ×4 SR; using them for
anisotropic through-plane restoration is experimental. QDM uses overlapping
128-pixel tiles with repeated reflection padding, the checkpoint's 0.15
quadtree threshold, and the official 15-step stochastic schedule. X is
area-reduced from the model’s ×4 output back to the original 512 columns;
no acquired input pixels are discarded. Z is aligned to native voxel centers
using the 4× pixel-center mapping (source centers at output positions
`1.5 + 4*k`) and sampled at `(N-1)*4+1` points. The unchanged NIfTI affine then
maps this volume into patient coordinates for the dashboard.

The default displayed result has a piecewise-linear residual correction
calculated from acquired planes. It preserves their full-range intensity
samples but cannot establish that intermediate detail is accurate. Disable
“Preserve acquired samples” to view raw model output; both variants are
retained. Independent 2D inference can cause AP flicker. No scanner slice
profile or physical averaging model is represented.

### Withheld-slice benchmark

For each study, retain native slices 0,2,4,… and withhold the interior odd
slices. An unbracketed final odd slice is excluded. This yields **59 withheld
axial planes** across four studies. Neither inference nor source correction
receives the withheld reference slices. The task doubles native spacing to
5–13.04 mm and tests reconstruction at existing acquired slice centers; it
does not validate truly missing thin slices. The existing fitted custom priors
are not included in this benchmark because their original fits used all slices.

The dashboard provides current-study and equal-study-weight aggregate scores,
and a separately scrollable withheld-reference/linear/neural comparison:

- Soft-tissue MAE: reference intensities −20…120 HU, including extracranial
  soft tissue; this is not an anatomical brain mask.
- Hemorrhage MAE: nonzero BHSD labels, nearest-resampled to the shared in-plane
  grid. The dashboard reports the actual voxel count for each study and flags
  very small hemorrhage regions.
- SSIM and PSNR: whole axial images clipped to the brain display window
  (W80/L40), including background. No learned perceptual score is used.

Scores use retained floating-point predictions before int16 viewer export.
The UI toggles corrected and raw-model scores with the source-preservation
control. These are descriptive scores for four selected positive examples,
not a clinical evaluation or dataset-wide generalization result.


`check_reference_adapters.py` additionally compares our RRDB inference forward
pass and QDM sampling tile numerically with the upstream class / sampler code
under identical weights, inputs, and random seeds. Both matched exactly in
this run (maximum absolute difference 0). This checks adapter implementation;
it does **not** establish appropriate CT preprocessing or domain generalization.
The QDM raw outputs in the initial previews showed substantial intensity drift,
which is why the raw/corrected comparison and assumed HU window are prominent.

## Full-resolution rebuild

The full-resolution pipeline rebuilds baselines, both fitted-prior seeds, all
three pretrained models, and withholding metrics at 512 × 512. A model still
rebuilding is disabled and explicitly marked as pending; old 192-grid neural
outputs are never mixed with the full-resolution viewer. Old pilot scores should not be mixed with the new full-resolution scores.
Source validation compares every retained voxel and the affine directly with the
canonical NIfTI. Browser checks verify 512-pixel source/coronal widths and physical
aspect ratio. The browser caches at most 256 MiB beyond its active references.

To stage a rebuild without replacing the served assets, set `CORONAL_ASSETS` to
a separate directory; `CORONAL_WORK` similarly overrides `work/full512/`.
Checkpoint weights, upstream repositories, and supplemental dependencies remain
in `work/weights`, `work/vendor`, and `work/deps`. Copy `work/checkpoint-hashes.json`
into the run directory before the pretrained finish stage.
