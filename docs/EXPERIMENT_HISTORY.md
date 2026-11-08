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
