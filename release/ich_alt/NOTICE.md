# Licenses and attribution

New model weights: Apache License 2.0, copyright 2026 Ian Pan; see LICENSE-weights.
Project runtime code: MIT, copyright 2026 Ian Pan; see LICENSE-code.
The runtime U-Net decoder is adapted from segmentation_models.pytorch (MIT),
https://github.com/qubvel-org/segmentation_models.pytorch; its license is bundled.
The runtime uses timm (Apache-2.0). Pretrained lineage:

- https://huggingface.co/timm/convnext_tiny.fb_in1k (Apache-2.0)
- https://huggingface.co/timm/maxvit_tiny_tf_512.in1k (Apache-2.0)
- https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k (Apache-2.0)

Training datasets retain their own terms. BHSD's official repository
https://github.com/White65534/BHSD states NCND/no commercial use, while
https://huggingface.co/datasets/Wendy-Fly/BHSD lists MIT. The author chose
Apache-2.0 for these trained weights after reviewing this discrepancy; this is
not a legal determination that either dataset statement permits all downstream
uses. RSNA challenge data also has separate terms. No training data is included,
and this license does not grant rights in third-party data or replace its terms.
ImageNet training lineage is inherited from the above pretrained weights.
