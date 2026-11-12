"""Bounded real-image parity test; writes aggregate evidence only, never pixels."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from skp.ich_alt import ICHEnsemble, FAMILIES, preprocess_slab
from importlib import import_module
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    from skp.configs.rsna_ich_convnext_tiny_s16_auxiliary_roi_w025 import cfg
    from skp.datasets.rsna_ich_2p5d import Dataset

    ds = Dataset(cfg, "val")
    indices = [int(np.flatnonzero(ds.df["any"].to_numpy() == v)[0]) for v in (0, 1)]
    slabs = np.stack([ds[i]["x"].numpy() for i in indices])
    for i, x in zip(indices, slabs):
        row = ds.df.iloc[i]

        def read(path):
            if not isinstance(path, str) or not path:
                return None
            return cv2.imread(str(Path(cfg.data_dir) / path), cv2.IMREAD_UNCHANGED)

        reproduced = preprocess_slab(
            read(row.prev_slice_path),
            read(row.slice_path),
            read(row.next_slice_path),
            rescale_slope=row.rescale_slope,
            rescale_intercept=row.rescale_intercept,
        )
        np.testing.assert_allclose(reproduced, x, atol=1e-6)
    ensemble = ICHEnsemble(a.release, device="cuda:0")
    out = ensemble.predict_series(slabs, batch_size=1, return_segmentation=True)
    assert out["segmentation"].shape == (2, 1, 512, 512) and out[
        "any_logit_map"
    ].shape == (2, 1, 32, 32)
    for value in out.values():
        assert np.isfinite(value).all()
    max_errors = {}
    for f, model in zip(FAMILIES, ensemble.models):
        name = "maxvit" if f == "maxvit_tiny" else f
        meta = json.loads((a.evidence / f"{name}_provenance.json").read_text())
        c = import_module("skp.configs." + meta["config"]).cfg.__deepcopy__()
        c.pretrained = False
        original = import_module("skp.models." + c.model).Net(c)
        state = torch.load(
            meta["checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )["state_dict"]
        original.load_state_dict(
            {
                k.removeprefix("model."): v
                for k, v in state.items()
                if k.startswith("model.")
            },
            strict=True,
        )
        original.cuda().eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            original_out = original(
                {"x": torch.from_numpy(slabs[:1]).cuda()},
                return_features=True,
                return_segmentation=True,
            )
            exported = model(
                {"x": torch.from_numpy(slabs[:1]).cuda()},
                return_features=True,
                return_segmentation=True,
            )
        max_errors[f] = {}
        for key in ["logits", "segmentation_logits", "logit_map"]:
            torch.testing.assert_close(original_out[key], exported[key], rtol=0, atol=0)
            max_errors[f][key] = float((original_out[key] - exported[key]).abs().max())
        # Public loader leaves weights trainable, and both tasks backpropagate.
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(
                {"x": torch.from_numpy(slabs[:1]).cuda()}, return_segmentation=True
            )
            loss = (
                v["logits"].float().square().mean()
                + v["segmentation_logits"].float().square().mean()
            )
        loss.backward()
        assert (
            model.linear.weight.grad is not None
            and model.segmentation_head.weight.grad is not None
        )
        assert any(p.grad is not None for p in model.backbone.parameters())
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        model.zero_grad(set_to_none=True)
        model.eval()
        del original, state
    result = dict(
        status="passed",
        samples=2,
        sample_content="one positive and one negative validation slice; no patient identifiers or pixels included",
        preprocessing="parity with training dataset",
        checkpoint_parity_max_errors=max_errors,
        output_shapes={k: list(v.shape) for k, v in out.items()},
        fine_tuning="finite backward gradients through backbone, classifier and decoder; no optimizer step",
        device=torch.cuda.get_device_name(0),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limits="Small runtime smoke, not full evaluation or clinical validation",
    )
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
