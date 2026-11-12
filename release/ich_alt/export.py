"""Export the approved three final raw auxiliary classifiers. No remote writes."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from safetensors.torch import save_file
from skp.ich_alt import FAMILIES, LABELS, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    selected = []
    for family in FAMILIES:
        name = "maxvit" if family == "maxvit_tiny" else family
        meta = json.loads((a.evidence / f"{name}_provenance.json").read_text())
        checkpoint = Path(meta["checkpoint"])
        if (
            meta["epoch"] != 49
            or meta["global_step"] != 14950
            or meta.get("prediction_weights") != "raw"
        ):
            raise ValueError("Wrong approved checkpoint state")
        if sha256(checkpoint) != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        selected.append((family, meta, checkpoint))
    a.output.mkdir(parents=True)
    (a.output / "weights").mkdir()
    components = []
    for family, meta, checkpoint in selected:
        c = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if c["epoch"] != 49 or c["global_step"] != 14950:
            raise ValueError("State mismatch")
        state = {
            k.removeprefix("model."): v.contiguous()
            for k, v in c["state_dict"].items()
            if k.startswith("model.")
        }
        save_file(
            state,
            a.output / "weights" / f"{family}.safetensors",
            metadata={
                "family": family,
                "state": "raw final epoch50",
                "source_checkpoint_sha256": meta["checkpoint_sha256"],
            },
        )
        components.append(
            dict(
                family=family,
                epoch=50,
                global_step=14950,
                state="raw",
                selection="final",
                source_checkpoint_sha256=meta["checkpoint_sha256"],
                config=meta["config"],
            )
        )
        del c, state
    root = Path(__file__).resolve().parent
    for name in [
        "README.md",
        "requirements.txt",
        "NOTICE.md",
        "LICENSE-weights",
        "LICENSE-smp",
    ]:
        shutil.copy2(root / name, a.output / name)
        if name == "README.md":
            card = a.output / name
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            card.write_text(card.read_text().replace("SOURCE_REVISION", revision))
    shutil.copy2(root.parents[1] / "LICENSE", a.output / "LICENSE-code")
    (a.output / "config.json").write_text(
        json.dumps(
            dict(
                labels=LABELS,
                input_shape=[9, 512, 512],
                output_stride=16,
                ensemble="mean slice logits",
                series_pooling="normalized_lse",
                temperature=0.25,
                segmentation="ungated mean probabilities",
                subtype_series_metrics="not measured",
            ),
            indent=2,
        )
    )
    inventory = {
        str(p.relative_to(a.output)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(a.output.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        schema_version=1,
        status="candidate",
        repo_id="ianpan/ct-head-hemorrhage-detection-alt",
        labels=LABELS,
        components=components,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        files=inventory,
    )
    (a.output / "release.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                files=len(inventory), bytes=sum(v["bytes"] for v in inventory.values())
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
