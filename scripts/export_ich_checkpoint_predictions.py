"""Export separate best and latest-checkpoint predictions with verified provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile
import torch
from analyze_ich_backbone_ensemble import export_predictions


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_info(path):
    path = Path(path).resolve(strict=True)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    info = dict(
        checkpoint=str(path),
        epoch=int(state["epoch"]),
        completed_epochs=int(state["epoch"]) + 1,
        global_step=int(state["global_step"]),
        prediction_weights=state.get("prediction_weights", "raw"),
    )
    info["checkpoint_sha256"] = file_sha256(path)
    return info


def export_both(config, run_dir, args):
    run_dir = Path(run_dir)
    directory = run_dir / "fold0/checkpoints"
    candidates = [checkpoint_info(p) for p in directory.glob("last*.ckpt")]
    if not candidates:
        raise FileNotFoundError(f"No latest checkpoint in {directory}")
    latest = max(
        candidates,
        key=lambda x: (
            x["global_step"],
            x["epoch"],
            Path(x["checkpoint"]).name == "last.ckpt",
        ),
    )
    selected = {"best": checkpoint_info(directory / "best.ckpt"), "final": latest}
    for role, info in selected.items():
        output = run_dir / f"validation_predictions_{role}.npz"
        metadata = output.with_suffix(".json")
        provenance = dict(
            info,
            config=config,
            role=role,
            split="fixed_validation",
            final_definition="latest saved checkpoint; inspect completed_epochs against planned budget",
        )
        if output.exists() and metadata.exists():
            old = json.loads(metadata.read_text())
            if all(old.get(k) == v for k, v in provenance.items()) and old.get(
                "archive_sha256"
            ) == file_sha256(output):
                print("Verified existing", output, flush=True)
                continue
            print("Refreshing changed checkpoint", output, flush=True)
        if output.exists() and not metadata.exists():
            raise RuntimeError(f"Export without provenance: {output}")
        fd, tmp = tempfile.mkstemp(prefix=".validation-", suffix=".npz", dir=run_dir)
        os.close(fd)
        os.unlink(tmp)
        try:
            export_predictions(config, Path(info["checkpoint"]), Path(tmp), args)
            # Metadata first: a crash cannot make an unverified archive reusable.
            metadata.write_text(
                json.dumps(
                    dict(
                        provenance,
                        exported_utc=datetime.now(timezone.utc).isoformat(),
                        archive_sha256=file_sha256(tmp),
                    ),
                    indent=2,
                )
            )
            os.replace(tmp, output)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print("EXPORTED", role, info["completed_epochs"], output, flush=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    export_both(args.config, args.run_dir, args)
