"""Package newly trained legacy weights locally; never uploads or overwrites."""

import argparse, importlib.util, json, shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["classifier", "sequence", "decoders", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.output.exists():
        raise FileExistsError(a.output)
    for f in [
        a.classifier,
        a.sequence,
        *[a.decoders / f"fold{i}/checkpoints/best.ckpt" for i in range(5)],
    ]:
        if not f.is_file():
            raise FileNotFoundError(f)
    spec = importlib.util.spec_from_file_location(
        "legacy_export", root / "hf_release/export_release.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.MODEL_DIR = a.output.resolve()
    m.CLASSIFIER_CHECKPOINT = a.classifier
    m.SEQUENCE_CHECKPOINT = a.sequence
    m.SEGMENTATION_PATTERN = a.decoders / "fold{fold}/checkpoints/best.ckpt"
    a.output.mkdir(parents=True)
    for name in ["config.json", "inference.py", "requirements.txt"]:
        shutil.copy2(root / "hf_release/model" / name, a.output / name)
    config_path = a.output / "config.json"
    config = json.loads(config_path.read_text())
    # A new training run must not claim the historical release's source revision.
    config.pop("source_code_revision", None)
    config["training_recipe"] = "docs/ICH_LEGACY_PIPELINE.md"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (a.output / "README.md").write_text(
        "New training of the prior ICH architecture. Historical model-card metrics do not apply. See docs/ICH_LEGACY_PIPELINE.md for training and evaluation. BHSD localization is descriptive due to teacher reuse.\n"
    )
    m.export_weights()
    (a.output / "training_sources.json").write_text(
        json.dumps(
            {
                k: str(getattr(a, k).resolve())
                for k in ["classifier", "sequence", "decoders"]
            },
            indent=2,
        )
    )
    m.write_checksums()


if __name__ == "__main__":
    main()
