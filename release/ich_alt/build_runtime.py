"""Build an allowlisted runtime wheel, excluding experiment configs and data paths."""

from pathlib import Path
import argparse
import shutil
import subprocess

FILES = [
    "__init__.py",
    "ich_alt.py",
    "configs/__init__.py",
    "configs/base.py",
    "configs/defaults.py",
    "models/__init__.py",
    "models/normalization.py",
    "models/pooling.py",
    "models/utils.py",
    "models/classification/net2d.py",
    "models/classification/ich_auxiliary.py",
    "models/classification/maxvit_logit_pool.py",
    "models/classification/spatial_logit_backbones.py",
    "models/segmentation/decoders/unet.py",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    a.build.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        target = a.build / "src/skp" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "src/skp" / name, target)
    shutil.copy2(root / "LICENSE", a.build / "LICENSE")
    (a.build / "pyproject.toml").write_text("""[project]
name = "skp-ich-alt-runtime"
version = "0.1.0"
description = "Inference and fine-tuning runtime for the alternate ICH ensemble"
requires-python = ">=3.11"
license = "MIT"
dependencies = ["torch==2.6.0", "timm==1.0.27", "numpy==2.4.6", "opencv-python-headless==4.13.0.92", "safetensors==0.7.0", "huggingface-hub==1.17.0", "einops==0.8.2"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/skp"]
""")
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(a.output.resolve()),
            str(a.build.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
