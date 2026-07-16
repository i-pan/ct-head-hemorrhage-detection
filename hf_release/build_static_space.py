from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "hf_release" / "space"
MODEL = ROOT / "hf_release" / "model"
os.environ.setdefault("LOCAL_MODEL_DIR", str(MODEL))
sys.path.insert(0, str(SPACE))

import app  # noqa: E402


def main() -> None:
    manifest = json.loads((SPACE / "examples" / "manifest.json").read_text())
    assets = SPACE / "assets"
    assets.mkdir(exist_ok=True)
    records = []

    for item in manifest:
        key = item["key"]
        prefix = SPACE / "examples" / key
        paths = [
            str(prefix.with_name(prefix.name + f"_{position}.png"))
            for position in ["previous", "center", "next"]
        ]
        labels, overlay, _ = app.predict(
            *paths,
            item["rescale_slope"],
            item["rescale_intercept"],
            "Any hemorrhage",
        )
        center = app.window_slice(
            app.load_png(paths[1]),
            item["rescale_slope"],
            item["rescale_intercept"],
        )[0]
        center = np.repeat(
            (np.clip(center, 0, 1) * 255).astype(np.uint8)[..., None],
            3,
            axis=-1,
        )
        center_name = f"{key}_brain.jpg"
        overlay_name = f"{key}_overlay.jpg"
        cv2.imwrite(
            str(assets / center_name),
            cv2.cvtColor(center, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        cv2.imwrite(
            str(assets / overlay_name),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        records.append(
            {
                "key": key,
                "ground_truth": item["ground_truth"],
                "probabilities": labels,
                "brain_image": f"assets/{center_name}",
                "overlay_image": f"assets/{overlay_name}",
            }
        )

    (SPACE / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} static examples to {assets}")


if __name__ == "__main__":
    main()
