from skp.toolbox.ich_paths import path as configured_path
"""Verified contiguous float16 probability cache for random slice access."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = configured_path("ICH_PSEUDOLABELS", ROOT / "data/pseudolabels/rsna_three_family_final_20261110")


def main():
    completion = json.loads((OUT / "completion.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert completion["status"] == "complete"
    assert completion["export_signature"] == provenance["export_signature"]
    path = OUT / "training_probabilities.npy"
    marker = OUT / "training_cache.json"
    n = completion["target_slices"]
    if path.exists() and marker.exists():
        info = json.loads(marker.read_text())
        a = np.load(path, mmap_mode="r")
        assert info["export_signature"] == completion["export_signature"]
        assert a.shape == (n, 512, 512) and a.dtype == np.float16
        print("Verified existing training cache", flush=True)
        return
    tmp = OUT / "training_probabilities.tmp.npy"
    a = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, 512, 512))
    for start in range(0, n, provenance["shard_size"]):
        end = min(start + provenance["shard_size"], n)
        with np.load(
            OUT / "shards" / f"{start // provenance['shard_size']:05d}.npz"
        ) as shard:
            assert str(shard["export_signature"]) == completion["export_signature"]
            assert np.array_equal(shard["target_index"], np.arange(start, end))
            p = shard["probability"]
            assert (
                p.shape == (end - start, 512, 512)
                and np.isfinite(p).all()
                and p.min() >= 0
                and p.max() <= 1
            )
            a[start:end] = p
            assert np.array_equal(a[start:end], p)
    a.flush()
    del a
    tmp.replace(path)
    marker.write_text(
        json.dumps(
            dict(
                export_signature=completion["export_signature"],
                shape=[n, 512, 512],
                dtype="float16",
                bytes=path.stat().st_size,
            ),
            indent=2,
        )
    )
    print("Verified and wrote training cache", flush=True)


if __name__ == "__main__":
    main()
