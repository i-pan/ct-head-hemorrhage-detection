"""Complete the original classifier cohort with the explicitly authorized 709 masks."""

import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
import export_ich_teacher_pseudolabels as export
import prepare_ich_pseudolabel_training_cache as cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pseudolabels/rsna_three_family_bhsd_related_20261111"


def main():
    torch.set_num_threads(4)
    original = export.DEFAULT_OUT
    provenance = json.loads((original / "provenance.json").read_text())
    source = export.dataset().df
    excluded = pd.read_csv(original / "excluded_bhsd_related_positive_slices.csv")
    target = (
        source[source.slice_path.isin(excluded.slice_path)]
        .copy()
        .reset_index(drop=True)
    )
    assert (
        len(target) == 709
        and (target["any"] == 1).all()
        and (target.split == "train").all()
    )
    target.insert(0, "target_index", np.arange(len(target)))
    target["prediction_shard"] = [
        f"shards/{i // export.SHARD_SIZE:05d}.npz" for i in target.target_index
    ]
    target["prediction_offset"] = target.target_index % export.SHARD_SIZE
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.csv").exists():
        pd.testing.assert_frame_equal(pd.read_csv(OUT / "manifest.csv"), target)
    else:
        target.to_csv(OUT / "manifest.csv", index=False)
    provenance.pop("export_signature")
    provenance.update(
        target_slices=len(target),
        target_series=target.series_uid.nunique(),
        target_patients=target.patient_id.nunique(),
        excluded_bhsd_related_positive_slices=0,
        manifest_sha256=export.sha(OUT / "manifest.csv"),
        targets="709 positive slices in original classifier training studies from BHSD-related patients. Explicitly authorized same15 final teachers despite overlap; descriptive training experiment, no leakage-free evaluation claimed.",
    )
    provenance["source_hashes"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        export.sha(__file__)
    )
    provenance["export_signature"] = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    if (OUT / "provenance.json").exists():
        assert json.loads((OUT / "provenance.json").read_text()) == provenance
    else:
        export.write_json(OUT / "provenance.json", provenance)
    export.run(argparse.Namespace(output=OUT, smoke=False, rank=0, world_size=1))
    for start in range(0, len(target), export.SHARD_SIZE):
        with np.load(OUT / "shards" / f"{start // export.SHARD_SIZE:05d}.npz") as a:
            assert str(a["export_signature"]) == provenance["export_signature"]
            assert np.array_equal(
                a["target_index"],
                np.arange(start, min(start + export.SHARD_SIZE, len(target))),
            )
    export.write_json(
        OUT / "completion.json",
        dict(
            status="complete",
            target_slices=len(target),
            teachers=15,
            export_signature=provenance["export_signature"],
        ),
    )
    cache.OUT = OUT
    cache.main()
    print("709 supplemental masks verified and ready", flush=True)


if __name__ == "__main__":
    main()
