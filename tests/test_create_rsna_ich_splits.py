import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_rsna_ich_splits.py"
SPEC = importlib.util.spec_from_file_location("create_rsna_ich_splits", SCRIPT_PATH)
splits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splits)


def _touch_rsna_slice(root: Path, patient: str, study: str, series: str, slice_id: str):
    path = root / "images" / patient / study / series / f"{slice_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_bhsd_exclusion_uses_available_identifiers(tmp_path):
    rsna_root = tmp_path / "rsna"
    bhsd_root = tmp_path / "bhsd"
    _touch_rsna_slice(rsna_root, "patient0", "study0", "series0", "slice0")
    _touch_rsna_slice(rsna_root, "patient1", "study1", "series1", "slice1")
    (bhsd_root / "study1" / "labels").mkdir(parents=True)

    df = splits.scan_rsna_images(rsna_root)
    matches = splits.collect_bhsd_matches(df, bhsd_root)
    df = splits.mark_bhsd_exclusions(df, matches)

    excluded = df.loc[df["excluded_bhsd"]]
    assert excluded["study_id"].tolist() == ["study1"]
    assert excluded["bhsd_match_level"].tolist() == ["study_id"]


def test_bhsd_manifest_prefers_study_over_patient_fallback(tmp_path):
    rsna_root = tmp_path / "rsna"
    bhsd_root = tmp_path / "bhsd"
    _touch_rsna_slice(rsna_root, "patient0", "study0", "series0", "slice0")
    _touch_rsna_slice(rsna_root, "patient0", "study1", "series1", "slice1")
    bhsd_root.mkdir()
    pd.DataFrame({"PatientID": ["patient0"], "SeriesID": ["study0"]}).to_csv(
        bhsd_root / "study_level.csv", index=False
    )
    pd.DataFrame({"PatientID": ["patient0"]}).to_csv(
        bhsd_root / "patient_only.csv", index=False
    )

    df = splits.scan_rsna_images(rsna_root)
    matches = splits.collect_bhsd_matches(df, bhsd_root)
    df = splits.mark_bhsd_exclusions(df, matches)

    assert matches["patient_id"] == set()
    assert df.loc[df["excluded_bhsd"], "study_id"].tolist() == ["study0"]


def test_assign_splits_has_no_patient_leakage(tmp_path):
    rsna_root = tmp_path / "rsna"
    bhsd_root = tmp_path / "bhsd"
    bhsd_root.mkdir()
    for patient_idx in range(12):
        patient = f"patient{patient_idx:02d}"
        study = f"study{patient_idx:02d}"
        _touch_rsna_slice(rsna_root, patient, study, "series0", "slice0")
        _touch_rsna_slice(rsna_root, patient, study, "series0", "slice1")

    df = splits.scan_rsna_images(rsna_root)
    df = splits.mark_bhsd_exclusions(df, splits.collect_bhsd_matches(df, bhsd_root))
    df = splits.assign_splits(df, test_studies=2, num_folds=5, seed=88)

    splits.validate_split(df, num_folds=5)
    train_patients = set(df.loc[df["split"] == "train", "patient_id"])
    test_patients = set(df.loc[df["split"] == "test", "patient_id"])
    assert train_patients.isdisjoint(test_patients)
    assert df.loc[df["split"] == "test", "study_id"].nunique() == 2
    assert set(df.loc[df["split"] == "train", "fold"].astype(int)) == set(range(5))
