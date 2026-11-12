import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from skp.toolbox.ich_paths import assert_teacher_isolation

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_teacher_guard_detects_other_study_same_patient():
    teachers = pd.DataFrame({"patient_id": ["p1", "p2"]})
    split = pd.DataFrame({"patient_id": ["p3", "p1"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="1 held-out"):
        assert_teacher_isolation(teachers, split)
    assert_teacher_isolation(teachers, split.assign(patient_id=["p3", "p4"]))


def test_patient_exclusion_prevents_cross_study_holdout():
    m = load("create_rsna_ich_splits")
    d = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p4", "p5"],
            "study_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "excluded_bhsd": [True, False, False, False, False, False],
        }
    )
    patients = set(d.loc[d.excluded_bhsd, "patient_id"])
    d["excluded_bhsd"] = d.patient_id.isin(patients)
    out = m.assign_splits(d, test_studies=1, val_studies=1, num_folds=2, seed=88)
    assert (out.loc[out.patient_id == "p1", "split"] == "excluded_bhsd").all()


def test_bhsd_raw_preparation(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import nibabel as nib

    m = load("prepare_bhsd_inputs")
    source = tmp_path / "raw"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    hu = np.full((512, 512, 3), 40, dtype=np.float32)
    mask = np.zeros_like(hu, dtype=np.uint8)
    mask[10:15, 30:34, 1] = 3
    affine = np.diag([0.5, 0.6, 5, 1])
    for folder, a in [("images", hu), ("labels", mask)]:
        nib.save(
            nib.Nifti1Image(a, affine), source / folder / "ID_patient_ID_study.nii.gz"
        )
    m.prepare(source, tmp_path / "prepared")
    d = pd.read_csv(tmp_path / "prepared/train_all_slices_png_3ch_kfold.csv")
    assert len(d) == 3 and (d.label != "empty").sum() == 1
    prep = load("prepare_bhsd_segmentation")
    result = prep.verify_orientation(d, bhsd_root=tmp_path / "prepared", nib=nib)
    assert result["exact_matches"] == 1
    with pytest.raises(FileExistsError):
        m.prepare(source, tmp_path / "prepared")


def test_cache_guard_runs_before_outputs(tmp_path, monkeypatch):
    import skp.toolbox.bhsd_convnext as m

    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"patient_id": ["p"], "fold": [0]}).to_csv(manifest, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p"],
            "study_id": ["s"],
            "split": ["test"],
            "excluded_bhsd": [False],
        }
    ).to_csv(splits, index=False)
    monkeypatch.setattr(m, "MANIFEST", manifest)
    monkeypatch.setattr(m, "configured_path", lambda key, default: splits)
    with pytest.raises(ValueError, match="held-out"):
        m.prepare_cache()


def test_pseudolabel_prepare_guard_runs_before_outputs(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    m = load("export_ich_teacher_pseudolabels")
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"patient_id": ["p"]}).to_csv(manifest, index=False)
    pd.DataFrame({"patient_id": ["p"], "split": ["test"]}).to_csv(splits, index=False)
    from types import SimpleNamespace

    monkeypatch.setattr(
        m,
        "dataset",
        lambda: SimpleNamespace(df=pd.DataFrame({"patient_id": ["p"], "any": [1]})),
    )
    monkeypatch.setattr(m, "configured_path", lambda key, default: manifest)
    monkeypatch.setattr(m.classifier_cfg, "annotations_file", str(splits))
    with pytest.raises(ValueError, match="held-out"):
        m.prepare(tmp_path / "out")
    assert not (tmp_path / "out").exists()
