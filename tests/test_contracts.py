"""Do the files and columns follow the day-0 contract?"""
import json

from ears_ids.common.config import FEATURE_DTYPE, LABEL_BIN, LABEL_MULTI, feature_columns


def test_fixture_schema(splits):
    for df in splits:
        feats = feature_columns(df)
        assert len(feats) == 20
        assert all(str(df[c].dtype) == FEATURE_DTYPE for c in feats)
        assert LABEL_BIN in df.columns and LABEL_MULTI in df.columns


def test_labels_consistent_and_in_class_order(splits, class_order):
    for df in splits:
        assert set(df[LABEL_MULTI].unique()) <= set(range(len(class_order)))
        assert ((df[LABEL_MULTI] != 0).astype(int) == df[LABEL_BIN]).all()   # class 0 = normal


def test_classes_are_imbalanced(splits):
    counts = splits[0][LABEL_MULTI].value_counts()
    assert counts.max() / counts.min() > 10


def test_saved_selected_features_are_names(splits, class_order, tmp_path):
    from ears_ids.council.members.skm_xgb import SKMXGBDetector
    det = SKMXGBDetector("multiclass", class_order=class_order).fit(splits[0], splits[1])
    det.save(tmp_path)
    names = json.load(open(tmp_path / "selected_features.json"))
    assert len(names) == 15 and all(isinstance(n, str) for n in names)
    assert set(names) <= set(feature_columns(splits[0]))


def test_default_paths_point_at_repo_root():
    from ears_ids.common.config import ARTIFACTS_DIR, DATA_DIR, REPO_ROOT
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert DATA_DIR == REPO_ROOT / "data" / "processed"
    assert ARTIFACTS_DIR == REPO_ROOT / "artifacts"
