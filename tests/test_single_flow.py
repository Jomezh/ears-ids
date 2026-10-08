"""One flow in -> one decision out (what Layers 4 and 5 will do), and save/load keeps behaviour."""
import numpy as np

from ears_ids.council.members.skm_xgb import SKMXGBDetector
from ears_ids.common.config import feature_columns


def test_single_flow_matches_batch(splits, class_order):
    train, val, test = splits
    det = SKMXGBDetector("multiclass", class_order=class_order).fit(train, val)
    row = test.iloc[0]
    out = det.predict_one(row[feature_columns(test)])
    assert out["label"] in class_order
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4
    assert out["label_index"] == int(det.predict(test.iloc[[0]])[0])


def test_binary_detector_names(splits):
    train, val, test = splits
    det = SKMXGBDetector("binary").fit(train, val)
    out = det.predict_one(test.iloc[0][feature_columns(test)])
    assert out["label"] in ("benign", "attack")


def test_save_load_roundtrip(splits, class_order, tmp_path):
    train, val, test = splits
    det = SKMXGBDetector("multiclass", class_order=class_order).fit(train, val)
    det.save(tmp_path)
    loaded = SKMXGBDetector.load(tmp_path, task="multiclass")
    assert np.allclose(det.predict_proba(test), loaded.predict_proba(test), atol=1e-6)
    assert loaded.class_order == class_order
