"""Same seed, same data -> exactly the same predictions."""
import numpy as np

from ears_ids.council.members.skm_xgb import SKMXGBDetector


def test_same_seed_same_output(splits, class_order):
    train, val, test = splits
    a = SKMXGBDetector("multiclass", seed=42, class_order=class_order).fit(train, val)
    b = SKMXGBDetector("multiclass", seed=42, class_order=class_order).fit(train, val)
    assert np.allclose(a.predict_proba(test), b.predict_proba(test))
    assert (a.balanced_train_.values == b.balanced_train_.values).all()
