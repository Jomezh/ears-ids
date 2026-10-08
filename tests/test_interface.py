"""The council contract: any member follows Detector, and voting works across members."""
import numpy as np
import pytest

from ears_ids.common.interface import Detector
from ears_ids.council import soft_vote
from ears_ids.council.members.skm_xgb import SKMXGBDetector


@pytest.fixture(scope="module")
def det(splits, class_order):
    train, val, _ = splits
    return SKMXGBDetector("multiclass", class_order=class_order).fit(train, val)


def test_member_follows_interface(det):
    assert isinstance(det, Detector)
    assert det.name == "skm_xgb"


def test_predict_proba_columns_follow_class_order(det, splits, class_order):
    proba = det.predict_proba(splits[2])
    assert proba.shape == (len(splits[2]), len(class_order))


def test_soft_vote_of_identical_members_equals_member(det, splits):
    test = splits[2]
    voted = soft_vote([det, det], test)
    assert np.allclose(voted, det.predict_proba(test), atol=1e-6)


def test_soft_vote_rejects_mismatched_members(det, splits):
    train, val, test = splits
    binary = SKMXGBDetector("binary").fit(train, val)
    with pytest.raises(ValueError):
        soft_vote([det, binary], test)
    with pytest.raises(ValueError):
        soft_vote([det, det], test, weights=[1.0])
