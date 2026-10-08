"""Fitting must only ever see TRAIN rows. We spy on every fit step and check what it received."""
import numpy as np
import pytest

from ears_ids.council.members.skm_xgb import SKMXGBDetector
from ears_ids.council.members.skm_xgb.dae import DAESelector
from ears_ids.council.members.skm_xgb.preprocessing import Preprocessor
from ears_ids.council.members.skm_xgb.sampling import SKMSampler


def test_preprocessor_stats_come_from_train_only(splits):
    train, val, test = splits
    pre = Preprocessor().fit(train)
    cols = pre.columns_
    assert np.allclose(pre.mean_, train[cols].mean().values, atol=1e-4)
    combined = np.concatenate([train[cols].values, test[cols].values]).mean(axis=0)
    assert not np.allclose(pre.mean_, combined, atol=1e-6)      # would match if test leaked in


def test_every_fit_step_sees_train_rows_only(splits, class_order, monkeypatch):
    train, val, test = splits
    seen = {}

    def spy(cls, method, key, nrows):
        orig = getattr(cls, method)

        def wrapped(self, X, *a, **kw):
            seen[key] = nrows(X)
            return orig(self, X, *a, **kw)
        monkeypatch.setattr(cls, method, wrapped)

    spy(Preprocessor, "fit", "preprocessor", len)
    spy(DAESelector, "fit", "dae", len)
    # sampler gets train-sized input; its output is allowed to be a different size
    spy(SKMSampler, "fit_resample", "sampler", len)

    SKMXGBDetector("multiclass", class_order=class_order).fit(train, val)
    assert seen["preprocessor"] == len(train)
    assert seen["dae"] == len(train)
    assert seen["sampler"] == len(train)
