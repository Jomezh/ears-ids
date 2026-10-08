"""OWNER: P4 (you).  SKM-XGB as a council member.

Chain:  Preprocessor -> DAESelector -> (SKMSampler, training only) -> XGBoost.
One detector = one task ('binary' or 'multiclass'). Follows ears_ids.common.interface.Detector.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ears_ids.common.config import ARTIFACTS_DIR, K_SELECTED, SEED
from ears_ids.common.data import split_xy
from ears_ids.common.interface import Detector

from .dae import DAESelector
from .preprocessing import Preprocessor
from .sampling import SKMSampler
from .xgb_train import train_xgb


class SKMXGBDetector(Detector):
    name = "skm_xgb"

    def __init__(self, task="multiclass", seed=SEED, k=K_SELECTED, class_order=None):
        super().__init__(task, seed, class_order)
        self.k = k

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        _, y_train = split_xy(train, self.task)
        _, y_val = split_xy(val, self.task)

        self.preprocessor_ = Preprocessor().fit(train)                       # train only
        X_train = self.preprocessor_.transform(train)
        self.selector_ = DAESelector(seed=self.seed).fit(X_train, k=self.k)  # train only
        X_train = self.selector_.transform(X_train)

        X_bal, y_bal = SKMSampler(seed=self.seed).fit_resample(X_train, y_train)   # train only
        self.balanced_train_ = X_bal.assign(**{y_bal.name or "label": y_bal.values})

        X_val = self._prepare(val)
        self.model_, self.best_params_ = train_xgb(X_bal, y_bal, X_val, y_val, self.task, self.seed)
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.selector_.transform(self.preprocessor_.transform(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._prepare(df))

    def save(self, directory=None, dataset="default"):
        d = Path(directory) if directory else ARTIFACTS_DIR / dataset / self.name
        d.mkdir(parents=True, exist_ok=True)
        self.preprocessor_.save(d / "preprocessor.joblib")
        with open(d / "selected_features.json", "w") as f:
            json.dump(self.selector_.selected_features_, f, indent=2)       # names, not positions
        self.model_.save_model(d / f"xgb_{self.task}.json")
        self.balanced_train_.to_parquet(d / f"skm_train_{self.task}.parquet")
        with open(d / f"meta_{self.task}.json", "w") as f:
            json.dump({"task": self.task, "seed": self.seed, "k": self.k,
                       "class_order": self.class_order, "best_params": self.best_params_}, f, indent=2)
        return d

    @classmethod
    def load(cls, directory, task="multiclass"):
        d = Path(directory)
        with open(d / f"meta_{task}.json") as f:
            meta = json.load(f)
        det = cls(task=task, seed=meta["seed"], k=meta["k"], class_order=meta["class_order"])
        det.best_params_ = meta["best_params"]
        det.preprocessor_ = Preprocessor.load(d / "preprocessor.joblib")
        det.selector_ = DAESelector(seed=meta["seed"])
        with open(d / "selected_features.json") as f:
            det.selector_.selected_features_ = json.load(f)
        det.model_ = XGBClassifier()
        det.model_.load_model(d / f"xgb_{task}.json")
        return det
