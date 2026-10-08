"""OWNER: P2.  PLACEHOLDER -- replace the internals, keep the interface.

Real version (paper): drop redundant fields, one-hot encode categoricals, standardize.
Rules that must hold (tests check them):
  * fit() sees TRAIN ONLY.
  * transform() keeps readable column names and returns float32.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ears_ids.common.config import FEATURE_DTYPE, feature_columns


class Preprocessor:
    def fit(self, train: pd.DataFrame):
        self.columns_ = feature_columns(train)
        self.scaler_ = StandardScaler().fit(train[self.columns_])
        self.mean_ = self.scaler_.mean_
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise KeyError(f"input is missing columns: {missing}")
        out = self.scaler_.transform(df[self.columns_])
        return pd.DataFrame(out, index=df.index, columns=self.columns_).astype(FEATURE_DTYPE)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
