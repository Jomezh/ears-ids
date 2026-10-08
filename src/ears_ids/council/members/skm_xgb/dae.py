"""OWNER: P1.  PLACEHOLDER -- replace with the real denoising autoencoder (needs torch).

Real version: train a DAE (architecture is YOUR choice -- the paper doesn't give it), then keep the
top-k input features ranked by encoder-weight magnitude. Compare your list with the paper's Table 2.
Rule: store selected features BY NAME, never by position.

The placeholder just keeps the first k columns so the pipeline runs end to end.
"""
import pandas as pd

from ears_ids.common.config import K_SELECTED, SEED


class DAESelector:
    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, X: pd.DataFrame, k: int = K_SELECTED):
        self.selected_features_ = list(X.columns[:k])      # PLACEHOLDER selection
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]
