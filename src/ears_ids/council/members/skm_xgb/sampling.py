"""OWNER: P1.  FIRST-CUT version -- P1 tunes it to match the paper exactly and logs differences.

Idea: classes are very uneven. Shrink big classes (K-means undersampling), grow small classes (SMOTE),
so both end up near the same size (here: the median class size).
Rule: used on the TRAINING set only -- never val/test.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

from ears_ids.common.config import FEATURE_DTYPE, SEED, SKM_K_CLUSTERS


class SKMSampler:
    def __init__(self, k_clusters: int = SKM_K_CLUSTERS, seed: int = SEED):
        self.k_clusters = k_clusters
        self.seed = seed

    def _kmeans_undersample(self, Xc: pd.DataFrame, target: int) -> pd.DataFrame:
        """Cluster one big class, then draw from each cluster in proportion to its size."""
        km = KMeans(n_clusters=min(self.k_clusters, len(Xc)), n_init=10, random_state=self.seed).fit(Xc)
        rng = np.random.default_rng(self.seed)
        chosen = []
        for c in range(km.n_clusters):
            members = np.where(km.labels_ == c)[0]
            take = min(len(members), max(1, round(target * len(members) / len(Xc))))
            chosen.extend(rng.choice(members, take, replace=False))
        return Xc.iloc[sorted(chosen)]

    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        counts = y.value_counts()
        target = int(counts.median())

        parts_X, parts_y = [], []
        for cls, n in counts.items():
            Xc = X[y == cls]
            if n > target:
                Xc = self._kmeans_undersample(Xc, target)
            parts_X.append(Xc)
            parts_y.append(pd.Series([cls] * len(Xc)))
        X_u = pd.concat(parts_X, ignore_index=True)
        y_u = pd.concat(parts_y, ignore_index=True)

        small = {cls: target for cls, n in y_u.value_counts().items() if n < target}
        if small:
            k_nn = max(1, min(5, int(y_u.value_counts().min()) - 1))
            X_u, y_u = SMOTE(sampling_strategy=small, k_neighbors=k_nn, random_state=self.seed).fit_resample(X_u, y_u)
        return X_u.astype(FEATURE_DTYPE).reset_index(drop=True), pd.Series(np.asarray(y_u), name=y.name)
