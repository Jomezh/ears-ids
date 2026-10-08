"""OWNER: P3.  FIRST-CUT version -- P3 replaces the fixed params with random-search CV
using the paper's ranges (n_estimators, learning_rate, max_depth) and returns the best params.
"""
from xgboost import XGBClassifier

from ears_ids.common.config import SEED, TASKS


def train_xgb(X_bal, y_bal, X_val, y_val, task: str, seed: int = SEED):
    """Return (fitted XGBClassifier, params dict). y must be integer-coded 0..n-1."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4}   # PLACEHOLDER (no search yet)
    model = XGBClassifier(**params, random_state=seed, n_jobs=1, tree_method="hist")  # n_jobs=1 -> deterministic
    model.fit(X_bal, y_bal, eval_set=[(X_val, y_val)], verbose=False)
    return model, params
