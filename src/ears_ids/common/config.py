"""Shared constants. CONTRACT FILE: change only with team agreement (P4 reviews)."""
from pathlib import Path

SEED = 42                      # one seed, everywhere
FEATURE_DTYPE = "float32"      # all feature columns
K_SELECTED = 15                # DAE keeps the top-15 features (paper)
SKM_K_CLUSTERS = 10            # K-means k used to undersample majority classes (paper)

LABEL_BIN = "label_bin"        # 0 = benign, 1 = attack
LABEL_MULTI = "label_multi"    # integer index into class_order.json
LABEL_COLUMNS = (LABEL_BIN, LABEL_MULTI)
TASKS = ("binary", "multiclass")

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/ears_ids/common/config.py -> repo root
DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def feature_columns(df):
    """Every column that is not a label. Order is preserved."""
    return [c for c in df.columns if c not in LABEL_COLUMNS]


def label_column(task):
    if task == "binary":
        return LABEL_BIN
    if task == "multiclass":
        return LABEL_MULTI
    raise ValueError(f"task must be one of {TASKS}, got {task!r}")
