"""OWNER: P2. Data loading. (Working version for the fixture; P2 extends it for NSL-KDD / UNSW-NB15.)

Layout:  data/processed/{dataset}/{train,val,test}.parquet   <- the paper's split ("paper")
         data/processed/{dataset}/official/{train,val,test}.parquet  <- official partition ("official")
         data/processed/{dataset}/class_order.json           <- list of class names; label_multi indexes into it
"""
import json
from pathlib import Path

import pandas as pd

from ears_ids.common.config import DATA_DIR, feature_columns, label_column


def _base(root):
    return Path(root) if root else DATA_DIR


def dataset_dir(name, split="paper", root=None):
    if split not in ("paper", "official"):
        raise ValueError("split must be 'paper' or 'official'")
    d = _base(root) / name
    return d if split == "paper" else d / "official"


def load_dataset(name, split="paper", root=None):
    """Return (train, val, test) DataFrames: feature columns + label_bin + label_multi."""
    d = dataset_dir(name, split, root)
    return tuple(pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test"))


def load_class_order(name, root=None):
    with open(_base(root) / name / "class_order.json") as f:
        return json.load(f)


def split_xy(df, task):
    """Features (DataFrame) and labels (Series) for 'binary' or 'multiclass'."""
    return df[feature_columns(df)], df[label_column(task)]
