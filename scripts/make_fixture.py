"""OWNER: P4.  Build a tiny fake dataset in the contract format so everyone can start on day 1.

    python scripts/make_fixture.py            # writes data/processed/fixture/

~5000 rows, 20 float32 features, 4 imbalanced classes (class 0 = 'normal').
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ears_ids.common.config import DATA_DIR, LABEL_BIN, LABEL_MULTI, SEED

CLASS_ORDER = ["normal", "attack_a", "attack_b", "attack_c"]


def make_fixture(out_root=None, n=5000, seed=SEED):
    out = Path(out_root) if out_root else DATA_DIR
    d = out / "fixture"
    d.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=4, n_classes=4,
        weights=[0.70, 0.20, 0.07, 0.03], class_sep=1.5, random_state=seed,
    )
    df = pd.DataFrame(X.astype("float32"), columns=[f"feat_{i:02d}" for i in range(20)])
    df[LABEL_MULTI] = y.astype("int64")
    df[LABEL_BIN] = (y != 0).astype("int64")

    train, rest = train_test_split(df, test_size=0.30, stratify=df[LABEL_MULTI], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[LABEL_MULTI], random_state=seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_parquet(d / f"{name}.parquet")
    with open(d / "class_order.json", "w") as f:
        json.dump(CLASS_ORDER, f)
    return d


if __name__ == "__main__":
    print("fixture written to", make_fixture())
