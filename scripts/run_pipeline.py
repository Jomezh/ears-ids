"""OWNER: P4.  Run the whole chain on a dataset, evaluate, save artifacts.

    python scripts/run_pipeline.py --dataset fixture
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ears_ids.common.config import SEED, TASKS
from ears_ids.common.data import load_class_order, load_dataset
from ears_ids.council.members.skm_xgb import SKMXGBDetector
from ears_ids.common.metrics import evaluate
from ears_ids.common.data import split_xy

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="fixture")
p.add_argument("--split", default="paper", choices=["paper", "official"])
a = p.parse_args()

train, val, test = load_dataset(a.dataset, a.split)
classes = load_class_order(a.dataset)
for task in TASKS:
    det = SKMXGBDetector(task=task, seed=SEED, class_order=classes).fit(train, val)
    X_test, y_test = split_xy(test, task)
    m = evaluate(det.model_, det._prepare(test), y_test)
    out = det.save(dataset=a.dataset)
    print(f"[{task}] acc={m['accuracy']:.3f} f1={m['f1']:.3f} far={m['far']:.3f}  -> saved to {out}")
