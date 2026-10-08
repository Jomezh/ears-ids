"""OWNER: P4.  THE contract every council member must follow.

Layers 2-5 only ever call the methods below, so they never need to know whether a member is
XGBoost, a neural net or an autoencoder-based detector. Adding a new member = subclass Detector,
implement the 4 abstract methods, done.

Rules:
  * predict_proba() columns follow self.class_order (shared class order -> votes line up).
  * Every member does its own preprocessing inside predict_proba() (raw feature table in, probabilities out).
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ears_ids.common.config import SEED, TASKS


class Detector(ABC):
    name = "detector"          # subclasses set this; used as the artifact folder name

    def __init__(self, task="multiclass", seed=SEED, class_order=None):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        self.task, self.seed = task, seed
        if task == "binary":
            self.class_order = ["benign", "attack"]
        else:
            self.class_order = list(class_order) if class_order is not None else None

    # ---- members implement these four ------------------------------------------------------
    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame): ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Shape (n_rows, n_classes); columns in class_order."""

    @abstractmethod
    def save(self, directory=None, dataset="default") -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, directory, task="multiclass"): ...

    # ---- shared behaviour (free for every member) -------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df).argmax(axis=1)

    def predict_one(self, flow) -> dict:
        """flow: dict or Series with the raw feature values of ONE flow."""
        proba = self.predict_proba(pd.DataFrame([dict(flow)]))[0]
        idx = int(proba.argmax())
        names = self.class_order or [str(i) for i in range(len(proba))]
        return {
            "label_index": idx,
            "label": names[idx],
            "confidence": float(proba[idx]),
            "probabilities": {n: float(p) for n, p in zip(names, proba)},
        }
