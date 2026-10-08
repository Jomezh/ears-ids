"""OWNER: P4 (later: the Thompson-sampling / bandit weighting idea goes here).

PLACEHOLDER: plain (weighted) average of every member's probabilities.
Works with ANY members that follow the Detector interface.
"""
import numpy as np


def soft_vote(detectors, df, weights=None) -> np.ndarray:
    if not detectors:
        raise ValueError("need at least one detector")
    first = detectors[0]
    for d in detectors:
        if d.task != first.task or d.class_order != first.class_order:
            raise ValueError("all council members must share the same task and class_order")
    w = np.ones(len(detectors)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(detectors):
        raise ValueError("one weight per detector")
    w = w / w.sum()
    return sum(wi * d.predict_proba(df) for wi, d in zip(w, detectors))
