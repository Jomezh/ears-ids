"""OWNER: P3.  Working version of the metrics. P3 adds timing, per-class tables and plots.

FAR (false alarm rate) = FP / (FP + TN): how often normal traffic is wrongly flagged.
Binary: FAR of the 'attack' class. Multiclass: average of each class's one-vs-rest FAR.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate(model, X, y) -> dict:
    y = np.asarray(y)
    pred = np.asarray(model.predict(X))
    n_classes = int(getattr(model, "n_classes_", len(np.unique(y))))
    binary = n_classes == 2
    labels = list(range(n_classes))
    avg = "binary" if binary else "macro"

    cm = confusion_matrix(y, pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    far_per_class = np.divide(fp, fp + tn, out=np.zeros(len(labels)), where=(fp + tn) > 0)
    far = float(far_per_class[1]) if binary else float(far_per_class.mean())

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y, pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y, pred, average=avg, zero_division=0)),
        "far": far,
        "confusion_matrix": cm.tolist(),
    }
