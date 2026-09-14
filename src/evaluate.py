"""Metrics that matter under 0.17% fraud. Accuracy is not one of them."""

from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_free(y_true, proba) -> dict:
    return {
        "auprc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Pick the probability cutoff that maximises F1 on this split (use val, not test)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    if len(thr) == 0:
        return 0.5, 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(f1.argmax())
    return float(thr[i]), float(f1[i])


def at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def full_report(y_true, proba, threshold: float) -> dict:
    out = threshold_free(y_true, proba)
    out.update(at_threshold(y_true, proba, threshold))
    return out
