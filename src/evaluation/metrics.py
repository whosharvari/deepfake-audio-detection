

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import INT_TO_LABEL


def compute_eer(y_true, y_prob) -> tuple[float, float]:
   
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def per_class_accuracy(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: dict[str, float] = {}
    for label_int, label_name in INT_TO_LABEL.items():
        mask = y_true == label_int
        if mask.sum() == 0:
            result[label_name] = float("nan")
        else:
            result[label_name] = float((y_pred[mask] == y_true[mask]).mean())
    return result


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
   
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    eer, eer_threshold = compute_eer(y_true, y_prob)
    multi_class = len(np.unique(y_true)) > 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if multi_class else float("nan"),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_class_accuracy": per_class_accuracy(y_true, y_pred),
    }


def format_metrics_report(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer"):
        lines.append(f"  {key:>10s}: {metrics[key]:.4f}")
    lines.append(f"  eer_threshold: {metrics['eer_threshold']:.4f}")

    pca = metrics["per_class_accuracy"]
    lines.append(f"  per-class accuracy: real={pca.get('real', float('nan')):.4f}, "
                  f"fake={pca.get('fake', float('nan')):.4f}")

    cm = metrics["confusion_matrix"]
    lines.append("  confusion matrix [rows=true, cols=pred] (real, fake):")
    lines.append(f"    real -> {cm[0]}")
    lines.append(f"    fake -> {cm[1]}")

    return "\n".join(lines)
