"""Binary classification metrics for drone detection (positive class = drone = 1)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,          # drone recall — missed-drone rate = 1 - this
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
