from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy, recall


def pawpularity_metrics(y_true, y_prob) -> dict[str, float]:
    metrics = {"log_loss": float(accuracy(y_true, y_prob))}
    metrics["recall"] = float(recall(y_true, y_prob))
    return metrics


def metric_score(metric_fn: Any, y_true, y_pred):
    return metric_fn(y_true, y_pred)
