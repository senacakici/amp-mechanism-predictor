"""
metrics.py
----------
Multi-label classification metrics for AMP mechanism prediction.

Standard accuracy is uninformative for multi-label problems.
We report: Hamming loss, subset accuracy, macro/micro F1,
label ranking average precision, and per-label breakdown.
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    label_ranking_average_precision_score,
    coverage_error,
    classification_report,
)


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Comprehensive multi-label evaluation.

    Args:
        y_true:      True binary label matrix (n_samples, n_labels)
        y_pred:      Predicted binary label matrix (n_samples, n_labels)
        y_score:     Predicted probability scores (optional, enables LRAP/coverage)
        label_names: List of label names for per-label reporting

    Returns:
        Dict of metric name → value
    """
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    metrics = {}

    # Core multi-label metrics
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["samples_f1"] = f1_score(y_true, y_pred, average="samples", zero_division=0)

    # Per-label F1
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_label_f1"] = {
        label_names[i]: float(per_label_f1[i]) for i in range(n_labels)
    }

    # Label prevalence
    metrics["label_prevalence"] = {
        label_names[i]: float(y_true[:, i].mean()) for i in range(n_labels)
    }

    # Ranking metrics (require probability scores)
    if y_score is not None:
        metrics["lrap"] = label_ranking_average_precision_score(y_true, y_score)
        metrics["coverage_error"] = coverage_error(y_true, y_score)
    else:
        # Use predictions as proxy scores when probabilities unavailable
        metrics["lrap"] = label_ranking_average_precision_score(
            y_true, y_pred.astype(float)
        )
        metrics["coverage_error"] = float("nan")

    # Label cardinality stats
    pred_cardinality = y_pred.sum(axis=1).mean()
    true_cardinality = y_true.sum(axis=1).mean()
    metrics["true_label_cardinality"] = float(true_cardinality)
    metrics["pred_label_cardinality"] = float(pred_cardinality)

    return metrics


def print_report(metrics: Dict, title: str = "Multi-Label Evaluation") -> None:
    """Pretty-print evaluation metrics."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    print("\nOverall Metrics:")
    overall_keys = [
        "hamming_loss", "subset_accuracy", "macro_f1",
        "micro_f1", "samples_f1", "lrap",
        "true_label_cardinality", "pred_label_cardinality",
    ]
    for k in overall_keys:
        v = metrics.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            print(f"  {k:<35} {v:.4f}")

    print("\nPer-Label F1 Score:")
    per_label = metrics.get("per_label_f1", {})
    prevalence = metrics.get("label_prevalence", {})
    for label, f1 in per_label.items():
        prev = prevalence.get(label, 0)
        bar = "█" * int(f1 * 20)
        print(f"  {label:<30} {f1:.3f}  {bar:<20}  (prev: {prev:.2f})")
    print()


if __name__ == "__main__":
    # Smoke test with random predictions
    np.random.seed(42)
    n_samples, n_labels = 100, 8
    y_true = (np.random.rand(n_samples, n_labels) > 0.7).astype(int)
    y_pred = (np.random.rand(n_samples, n_labels) > 0.6).astype(int)

    label_names = [
        "membrane_disruption", "membrane_depolarization",
        "cell_wall_synthesis", "protein_synthesis",
        "dna_rna_targeting", "cell_division",
        "metabolic_disruption", "immunomodulatory",
    ]

    metrics = evaluate_multilabel(y_true, y_pred, label_names=label_names)
    print_report(metrics, title="Smoke Test — Random Predictions")
