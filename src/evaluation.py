# Dependencies:
# - pandas
# - numpy
# - matplotlib
# - seaborn
# - scikit-learn
#
# How to use this module:
# 1. Fit classifiers using helpers from `src.models_game_outcome`.
# 2. Call `evaluate_classifier` to obtain metrics and predictions.
# 3. Use the plotting utilities to save confusion matrices / ROC curves into the `figures/` folder.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics


plt.style.use("seaborn-v0_8")


def ensure_directory(path: Path | str) -> Path:
    """Create parent directories for the output path if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y_true: pd.Series,
    average: str = "binary",
) -> Dict[str, float]:
    """Compute common classification metrics for a fitted estimator."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = y_pred

    results = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    results["y_pred"] = y_pred
    results["y_score"] = y_score
    return results


def metrics_to_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert a dictionary of metric dictionaries into a tidy DataFrame."""
    rows = []
    for model_name, metrics_dict in results.items():
        metric_row = {k: v for k, v in metrics_dict.items() if k not in {"y_pred", "y_score"}}
        metric_row["model"] = model_name
        rows.append(metric_row)
    df = pd.DataFrame(rows).set_index("model")
    return df


def save_metrics_table(metrics_df: pd.DataFrame, path: Path | str) -> None:
    """Persist a metrics DataFrame as CSV."""
    path = ensure_directory(path)
    metrics_df.to_csv(path, float_format="%.4f")


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Tuple[str, str] = ("Team B wins", "Team A wins"),
    title: str = "Confusion Matrix",
    save_path: Optional[Path | str] = None,
) -> plt.Axes:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        out_path = ensure_directory(save_path)
        fig.savefig(out_path, dpi=300)
    return ax


def plot_roc_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    label: str,
    title: str = "ROC Curve",
    save_path: Optional[Path | str] = None,
) -> plt.Axes:
    """Plot ROC curve for classifier probabilities."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        out_path = ensure_directory(save_path)
        fig.savefig(out_path, dpi=300)
    return ax


def plot_feature_importances(
    model,
    feature_names: Iterable[str],
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[Path | str] = None,
) -> plt.Axes:
    """Visualise feature importance scores for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose feature_importances_.")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    sorted_features = np.array(list(feature_names))[order]
    sorted_values = importances[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sorted_values, y=sorted_features, ax=ax, palette="viridis")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        out_path = ensure_directory(save_path)
        fig.savefig(out_path, dpi=300)
    return ax
