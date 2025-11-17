"""ADD-ON MODULE — NOT required for the core COMP721 submission.

SHAP-based explainability tools for matchup models (Project 2.0)."""

# Dependencies:
# - numpy
# - pandas
# - shap
# - matplotlib
# - scikit-learn (for fitted estimators)
#
# How to test this module:
# 1. Train a tree-based classifier in notebooks/core/02_game_outcome_modelling.ipynb.
# 2. Import this module from notebooks/addons/A2_model_explainability_shap.ipynb.
# 3. Call `compute_shap_values` on matchup features to obtain an explainer and SHAP values.
# 4. Use the plotting helpers to visualise overall and per-feature contributions.
#
# Advantages:
# - Provides local and global interpretability for complex tree ensembles.
# - Highlights which matchup features drive predictions, aiding transparency.
#
# Disadvantages:
# - Computing SHAP values can be expensive for large datasets/models.
# - Visualisations require careful explanation for non-technical stakeholders.

from __future__ import annotations

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> Tuple[shap.Explainer, np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a tree-based matchup model."""
    if max_samples and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.copy().reset_index(drop=True)
    X_sample = X_sample.fillna(0.0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def plot_shap_summary(
    shap_values,
    X_sample: pd.DataFrame,
    class_idx: int = 1,
    show: bool = False,
    save_path: str | None = None,
) -> None:
    """Create and optionally save a SHAP summary plot for the positive class."""
    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[class_idx], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_shap_dependence(
    shap_values,
    X_sample: pd.DataFrame,
    feature_name: str,
    class_idx: int = 1,
    interaction_index: str | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot SHAP dependence for a specific feature to show effect magnitude."""
    if feature_name not in X_sample.columns:
        raise ValueError(f"{feature_name} not found in provided features.")

    if isinstance(shap_values, list):
        shap_array = shap_values[class_idx]
    else:
        shap_array = shap_values

    if shap_array.shape[1] != X_sample.shape[1]:
        raise ValueError("Mismatch between SHAP array width and feature matrix width.")

    feature_idx = list(X_sample.columns).index(feature_name)
    feature_values = np.asarray(X_sample.iloc[:, feature_idx].values).ravel()
    shap_feature = np.asarray(shap_array[:, feature_idx]).ravel()
    n = min(feature_values.size, shap_feature.size)
    if feature_values.size != shap_feature.size:
        # When SHAP returns class-specific arrays of slightly different length (due to subsampling),
        # trim both vectors to the smallest common size for plotting.
        feature_values = feature_values[:n]
        shap_feature = shap_feature[:n]

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        feature_values,
        shap_feature,
        c=feature_values,
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP dependence: {feature_name}")
    fig.colorbar(scatter, ax=ax, label=f"{feature_name} value")

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
