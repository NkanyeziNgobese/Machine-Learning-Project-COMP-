# Dependencies:
# - pandas
# - numpy
# - scipy
# - scikit-learn
#
# How to use this module:
# 1. Import the desired helper (e.g., `from src.models_outliers import run_isolation_forest`).
# 2. Pass in the engineered player feature DataFrame plus the feature columns you wish to analyse.
# 3. Examine the returned scores/flags and pair with visualisations or tables inside the notebooks.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class OutlierResult:
    """Container for storing outlier scores and flags."""

    dataframe: pd.DataFrame
    score_column: str
    flag_column: str


def _validate_features(df: pd.DataFrame, features: Iterable[str]) -> List[str]:
    """Ensure the feature columns exist."""
    features = list(features)
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return features


def compute_robust_z_scores(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Compute Median Absolute Deviation (MAD) based z-scores for the selected features."""
    features = _validate_features(df, features)
    score_df = df.copy()
    for feature in features:
        values = score_df[feature].astype(float)
        median = np.median(values)
        mad = stats.median_abs_deviation(values, scale="normal")
        if mad == 0:
            # When variance is ultra-low, revert to standard deviation to avoid division by zero.
            mad = np.std(values) or 1.0
        score_df[f"{feature}_robust_z"] = (values - median) / mad
    return score_df


def detect_outliers_via_mad(
    df: pd.DataFrame,
    features: Iterable[str],
    threshold: float = 3.5,
) -> OutlierResult:
    """Flag rows whose average absolute robust z-score exceeds the threshold."""
    score_df = compute_robust_z_scores(df, features)
    robust_cols = [col for col in score_df.columns if col.endswith("_robust_z")]
    score_df["mad_score"] = score_df[robust_cols].abs().mean(axis=1)
    score_df["is_outlier_mad"] = score_df["mad_score"] >= threshold
    return OutlierResult(score_df, score_column="mad_score", flag_column="is_outlier_mad")


def run_isolation_forest(
    df: pd.DataFrame,
    features: Iterable[str],
    contamination: float = 0.02,
    random_state: int = 42,
) -> OutlierResult:
    """Fit IsolationForest on the feature space and return anomaly scores and flags."""
    features = _validate_features(df, features)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=400,
        max_samples="auto",
    )
    model.fit(X)
    scores = model.decision_function(X)
    predictions = model.predict(X)  # -1 indicates anomaly
    score_df = df.copy()
    score_df["iforest_score"] = scores
    score_df["is_outlier_iforest"] = predictions == -1
    return OutlierResult(score_df, score_column="iforest_score", flag_column="is_outlier_iforest")


def run_local_outlier_factor(
    df: pd.DataFrame,
    features: Iterable[str],
    n_neighbors: int = 20,
    contamination: float = 0.02,
) -> OutlierResult:
    """Compute Local Outlier Factor scores to capture density-based anomalies."""
    features = _validate_features(df, features)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    predictions = lof.fit_predict(X)
    negative_factor = lof.negative_outlier_factor_
    score_df = df.copy()
    score_df["lof_score"] = negative_factor
    score_df["is_outlier_lof"] = predictions == -1
    return OutlierResult(score_df, score_column="lof_score", flag_column="is_outlier_lof")


def summarize_top_outliers(
    result: OutlierResult,
    identifier_cols: Iterable[str],
    top_n: int = 15,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return a table of the top-N outliers according to the provided score."""
    df = result.dataframe.copy()
    identifier_cols = list(identifier_cols)
    display_cols = identifier_cols + [result.score_column, result.flag_column]
    # Higher MAD scores mean more anomalous, whereas IsolationForest scores behave oppositely by default.
    sorted_df = df.sort_values(result.score_column, ascending=ascending)
    return sorted_df[display_cols].head(top_n)
