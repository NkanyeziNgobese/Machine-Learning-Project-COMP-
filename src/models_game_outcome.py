# Dependencies:
# - pandas
# - numpy
# - scikit-learn
#
# How to test this module:
# 1. Generate team matchup data via `build_pairwise_matchups`.
# 2. Call `season_train_test_split` to obtain train/test DataFrames.
# 3. Use `prepare_features` plus the `train_*` helpers to fit and evaluate models inside a notebook.

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_PREFIX = "diff_"


def select_feature_columns(df: pd.DataFrame, include: Optional[Iterable[str]] = None) -> List[str]:
    """Return the numeric feature columns used for modeling."""
    if include is not None:
        return [col for col in include if col in df.columns]
    return [col for col in df.columns if col.startswith(DEFAULT_FEATURE_PREFIX)]


def season_train_test_split(
    matchups_df: pd.DataFrame,
    season_col: str = "year",
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season so that entire seasons reside in either train or test sets."""
    seasons = sorted(matchups_df[season_col].unique())
    if not seasons:
        raise ValueError("No seasons available for splitting.")
    cutoff_idx = max(1, int(len(seasons) * train_ratio))
    train_seasons = seasons[:cutoff_idx]
    test_seasons = seasons[cutoff_idx:]
    train_df = matchups_df[matchups_df[season_col].isin(train_seasons)].copy()
    test_df = matchups_df[matchups_df[season_col].isin(test_seasons)].copy()
    return train_df, test_df


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract the modeling matrix X and target vector y."""
    features = select_feature_columns(df, feature_cols)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not present.")
    X = df[features].copy()
    y = df[label_col].astype(int)
    return X, y, features


def build_logistic_regression_pipeline(
    numeric_features: Iterable[str],
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """Construct a scaling + logistic regression pipeline for interpretable baselines."""
    numeric_features = list(numeric_features)
    scaler = ColumnTransformer([("num", StandardScaler(), numeric_features)], remainder="drop")
    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=500,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline([("scaler", scaler), ("clf", clf)])


def build_gradient_boosting_model(
    **kwargs,
) -> GradientBoostingClassifier:
    """Return a GradientBoostingClassifier with sensible defaults for tabular data."""
    default_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    default_params.update(kwargs)
    return GradientBoostingClassifier(**default_params)


def build_random_forest_model(**kwargs) -> RandomForestClassifier:
    """Return a RandomForestClassifier for feature-importance inspection."""
    default_params = dict(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    default_params.update(kwargs)
    return RandomForestClassifier(**default_params)


def train_models(
    train_df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
) -> Dict[str, Pipeline | GradientBoostingClassifier | RandomForestClassifier]:
    """Train baseline and advanced models on the provided training dataframe."""
    X_train, y_train, features = prepare_features(train_df, feature_cols)
    models: Dict[str, Pipeline | GradientBoostingClassifier | RandomForestClassifier] = {}

    logit_pipeline = build_logistic_regression_pipeline(features)
    logit_pipeline.fit(X_train, y_train)
    models["logistic_regression"] = logit_pipeline

    gb_model = build_gradient_boosting_model()
    gb_model.fit(X_train, y_train)
    models["gradient_boosting"] = gb_model

    rf_model = build_random_forest_model()
    rf_model.fit(X_train, y_train)
    models["random_forest"] = rf_model

    return models
