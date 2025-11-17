"""Core modules (re-exported for compatibility with Project 2.0 add-ons)."""

from .. import data_loading, evaluation, feature_engineering, models_game_outcome, models_outliers

__all__ = [
    "data_loading",
    "feature_engineering",
    "models_outliers",
    "models_game_outcome",
    "evaluation",
]
