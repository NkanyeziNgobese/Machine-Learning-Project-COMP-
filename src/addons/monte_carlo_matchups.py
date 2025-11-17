"""ADD-ON MODULE — NOT required for the core COMP721 submission.

Monte Carlo simulation utilities for matchup forecasting (Project 2.0)."""

# Dependencies:
# - numpy
# - pandas
# - scikit-learn (classifier/regressor interfaces)
#
# How to test this module:
# 1. Train classifier/regressor models using notebooks/core outputs.
# 2. Import this module in notebooks/addons/A3_monte_carlo_matchups.ipynb.
# 3. Build matchup feature vectors for selected team pairs.
# 4. Run the simulation helpers and verify the returned probabilities/margins.
#
# Advantages:
# - Extends binary predictions into probabilistic win estimates and expected margins.
# - Supports richer scenario analysis for hypothetical matchups.
#
# Disadvantages:
# - Requires assumptions about score/probability noise distributions.
# - Adds computational overhead and potential calibration complexity.

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin


def simulate_matchup_from_classifier(
    model: ClassifierMixin,
    x_matchup: np.ndarray,
    n_sim: int = 10_000,
    noise_std: float = 0.05,
    return_samples: bool = False,
) -> Dict[str, float]:
    """Monte Carlo sampling on classifier probabilities to estimate win likelihood."""
    x_input = x_matchup.reshape(1, -1) if x_matchup.ndim == 1 else x_matchup
    base_proba = model.predict_proba(x_input)[0, 1]
    draws = np.clip(np.random.normal(loc=base_proba, scale=noise_std, size=n_sim), 0, 1)
    result = {
        "base_proba": float(base_proba),
        "simulated_win_prob": float(draws.mean()),
        "sim_std": float(draws.std()),
        "n_sim": int(n_sim),
    }
    if return_samples:
        result["samples"] = draws
    return result


def simulate_matchup_from_regressor(
    model: RegressorMixin,
    x_matchup: np.ndarray,
    n_sim: int = 10_000,
    noise_std: float = 10.0,
    return_samples: bool = False,
) -> Dict[str, float]:
    """Monte Carlo simulation based on predicted point differential."""
    x_input = x_matchup.reshape(1, -1) if x_matchup.ndim == 1 else x_matchup
    base_diff = model.predict(x_input)[0]
    draws = np.random.normal(loc=base_diff, scale=noise_std, size=n_sim)
    result = {
        "base_point_diff": float(base_diff),
        "expected_margin": float(draws.mean()),
        "simulated_win_prob": float((draws > 0).mean()),
        "n_sim": int(n_sim),
    }
    if return_samples:
        result["samples"] = draws
    return result
