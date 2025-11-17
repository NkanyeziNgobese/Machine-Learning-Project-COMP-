"""ADD-ON MODULE — NOT required for the core COMP721 submission.

Autoencoder-based anomaly detection for player features (Project 2.0)."""

# Dependencies:
# - numpy
# - pandas
# - scikit-learn
# - tensorflow / keras
# - matplotlib (optional for visualisation)
#
# How to test this module:
# 1. In notebooks/addons/A1_autoencoder_outliers_experiments.ipynb import the functions below.
# 2. Load the engineered player feature table from src.core.feature_engineering.
# 3. Train the autoencoder with `train_autoencoder`, monitor loss, and compute reconstruction errors.
# 4. Rank players via `get_top_anomalies` and compare with MAD/IsolationForest outputs.
#
# Advantages:
# - Captures non-linear relationships between player metrics beyond classical robust z-scores.
# - Provides a reconstruction-error score that can surface subtle standout performances.
#
# Disadvantages:
# - Requires additional hyperparameter tuning and compute; risk of overfitting.
# - Interpretation of reconstruction error is less intuitive than traditional statistics.

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models


def build_autoencoder_model(input_dim: int, latent_dim: int = 8) -> models.Model:
    """Build and compile a fully-connected autoencoder for player features."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    bottleneck = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs, name="player_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(
    X: pd.DataFrame,
    latent_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> Tuple[models.Model, StandardScaler, np.ndarray]:
    """Train an autoencoder on numeric player features and return reconstruction errors."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = build_autoencoder_model(input_dim=X_scaled.shape[1], latent_dim=latent_dim)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_scaled,
        X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0,
    )

    recon = model.predict(X_scaled, verbose=0)
    errors = np.mean((X_scaled - recon) ** 2, axis=1)
    return model, scaler, errors


def compute_reconstruction_errors(
    model: models.Model,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute reconstruction errors for new player data using a trained autoencoder."""
    X_scaled = scaler.transform(X.values)
    recon = model.predict(X_scaled, verbose=0)
    return np.mean((X_scaled - recon) ** 2, axis=1)


def get_top_anomalies(
    X: pd.DataFrame,
    player_id_col: pd.Series,
    errors: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return a table of the top-k anomalous players sorted by reconstruction error."""
    df = X.copy()
    df["player_id"] = player_id_col.values
    df["reconstruction_error"] = errors
    ordered_cols = ["player_id", "reconstruction_error"] + [col for col in X.columns if col not in {"player_id"}]
    return df.sort_values("reconstruction_error", ascending=False).head(top_k)[ordered_cols]
