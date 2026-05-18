"""Classical LOB baselines (Phase 3).

Self-contained per Constitution VI. Currently exposes a thin sklearn LDA
wrapper — the deep baselines (MLP, CNN-I, CNN-II, LSTM) live in
`src/models/deeplob.py` because they require PyTorch + Modal training.

LDA is a closed-form fit (no SGD, no GPU) so it lives on the CPU side.
"""
from __future__ import annotations

import numpy as np


def fit_lda(X_train: np.ndarray, y_train: np.ndarray):
    """Fit Linear Discriminant Analysis on flattened LOB features.

    Args:
        X_train: (n_samples, n_features) — typically (N, T*40) flattened
            LOB lookback windows.
        y_train: (n_samples,) — 3-class labels in {0, 1, 2}.

    Returns:
        Fitted `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model


def predict_lda(model, X_test: np.ndarray) -> np.ndarray:
    """Predict 3-class labels for a fitted LDA model. Returns (n_samples,)."""
    return model.predict(X_test)


def predict_proba_lda(model, X_test: np.ndarray) -> np.ndarray:
    """Predict 3-class probabilities. Returns (n_samples, 3)."""
    return model.predict_proba(X_test)
