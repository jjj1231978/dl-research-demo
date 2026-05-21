"""Classical LOB baselines (Phase 3).

Self-contained per Constitution VI. The deep baselines (MLP, CNN-I,
CNN-II, LSTM) live in `src/models/deeplob.py` because they require
PyTorch + Modal training. Closed-form CPU baselines live here:

- ``fit_lda`` / ``predict_lda`` — Linear Discriminant Analysis.
- ``fit_gatheral_oomen_threshold`` / ``gatheral_oomen_predict`` —
  Level-1 depth-imbalance heuristic derived from Gatheral & Oomen
  (2010) volume-weighted mid-price. P_VW = (V_b·P_a + V_a·P_b)/(V_b+V_a)
  so sign(P_VW − P_mid) = sign(V_b − V_a) at L1. Discretized to 3
  classes via quantile-calibrated thresholds on the training labels.
"""
from __future__ import annotations

import numpy as np

# FI-2010 level-1 column indices: f00=ask price, f01=ask vol,
# f02=bid price, f03=bid vol (confirmed in scripts/fetch_lob_fi2010.py).
_F_ASK_VOL_L1 = 1
_F_BID_VOL_L1 = 3


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


def _l1_imbalance(X: np.ndarray) -> np.ndarray:
    """Level-1 depth imbalance at the LAST tick of each lookback window.

    I = (V_bid - V_ask) / (V_bid + V_ask).  Sign matches sign of
    (microprice − mid) per Gatheral & Oomen (2010). Returns shape (n,).
    """
    last = X[:, -1, :]
    v_ask = last[:, _F_ASK_VOL_L1].astype(np.float64)
    v_bid = last[:, _F_BID_VOL_L1].astype(np.float64)
    denom = v_bid + v_ask
    return np.where(denom > 0, (v_bid - v_ask) / np.maximum(denom, 1e-12), 0.0)


def fit_gatheral_oomen_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    """Calibrate (τ_down, τ_up) by matching training class frequencies.

    Picks τ_down as the p_down-quantile of training imbalance (so a fraction
    p_down of windows fall below it ⇒ predicted "down"), and τ_up as the
    (1-p_up)-quantile (so a fraction p_up fall above ⇒ predicted "up"). This
    keeps the predicted class distribution close to the realized one and
    leaves no free hyperparameter.

    Args:
        X_train: (n, lookback, 40) raw lookback windows.
        y_train: (n,) 3-class labels in {0, 1, 2}.

    Returns:
        (τ_down, τ_up) thresholds for ``gatheral_oomen_predict``.
    """
    I = _l1_imbalance(X_train)
    p_down = float(np.mean(y_train == 0))
    p_up = float(np.mean(y_train == 2))
    tau_down = float(np.quantile(I, p_down)) if p_down > 0 else float(I.min() - 1.0)
    tau_up = float(np.quantile(I, 1.0 - p_up)) if p_up > 0 else float(I.max() + 1.0)
    return tau_down, tau_up


def gatheral_oomen_predict(
    X_test: np.ndarray,
    thresholds: tuple[float, float],
) -> np.ndarray:
    """Predict 3-class direction from L1 imbalance vs calibrated thresholds.

    `I < τ_down ⇒ 0 (down)`, `I > τ_up ⇒ 2 (up)`, else `1 (stationary)`.
    """
    tau_down, tau_up = thresholds
    I = _l1_imbalance(X_test)
    preds = np.full(len(X_test), 1, dtype=np.int64)
    preds[I < tau_down] = 0
    preds[I > tau_up] = 2
    return preds
