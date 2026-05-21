"""Unit tests for the Gatheral-Oomen volume-weighted-mid LOB baseline.

Verifies the L1 imbalance formula and quantile-calibrated thresholds
produce sensible 3-class predictions on a tiny hand-built fixture.
"""
from __future__ import annotations

import numpy as np

from src.strategies.lob_classical import (
    _l1_imbalance,
    fit_gatheral_oomen_threshold,
    gatheral_oomen_predict,
)


def _window(v_ask_l1: float, v_bid_l1: float) -> np.ndarray:
    """Build a (lookback=3, 40) window whose LAST tick has the given L1 vols.

    All other features are zero; only f01 (ask vol) and f03 (bid vol) at
    the last tick matter for the L1 imbalance computation.
    """
    w = np.zeros((3, 40), dtype=np.float32)
    w[-1, 1] = v_ask_l1
    w[-1, 3] = v_bid_l1
    return w


def test_imbalance_signs():
    # bid-heavy → positive imbalance; ask-heavy → negative; balanced → 0.
    X = np.stack([
        _window(v_ask_l1=1.0, v_bid_l1=3.0),   # I = +0.5
        _window(v_ask_l1=3.0, v_bid_l1=1.0),   # I = -0.5
        _window(v_ask_l1=2.0, v_bid_l1=2.0),   # I = 0
        _window(v_ask_l1=0.0, v_bid_l1=0.0),   # safe-divide → 0
    ])
    I = _l1_imbalance(X)
    np.testing.assert_allclose(I, [0.5, -0.5, 0.0, 0.0], atol=1e-9)


def test_threshold_quantile_calibration_matches_class_frequencies():
    # 100 windows: imbalance is just np.arange so quantiles are predictable.
    # Build windows where v_bid = i, v_ask = 100-i — imbalance grows monotonically with i.
    X = np.stack([_window(v_ask_l1=float(100 - i), v_bid_l1=float(i))
                  for i in range(1, 101)])
    # Labels: 30% down (i ≤ 30), 50% stat (31 ≤ i ≤ 80), 20% up (i ≥ 81).
    y = np.zeros(100, dtype=np.int64)
    y[30:80] = 1
    y[80:] = 2

    tau_down, tau_up = fit_gatheral_oomen_threshold(X, y)
    # By construction, training-set predictions should *exactly* match the
    # class distribution: ~30 predicted down, ~20 up, ~50 stationary.
    preds = gatheral_oomen_predict(X, (tau_down, tau_up))
    bincount = np.bincount(preds, minlength=3)
    assert abs(int(bincount[0]) - 30) <= 1
    assert abs(int(bincount[2]) - 20) <= 1
    assert abs(int(bincount[1]) - 50) <= 1


def test_predict_classifies_extreme_imbalances():
    # Train on a balanced 3-class set so τ_down < 0 < τ_up.
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.5, 1.5, 90)
    asks = rng.uniform(0.5, 1.5, 90)
    X = np.stack([_window(v_ask_l1=float(a), v_bid_l1=float(b))
                  for a, b in zip(asks, bids)])
    y = np.tile([0, 1, 2], 30)
    thresholds = fit_gatheral_oomen_threshold(X, y)

    # Strongly bid-heavy and strongly ask-heavy held-out cases.
    X_test = np.stack([
        _window(v_ask_l1=0.01, v_bid_l1=10.0),   # I ≈ +1 → up
        _window(v_ask_l1=10.0, v_bid_l1=0.01),   # I ≈ -1 → down
    ])
    preds = gatheral_oomen_predict(X_test, thresholds)
    assert preds[0] == 2
    assert preds[1] == 0


def test_predict_degenerate_single_class_label():
    # If training labels are all class 1, no down/up predictions should fire.
    X = np.stack([_window(v_ask_l1=float(a), v_bid_l1=float(b))
                  for a, b in zip(range(1, 11), range(10, 0, -1))])
    y = np.ones(10, dtype=np.int64)
    thresholds = fit_gatheral_oomen_threshold(X, y)
    preds = gatheral_oomen_predict(X, thresholds)
    assert (preds == 1).all()
