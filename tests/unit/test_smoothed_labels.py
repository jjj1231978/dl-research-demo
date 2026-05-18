"""Smoothed-label test for Zhang et al. 2019 Eq. 4 (FR-019).

The paper labels each tick by comparing the previous-k and next-k mean
mid-prices:

    m_-(t) = (1/k) * Σ_{i=0}^{k-1} p_{t-i}      (previous k incl. t)
    m_+(t) = (1/k) * Σ_{i=1}^{k}   p_{t+i}       (next k)
    l_t    = (m_+(t) - m_-(t)) / m_-(t)
    y_t    = 0 if l_t < -alpha   (down)
             2 if l_t >  alpha   (up)
             1 otherwise          (stationary)

This is the labelling rule baked into the FI-2010 .txt files at horizons
k ∈ {10, 20, 30, 50, 100} with paper-tuned alpha. We verify the formula
on a fixed-seed synthetic mid-price series.
"""
from __future__ import annotations

import numpy as np
import pytest


def smoothed_labels_eq4(prices: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Apply Zhang et al. 2019 Eq. 4 over a mid-price series.

    Args:
        prices: 1D float series of mid-prices.
        k: prediction horizon (e.g. 10, 20, ...).
        alpha: classification threshold.

    Returns:
        1D int array of labels in {0, 1, 2}. Length = max(0, len(prices) - 2k + 1).
        Positions where the trailing/leading window can't be formed are dropped.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    out = []
    for t in range(k - 1, n - k):
        m_minus = prices[t - k + 1: t + 1].mean()
        m_plus = prices[t + 1: t + 1 + k].mean()
        l = (m_plus - m_minus) / m_minus
        if l < -alpha:
            out.append(0)
        elif l > alpha:
            out.append(2)
        else:
            out.append(1)
    return np.asarray(out, dtype=np.int8)


def test_smoothed_labels_constant_series_all_stationary():
    """All-equal prices → m_+ == m_- → l == 0 → all class 1."""
    labels = smoothed_labels_eq4(np.full(100, 50.0), k=10, alpha=1e-4)
    assert set(labels.tolist()) == {1}


def test_smoothed_labels_upward_trend_all_up():
    """Monotone uptrend with slope >> alpha → all class 2."""
    prices = np.linspace(100, 200, 100)  # +1 per tick
    labels = smoothed_labels_eq4(prices, k=10, alpha=1e-4)
    assert set(labels.tolist()) == {2}


def test_smoothed_labels_downward_trend_all_down():
    """Monotone downtrend → all class 0."""
    prices = np.linspace(200, 100, 100)
    labels = smoothed_labels_eq4(prices, k=10, alpha=1e-4)
    assert set(labels.tolist()) == {0}


def test_smoothed_labels_fixed_seed_distribution():
    """On a fixed-seed mean-reverting noise series, all 3 classes appear."""
    rng = np.random.default_rng(42)
    prices = 100 + rng.standard_normal(500).cumsum() * 0.5
    labels = smoothed_labels_eq4(prices, k=10, alpha=1e-4)
    unique = set(labels.tolist())
    assert unique.issubset({0, 1, 2}), f"unexpected label values: {unique}"
    # On a 500-tick random walk, all 3 classes are practically certain.
    assert len(unique) == 3, f"only saw {unique}; expected all of {{0,1,2}}"


def test_smoothed_labels_length():
    """Output length matches the (k-1, n-k) sliding-window contract."""
    n = 300
    k = 20
    prices = np.linspace(0, 1, n) + 100
    labels = smoothed_labels_eq4(prices, k=k, alpha=1e-9)
    assert len(labels) == n - 2 * k + 1


@pytest.mark.parametrize("k", [10, 20, 30, 50, 100])
def test_smoothed_labels_horizons(k):
    """All paper horizons produce valid labels."""
    rng = np.random.default_rng(7)
    prices = 50 + rng.standard_normal(2 * k + 50).cumsum() * 0.01
    labels = smoothed_labels_eq4(prices, k=k, alpha=1e-5)
    assert ((labels >= 0) & (labels <= 2)).all()
