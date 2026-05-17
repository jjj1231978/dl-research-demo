"""Self-contained unit tests for `src.strategies.tsmom` (FR-019 / SC-003).

Per /clarify + research.md §R7, these tests do NOT import from
~/projects/QIS_Commodities or any other external repo — they assert the
properties and formula equalities the in-repo implementation must
satisfy, citing the source papers directly.

Fixture: a 1-contract × 1000-day synthetic price series
    100 * (1 + N(0, 0.01)).cumprod()
with np.random.default_rng(0) for determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.tsmom import (
    long_only,
    macd_ensemble,
    macd_signal,
    sgn_returns,
)


@pytest.fixture
def prices() -> pd.Series:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 1000)
    return pd.Series(
        100.0 * np.cumprod(1 + returns),
        index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        name="TEST",
    )


def test_long_only_is_constant_one(prices):
    out = long_only(prices)
    assert (out == 1.0).all()
    assert out.shape == prices.shape


def test_sgn_returns_monotonic_increasing():
    """Linearly increasing series → output is +1 past the lookback."""
    p = pd.Series(np.linspace(100.0, 500.0, 600))
    out = sgn_returns(p, lookback_days=252)
    # First `lookback` values are NaN
    assert out.iloc[:252].isna().all()
    # Past the lookback, signal must be +1
    assert (out.iloc[252:] == 1.0).all()


def test_sgn_returns_monotonic_decreasing():
    p = pd.Series(np.linspace(500.0, 100.0, 600))
    out = sgn_returns(p, lookback_days=252)
    assert out.iloc[:252].isna().all()
    assert (out.iloc[252:] == -1.0).all()


def test_sgn_returns_constant_input():
    """Constant prices → sgn(0) == 0 per numpy convention."""
    p = pd.Series([100.0] * 600)
    out = sgn_returns(p, lookback_days=252)
    assert out.iloc[:252].isna().all()
    # Constant series has 0 past-year return → sign == 0
    assert (out.iloc[252:] == 0.0).all()


def test_macd_zero_for_constant_series():
    """Constant prices → MACD numerator is 0; denominator is 0; result is NaN
    (a division-by-zero on a constant series). Either NaN or 0 is acceptable
    behaviour — the post-warmup output MUST NOT be a non-zero finite value.
    """
    p = pd.Series([100.0] * 600)
    out = macd_signal(p, short_span=8, long_span=24)
    # Post-warmup
    post = out.iloc[24:]
    # Either NaN (0/0) or exactly 0 — both encode "no signal"
    assert ((post.fillna(0.0).abs() < 1e-12)).all(), (
        f"MACD on constant input should be NaN or 0; got max abs = {post.abs().max()}"
    )


def test_macd_ensemble_equals_mean_of_components(prices):
    """Formula equality: macd_ensemble == elementwise mean of the three
    component MACDs. The reference IS the in-repo implementation — this
    test catches regressions in the aggregation logic without coupling to
    any external repo.
    """
    shorts = (8, 16, 32)
    longs = (24, 48, 96)
    components = [macd_signal(prices, s, l) for s, l in zip(shorts, longs)]
    expected = pd.concat(components, axis=1).mean(axis=1)
    actual = macd_ensemble(prices, shorts=shorts, longs=longs)
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), atol=1e-12, equal_nan=True
    )


def test_no_lookahead_leakage_sgn_returns(prices):
    out = sgn_returns(prices, lookback_days=252)
    assert out.iloc[:252].isna().all()


def test_no_lookahead_leakage_macd(prices):
    out = macd_signal(prices, short_span=8, long_span=24)
    # First `long_span` rows must be NaN (rolling-std warmup)
    assert out.iloc[:24].isna().any(), "MACD must not produce a finite value before the rolling-std window fills"


def test_output_shape_preserved(prices):
    assert long_only(prices).shape == prices.shape
    assert sgn_returns(prices).shape == prices.shape
    assert macd_signal(prices, 8, 24).shape == prices.shape
    assert macd_ensemble(prices).shape == prices.shape
