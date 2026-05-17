"""Unit tests for `src.strategies.vol_targeting` (FR-020).

The vol-targeting helper is a stochastic transform — realised σ of the
output won't equal σ_target exactly because EWMA-std is itself an
estimator. We assert a 5% tolerance band on a 1000-day fixture (paper-
sized window), plus finite-output sanity at the extremes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.vol_targeting import vol_target


@pytest.fixture
def daily_returns() -> pd.Series:
    """Synthetic daily return series — 1000 days, σ_daily = 0.01 (≈ 15.8% annualised)."""
    rng = np.random.default_rng(0)
    return pd.Series(
        rng.normal(0, 0.01, 1000),
        index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        name="ret",
    )


def test_vol_targeting_hits_target(daily_returns):
    """σ_target = 0.15 produces a scaled-return series whose realised
    annualised vol lands in [0.142, 0.158] — a 5% tolerance band (band
    width matches the EWMA estimator's standard error on N=1000 days).
    """
    scaled = vol_target(daily_returns, target_vol=0.15, ewma_span=60)
    # Drop EWMA warmup rows; assert post-warmup realised vol matches target
    post = scaled.iloc[120:].dropna()
    realised = post.std() * np.sqrt(252)
    assert 0.142 <= realised <= 0.158, (
        f"realised annualised vol {realised:.4f} outside [0.142, 0.158] tolerance"
    )


def test_vol_targeting_extreme_low(daily_returns):
    """σ_target = 0.01 → tiny scaling factor; output must still be finite."""
    scaled = vol_target(daily_returns, target_vol=0.01, ewma_span=60)
    post = scaled.iloc[120:].dropna()
    assert np.isfinite(post.to_numpy()).all()
    realised = post.std() * np.sqrt(252)
    # 5% tolerance band relative to target
    assert 0.0095 <= realised <= 0.0105, (
        f"realised {realised:.5f} outside [0.0095, 0.0105]"
    )


def test_vol_targeting_extreme_high(daily_returns):
    """σ_target = 1.0 → large scaling factor; output must still be finite."""
    scaled = vol_target(daily_returns, target_vol=1.0, ewma_span=60)
    post = scaled.iloc[120:].dropna()
    assert np.isfinite(post.to_numpy()).all()
    realised = post.std() * np.sqrt(252)
    assert 0.95 <= realised <= 1.05, (
        f"realised {realised:.4f} outside [0.95, 1.05]"
    )


def test_vol_targeting_rejects_invalid_target(daily_returns):
    with pytest.raises(ValueError):
        vol_target(daily_returns, target_vol=0.0)
    with pytest.raises(ValueError):
        vol_target(daily_returns, target_vol=-0.1)


def test_vol_targeting_rejects_invalid_span(daily_returns):
    with pytest.raises(ValueError):
        vol_target(daily_returns, ewma_span=1)
