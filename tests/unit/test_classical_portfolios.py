"""Unit tests for `src.strategies.portfolios` (FR-019).

Self-contained per Constitution VI — no imports from external repos.
Synthetic fixture: 252 days × 8 assets, fixed seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.strategies.portfolios import (
    diversity_weighted,
    equal_weight,
    fixed_allocation,
    max_diversification,
    min_variance,
)


@pytest.fixture
def returns_8x252() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, (252, 8))


@pytest.fixture
def homogeneous_returns_8x252() -> np.ndarray:
    """Identical vols, identical pairwise correlation = 0.3 — symmetric."""
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 252) * np.sqrt(0.3)
    idiosync = rng.normal(0, 0.01, (252, 8)) * np.sqrt(0.7)
    return common[:, None] + idiosync


def _assert_simplex(w: np.ndarray) -> None:
    assert w.shape == (w.shape[0],), f"weights must be 1-D; got {w.shape}"
    assert abs(w.sum() - 1.0) < 1e-6, f"weights must sum to 1; got {w.sum():.6f}"
    assert (w >= -1e-9).all(), f"long-only violated: min={w.min():.6f}"
    assert (w <= 1.0 + 1e-9).all(), f"weight > 1: max={w.max():.6f}"
    assert np.isfinite(w).all()


def test_equal_weight_invariants():
    for n in (2, 4, 20):
        w = equal_weight(n)
        _assert_simplex(w)
        assert np.allclose(w, 1.0 / n)


def test_equal_weight_rejects_invalid():
    with pytest.raises(ValueError):
        equal_weight(0)


def test_min_variance_simplex(returns_8x252):
    _assert_simplex(min_variance(returns_8x252))


def test_max_diversification_simplex(returns_8x252):
    _assert_simplex(max_diversification(returns_8x252))


def test_min_variance_lower_vol_than_equal_weight(returns_8x252):
    """On any fixture, min-variance produces ≤ equal-weight realised vol."""
    mv = min_variance(returns_8x252)
    ew = equal_weight(returns_8x252.shape[1])
    vol_mv = float(np.std(returns_8x252 @ mv))
    vol_ew = float(np.std(returns_8x252 @ ew))
    assert vol_mv <= vol_ew + 1e-6, (
        f"min_variance vol {vol_mv:.6f} should be ≤ equal_weight {vol_ew:.6f}"
    )


def test_max_diversification_reduces_to_equal_weight_when_symmetric(
    homogeneous_returns_8x252,
):
    """With identical stdevs and equal pairwise correlations, max-div
    converges close to equal weight (closed-form result, modulo SLSQP +
    LedoitWolf shrinkage noise — exact symmetry is broken because the
    realised sample covariance is not perfectly symmetric).
    """
    md = max_diversification(homogeneous_returns_8x252)
    ew = equal_weight(homogeneous_returns_8x252.shape[1])
    # Generous tolerance: any solution that's qualitatively diversified
    # (i.e., no single weight dominates) and close to 1/N counts.
    assert md.max() < 0.30, f"max weight {md.max():.3f} too concentrated"
    assert md.min() > 0.05, f"min weight {md.min():.3f} too small"
    # Distance from 1/N stays bounded — generous tolerance for SLSQP slack
    np.testing.assert_allclose(md, ew, atol=0.15)


def test_diversity_weighted_formula_equality():
    """Hand-computed verification: μ = [0.4, 0.2, 0.2, 0.2], p=0.5
    → w_i ∝ sqrt(μ_i). Normalised closed form.
    """
    mcaps = np.array([400.0, 200.0, 200.0, 200.0])
    w = diversity_weighted(mcaps, p=0.5)
    mu = mcaps / mcaps.sum()
    expected = np.sqrt(mu) / np.sqrt(mu).sum()
    np.testing.assert_allclose(w, expected, atol=1e-12)
    _assert_simplex(w)


def test_diversity_weighted_p_zero_is_equal_weight():
    mcaps = np.array([1e10, 5e9, 2e10, 8e9])
    w = diversity_weighted(mcaps, p=0.0)
    np.testing.assert_allclose(w, equal_weight(4), atol=1e-12)


def test_diversity_weighted_rejects_zero_market_caps():
    with pytest.raises(ValueError):
        diversity_weighted(np.zeros(4))


def test_fixed_allocation_round_trip():
    w = fixed_allocation(
        {"VTI": 0.25, "AGG": 0.25, "DBC": 0.25, "VIXY": 0.25},
        ["VTI", "AGG", "DBC", "VIXY"],
    )
    np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25])
    _assert_simplex(w)


def test_fixed_allocation_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        fixed_allocation(
            {"VTI": 0.5, "AGG": 0.5},
            ["VTI", "AGG", "DBC", "VIXY"],
        )


def test_fixed_allocation_rejects_non_unit_sum():
    with pytest.raises(ValueError):
        fixed_allocation(
            {"VTI": 0.5, "AGG": 0.6, "DBC": 0.0, "VIXY": 0.0},
            ["VTI", "AGG", "DBC", "VIXY"],
        )
