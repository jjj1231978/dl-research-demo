"""Unit tests for src/metrics.py — Phase 0 US3 / FR-020 / SC-004.

Two test families:
1. **Principle V regression**: the three original keys (annual_ret,
   annual_std, annual_sharpe) MUST return values byte-identical to the
   pre-extension formula. The test recomputes the formula inline and asserts
   pytest.approx equality.
2. **Extension coverage**: the six new keys (downside_dev, sortino,
   max_drawdown, calmar, pct_positive, avg_p_over_l) MUST be present, finite,
   and equal to their documented formulas.

The brief's §13.7 ALSO suggested tolerance bands (Sharpe ≈ 0.79 etc.) — those
turned out to be the THEORETICAL Sharpe for the population (mean=0.0005,
std=0.01), not the empirical Sharpe for a single N=1000 draw. Empirical
Sharpe at that sample size has SE ~0.5 annualized, so the brief's bands are
mathematically optimistic. Formula-equality assertions sidestep the issue
and give a stronger test (catches any change to the formula, not just ones
that drift outside a window). See spec.md Clarifications session 2026-05-17.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.metrics import report_metrics


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_returns_series() -> np.ndarray:
    """Fixed-seed N(0.0005, 0.01, 1000) per brief §13.7 + contracts.

    Uses np.random.default_rng(0) (modern API). Reproducible across numpy
    versions ≥ 1.17.
    """
    return np.random.default_rng(0).normal(0.0005, 0.01, 1000)


# ---------------------------------------------------------------------------
# Principle V regression — original keys byte-identical
# ---------------------------------------------------------------------------


def test_original_keys_regression(synthetic_returns_series):
    """The three original report_metrics keys MUST equal their inline-formula
    values to numpy.float64 tolerance. Any change to the formula breaks this.
    """
    ret = synthetic_returns_series
    m = report_metrics(ret)

    # Recompute by hand — these expressions ARE the original-key invariants
    expected_annual_ret = np.mean(ret) * 252
    expected_annual_std = np.std(ret) * np.sqrt(252)
    expected_annual_sharpe = (np.mean(ret) / np.std(ret)) * np.sqrt(252)

    assert m["annual_ret"] == pytest.approx(expected_annual_ret, rel=1e-12), (
        f"annual_ret regression: {m['annual_ret']} != {expected_annual_ret} "
        f"(Principle V violation — original key changed)"
    )
    assert m["annual_std"] == pytest.approx(expected_annual_std, rel=1e-12), (
        f"annual_std regression: {m['annual_std']} != {expected_annual_std}"
    )
    assert m["annual_sharpe"] == pytest.approx(expected_annual_sharpe, rel=1e-12), (
        f"annual_sharpe regression: {m['annual_sharpe']} != {expected_annual_sharpe}"
    )


# ---------------------------------------------------------------------------
# Extension key contract — set, finiteness, formulas
# ---------------------------------------------------------------------------


def test_extended_keys_present(synthetic_returns_series):
    """Key-set invariant from contracts/metrics_report_schema.md §"Invariants" #2."""
    m = report_metrics(synthetic_returns_series)
    expected = {
        "annual_ret", "annual_std", "annual_sharpe",
        "downside_dev", "sortino", "max_drawdown",
        "calmar", "pct_positive", "avg_p_over_l",
    }
    assert set(m.keys()) >= expected, f"Missing keys: {expected - set(m.keys())}"


def test_extended_keys_finite(synthetic_returns_series):
    """Finiteness invariant from §"Invariants" #3."""
    m = report_metrics(synthetic_returns_series)
    for k, v in m.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_downside_dev_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    expected = np.std(ret[ret < 0]) * np.sqrt(252)
    assert report_metrics(ret)["downside_dev"] == pytest.approx(expected, rel=1e-12)


def test_sortino_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    expected = (np.mean(ret) / np.std(ret[ret < 0])) * np.sqrt(252)
    assert report_metrics(ret)["sortino"] == pytest.approx(expected, rel=1e-12)


def test_max_drawdown_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    equity = (1 + ret).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    expected = abs(drawdown.min())
    assert report_metrics(ret)["max_drawdown"] == pytest.approx(expected, rel=1e-12)


def test_calmar_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    annual_ret = np.mean(ret) * 252
    equity = (1 + ret).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min())
    expected = annual_ret / max_dd
    assert report_metrics(ret)["calmar"] == pytest.approx(expected, rel=1e-12)


def test_pct_positive_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    expected = (ret > 0).mean()
    assert report_metrics(ret)["pct_positive"] == pytest.approx(expected, rel=1e-12)


def test_avg_p_over_l_formula(synthetic_returns_series):
    ret = synthetic_returns_series
    expected = ret[ret > 0].mean() / abs(ret[ret < 0].mean())
    assert report_metrics(ret)["avg_p_over_l"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Signature invariant (Principle V)
# ---------------------------------------------------------------------------


def test_signature_unchanged():
    """report_metrics(ret) — parameter list must not change (Principle V)."""
    sig = inspect.signature(report_metrics)
    params = list(sig.parameters.keys())
    assert params == ["ret"], (
        f"report_metrics signature changed: parameters are {params}, expected ['ret']"
    )
