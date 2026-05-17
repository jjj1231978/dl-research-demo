"""Classical portfolio-allocation methods (Phase 2 — Zhang et al. 2020).

Self-contained per FR-005 + Constitution VI: no imports from external
repos. Math comes from the cited papers + the reference notebook
`notebooks/reference/01_classical_portfolio_optimization.ipynb`.

Five methods:

- ``equal_weight(n)`` — 1/N benchmark.
- ``min_variance(returns_window, …)`` — Markowitz 1952 minimum-variance
  long-only weights via LedoitWolf-shrunk covariance + SLSQP.
- ``max_diversification(returns_window)`` — Choueifaty & Coignard 2008.
- ``diversity_weighted(market_caps, p=0.5)`` — Samo & Vervuurt 2016.
- ``fixed_allocation(weights_dict, asset_ordering)`` — for paper Table 1's
  four pre-specified ETF allocations.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def equal_weight(n_assets: int) -> np.ndarray:
    """Constant 1/N allocation. Paper Table 1 baseline."""
    if n_assets <= 0:
        raise ValueError(f"n_assets must be positive; got {n_assets}")
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)


def _shrunk_covariance(returns_window: np.ndarray, method: str) -> np.ndarray:
    """Estimate covariance with sklearn shrinkage. Falls back from
    LedoitWolf → ShrunkCovariance if the LW solver fails to converge
    (rare but possible on near-degenerate input).
    """
    from sklearn.covariance import LedoitWolf, ShrunkCovariance

    if method == "ledoit_wolf":
        try:
            return LedoitWolf().fit(returns_window).covariance_
        except (ValueError, np.linalg.LinAlgError):
            method = "shrunk"
    return ShrunkCovariance(shrinkage=0.1).fit(returns_window).covariance_


def _project_to_simplex(w: np.ndarray) -> np.ndarray:
    """Clamp to [0, ∞) then renormalize so the SLSQP near-solution sits
    cleanly on the long-only simplex despite tiny numerical slack.
    """
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return equal_weight(len(w))
    return w / s


def min_variance(
    returns_window: np.ndarray,
    shrinkage: str = "ledoit_wolf",
) -> np.ndarray:
    """Markowitz 1952 minimum-variance long-only weights.

    Solves ``min w' Σ w s.t. sum(w) == 1, w >= 0`` via SLSQP with the
    shrunk covariance.

    Args:
        returns_window: shape (lookback, n_assets), daily returns.
        shrinkage: ``ledoit_wolf`` (default) or ``shrunk``.

    Returns:
        ``(n_assets,)`` weights vector on the long-only simplex.
    """
    from scipy.optimize import minimize

    n = returns_window.shape[1]
    cov = _shrunk_covariance(returns_window, shrinkage)

    def variance(w):
        return float(w @ cov @ w)

    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(
        variance, x0=equal_weight(n), method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    return _project_to_simplex(res.x)


def max_diversification(returns_window: np.ndarray) -> np.ndarray:
    """Choueifaty & Coignard 2008 maximum-diversification weights.

    Solves ``max (w' σ) / sqrt(w' Σ w) s.t. sum(w) == 1, w >= 0`` where
    ``σ`` is the vector of per-asset stdevs.
    """
    from scipy.optimize import minimize

    n = returns_window.shape[1]
    cov = _shrunk_covariance(returns_window, "ledoit_wolf")
    sigma = np.sqrt(np.diag(cov))

    def neg_diversification_ratio(w):
        port_vol = np.sqrt(max(w @ cov @ w, 1e-18))
        weighted_sigma = w @ sigma
        return -weighted_sigma / port_vol

    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(
        neg_diversification_ratio, x0=equal_weight(n), method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    return _project_to_simplex(res.x)


def diversity_weighted(market_caps: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Samo & Vervuurt 2016 diversity-weighted portfolio.

    ``w_i = μ_i^p / Σ μ_j^p`` where ``μ`` is the normalised market-cap
    vector. With ``p ∈ (0, 1)`` (default 0.5 per the paper).

    Args:
        market_caps: ``(n_assets,)`` per-asset market caps (close ×
            shares_outstanding).
        p: power exponent. ``p=1`` reduces to cap-weight; ``p=0`` reduces
            to equal-weight.
    """
    if np.any(market_caps < 0):
        raise ValueError("market_caps must be non-negative")
    total = market_caps.sum()
    if total <= 0:
        raise ValueError("market_caps unavailable — skip DWP")
    mu = market_caps / total
    powered = np.power(mu, p)
    return powered / powered.sum()


def fixed_allocation(
    weights_dict: Mapping[str, float],
    asset_ordering: Sequence[str],
) -> np.ndarray:
    """Return a weight vector positioned per ``asset_ordering``.

    For paper Table 1's four pre-specified ETF allocations.
    """
    missing = set(asset_ordering) - set(weights_dict)
    if missing:
        raise ValueError(f"weights_dict missing entries for {sorted(missing)}")
    weights = np.array([weights_dict[a] for a in asset_ordering], dtype=np.float64)
    s = float(weights.sum())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0; got {s:.6f}")
    return weights
