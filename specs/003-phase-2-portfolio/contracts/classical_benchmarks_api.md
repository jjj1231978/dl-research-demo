# Contract: `src/strategies/portfolios.py`

**Importer**: `pages/2_💼_Portfolio.py`, `scripts/run_backtests.py`,
`tests/unit/test_classical_portfolios.py`.

Self-contained per Constitution Principle VI (no imports from
`~/projects/QIS_Commodities` or any external repo).

## Public surface

```python
from src.strategies.portfolios import (
    equal_weight,
    min_variance,
    max_diversification,
    diversity_weighted,
    fixed_allocation,
)
```

## `equal_weight(n_assets)` → np.ndarray of shape (n_assets,)

Returns `[1/n, 1/n, ..., 1/n]`. Trivial; included for API symmetry.

## `min_variance(returns_window, shrinkage="ledoit_wolf")` → np.ndarray

Markowitz 1952 minimum-variance long-only weights computed via
constrained optimisation:

```
min  w' Σ w
s.t. sum(w) == 1, w >= 0
```

where `Σ` is the shrunk covariance (`LedoitWolf` from sklearn; falls back
to `ShrunkCovariance(shrinkage=0.1)` if LedoitWolf fails to converge).

`returns_window`: (lookback, n_assets) array of daily returns.

Solved with `scipy.optimize.minimize(method='SLSQP')`. Initial guess =
equal weight.

## `max_diversification(returns_window)` → np.ndarray

Choueifaty & Coignard 2008 weights:

```
w* = argmax  (w' σ) / sqrt(w' Σ w)
s.t. sum(w) == 1, w >= 0
```

where `σ` is the vector of per-asset stdevs.

Same SLSQP solver as `min_variance`.

## `diversity_weighted(market_caps, p=0.5)` → np.ndarray

Samo & Vervuurt 2016 weights:

```
w_i = μ_i^p / Σ μ_j^p
```

where `μ` is the normalised market-cap vector. `p ∈ (0, 1)`; default 0.5
per the paper.

`market_caps`: (n_assets,) vector of current market caps. From the
parquet's `close × shares_outstanding`.

**Skip-with-warning**: When `shares_outstanding` is not in the parquet
(FR-001 marks it optional), `diversity_weighted` raises
`ValueError("market_caps unavailable — skip DWP")`. Callers catch the
ValueError and surface a warning in the page.

## `fixed_allocation(weights_dict, asset_ordering)` → np.ndarray

For the four fixed allocations in paper Table 1, applicable to ETF basket
only.

```python
fixed_allocation(
    {"VTI": 0.25, "AGG": 0.25, "DBC": 0.25, "VIXY": 0.25},
    asset_ordering=["VTI", "AGG", "DBC", "VIXY"],
)  # → array([0.25, 0.25, 0.25, 0.25])
```

Raises `ValueError` if `weights_dict` doesn't cover all assets in
`asset_ordering` or if the values don't sum to 1.

The four paper allocations live in `pages/2_💼_Portfolio.py`'s ETF
configuration table:

```python
ETF_FIXED_ALLOCATIONS = {
    "alloc_25_25_25_25": {"VTI": 0.25, "AGG": 0.25, "DBC": 0.25, "VIXY": 0.25},
    "alloc_50_10_20_20": {"VTI": 0.50, "AGG": 0.10, "DBC": 0.20, "VIXY": 0.20},
    "alloc_10_50_20_20": {"VTI": 0.10, "AGG": 0.50, "DBC": 0.20, "VIXY": 0.20},
    "alloc_40_40_10_10": {"VTI": 0.40, "AGG": 0.40, "DBC": 0.10, "VIXY": 0.10},
}
```

## Invariants (asserted by `test_classical_portfolios.py`)

For all five functions and a fixed-seed synthetic fixture
`(252 days × N=8 assets)` where reasonable input is provided:

1. Output shape = `(n_assets,)`.
2. `abs(weights.sum() - 1.0) < 1e-6`.
3. `(weights >= -1e-9).all()` (allow tiny negative numerical slack).
4. `(weights <= 1.0 + 1e-9).all()`.
5. `np.isfinite(weights).all()`.

Property tests:

- `test_min_variance_lower_vol_than_equal_weight`: on a fixture with
  heterogeneous asset vols, realised portfolio vol of `min_variance` is
  strictly less than `equal_weight` over the fixture window.
- `test_max_diversification_equals_equal_weight_when_symmetric`: with
  identical correlations and stdevs, `max_diversification` reduces to
  `equal_weight` (within 1e-6).
- `test_diversity_weighted_formula_equality`: explicit formula equality
  on a hand-computed fixture.
- `test_fixed_allocation_rejects_mismatched`: passing a dict missing one
  ticker raises `ValueError`.
