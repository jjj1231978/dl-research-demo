# Research: Phase 2 — Portfolio Page (Zhang et al. 2020)

**Branch**: `003-phase-2-portfolio` | **Date**: 2026-05-17
**Inputs**: `spec.md`, `Project_brief.md` §6 Page 2, Phase 1 retrospective.

This records the stack/algorithmic decisions for Phase 2. Locked clarifications
are in `spec.md` Clarifications session 2026-05-17 and are not re-litigated.

---

## R1 — Classical-method library: in-repo, scikit-learn for covariance only

**Decision**: Implement all four classical portfolio methods in
`src/strategies/portfolios.py` directly. Use `sklearn.covariance.LedoitWolf`
for the regularised covariance estimate that `min_variance` and
`max_diversification` need; everything else is plain numpy.

**Rationale**:
- Phase 1 precedent (self-contained signal code). No external repo imports
  at test time (Constitution VI).
- `sklearn.covariance.LedoitWolf` is already in `requirements.txt` from
  Phase 0 (the brief anticipated it for Phase 2). Production-grade
  shrinkage estimator — no need to reimplement.
- All four methods have closed-form formulas that fit in ≤ 30 lines each.

**The four methods**:

| Method | Formula | Inputs |
|---|---|---|
| `equal_weight(n)` | `w_i = 1/n` | `n` (asset count) |
| `min_variance(returns_window)` | `w* = argmin w' Σ w  s.t. Σw=1, w≥0` — solved via SLSQP on the LedoitWolf-shrunk covariance | rolling 50-day return window (paper §4.2) |
| `max_diversification(returns_window)` | `w* = argmax (w'σ) / sqrt(w'Σw)`, σ = vector of per-asset stdevs (Choueifaty & Coignard 2008) — solved via SLSQP | same |
| `diversity_weighted(market_caps, p=0.5)` | `w_i = μ_i^p / Σ μ_j^p` where μ is normalised market cap (Samo & Vervuurt 2016) | per-day market caps |

**Alternatives considered**:
- `cvxpy` for QP solvers — heavier dep, slower import, overkill for 4-asset
  and 20-asset problems where SLSQP converges in ms.
- `pypfopt` library — gives you 90% of this code in 10 lines, but it's a
  >5MB dep that pulls in matplotlib, and the brief's "Pre-existing Canon"
  principle (V) prefers small focused dependencies.

---

## R2 — Deep model output: Softmax for long-only weight constraint

**Decision**: `DeepPortfolioMLP` final layer is `nn.Linear(hidden, n_assets)`
followed by `nn.Softmax(dim=-1)`. Output is a length-N vector with
non-negative entries summing to 1.

**Rationale**:
- Paper §4.3: "We choose softmax as the activation function for the output
  layer". Direct paper match.
- Long-only constraint satisfied by construction; no projection step
  needed at inference.
- Differentiable end-to-end through the Sharpe loss.

**Architecture** (per `02_deep_portfolio_optimization.ipynb` + paper §4.3):

```
input: (B, lookback=50, 2*N_assets)   — close & return per asset, flat
  → Flatten            → (B, 100*N)
  → Linear(100*N, 64)  → ReLU
  → Linear(64, 64)     → ReLU
  → Linear(64, N)
  → Softmax(dim=-1)
  → output: (B, N) — weights summing to 1
```

For ETF basket (N=4): input dim = 100×4 = 400, output dim = 4.
For sp500_20 (N=20): input dim = 100×20 = 2,000, output dim = 20.

**Alternatives considered**:
- Tanh + projection-to-simplex (Wang et al. 2024) — paper doesn't use it;
  introduces a non-differentiability at the simplex boundary.
- Sigmoid + L1 normalisation — slow gradients near saturation.

---

## R3 — Training: per-day portfolio Sharpe-loss aggregation (Phase 1 retro)

**Decision**: Same per-day-batch shape as Phase 1's
`train_deep_momentum.py`:
- Each training sample is one day with all-N asset features stacked
- Per-day portfolio return = `(weights * next_day_returns).sum()` (NOT
  mean — softmax weights already normalize)
- Sharpe loss runs on the per-day portfolio return time series

**Rationale**: Phase 1 commit 93578ca documented the per-sample
cross-section pitfall. Same root cause would inflate Phase 2 metrics if
not handled correctly.

**Implementation differences vs Phase 1**:
- No per-asset mask needed (Portfolio always uses all N assets, while
  Momentum had per-asset entry-date staggering for BCOM commodities)
- Weights × returns uses `.sum()` not `.mean()` because softmax weights
  already represent capital allocation fractions

```python
def _portfolio_returns(xb, yb):
    """xb: (B, T=50, 2N); yb: (B, N) next-day returns. Returns (B,)."""
    weights = model(xb)                   # (B, N) — softmax
    return (weights * yb).sum(dim=1)     # (B,)
```

---

## R4 — Hyperparameters (per-universe defaults)

**Decision**: Same defaults across both universes (modulo input dimension):

| Hyperparameter | Value | Source |
|---|---|---|
| `lookback` | 50 | Paper §4.3 |
| `hidden_size` | 64 | Notebook reference + paper Table 3 |
| `optimizer` | Adam | Paper §4.3 |
| `lr` | 1e-3 | Notebook reference |
| `batch_size_days` | 32 | Phase 1 precedent (commit 93578ca) |
| `max_epochs` | 200 | Cap; early-stopping triggers earlier |
| `patience` | 25 | Canonical EarlyStopping default |
| `min_delta` | 1e-4 | Canonical |
| `seed` | 42 | `torch.manual_seed(42)` + `np.random.seed(42)` at function entry |

---

## R5 — Transaction-cost model

**Decision**: At backtest time, charge `cost_rate × Σ |Δw_i|` per
rebalancing day (turnover cost). Two cost rates per paper Table 1:
`C ∈ {0.0001, 0.0010}` (0.01% and 0.10%).

**Implementation in `scripts/run_backtests.py --portfolio`**:

```python
turnover = (weights.shift(1) - weights).abs().sum(axis=1)
cost = cost_rate * turnover.fillna(0)
pnl_after_cost = (weights.shift(1) * returns).sum(axis=1) - cost
```

This is identical to the paper's §4.5 formula. The backtest panel stores
`portfolio_return` already-net-of-cost, with `cost_rate` as a discriminating
column.

---

## R6 — Modal training script structure

**Decision**: Mirror Phase 1's `src/training/train_deep_momentum.py`
exactly, parameterised by `--universe ∈ {etfs, 20stock}`. Single file
with Modal scaffolding at top and device-agnostic `train()` body at
bottom.

**Differences from Phase 1**:
- `--universe` flag instead of `--arch` (architecture is fixed; universe
  varies)
- Data loader reads `etf_basket.parquet` or `sp500_20.parquet`
- Checkpoint named `deep_portfolio_{universe}.pt`

Same per-day batching, same Modal Volume, same image cache.

---

## R7 — Test fixtures (self-contained — no external repo dependency)

**Decision**: Same pattern as Phase 1. Fixture is a fixed-seed synthetic
return panel `(252 days × N assets)`; property + formula-equality
assertions on the in-repo implementations.

- `test_classical_portfolios.py`:
  - `test_equal_weight_invariants` — sums to 1, all entries equal.
  - `test_min_variance_lower_vol_than_equal_weight` — on a fixture with
    diverse asset volatilities, min-var should produce a portfolio with
    realised vol strictly lower than equal-weight (a strong claim that
    only holds for the optimizer's own solution).
  - `test_max_diversification_equals_equal_weight_when_equal_corrs` —
    on a fixture with identical correlations and stdevs, max
    diversification reduces to equal weight (closed-form sanity).
  - `test_diversity_weighted_matches_formula` — `w_i = μ_i^0.5 / Σ μ_j^0.5`
    asserted with `np.allclose(atol=1e-12)`.
- `test_softmax_head.py`: random `(1, 50, 2N)` → output sums to 1 ± 1e-6
  and all weights in `[0, 1]`.
- `test_portfolio_checkpoint_smoke.py`: load each `.pt`, forward, assert
  shape + finiteness + simplex. Skip if checkpoint absent (US1 ships
  before US2 lands).
- `test_portfolio_train_smoke.py`: 1-epoch CPU run on a synthetic
  fixture, assert dict return + .pt written.

---

## R8 — Tab 4 column mapping (paper Table 1 → `src/metrics.py`)

| Paper Table 1 column | `report_metrics` key |
|---|---|
| E[R] | `annual_ret` |
| Std(R) | `annual_std` |
| Sharpe | `annual_sharpe` |
| DD(R) (downside deviation) | `downside_dev` |
| Sortino | `sortino` |
| MDD | `max_drawdown` |
| % of +Ret | `pct_positive` |
| Ave. P / Ave. L | `avg_p_over_l` |

8 columns total. Same bijection as Phase 1's R8, minus `Calmar` (paper
Table 1 doesn't include it).
