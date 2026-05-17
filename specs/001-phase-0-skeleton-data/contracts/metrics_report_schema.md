# Contract: `src/metrics.py:report_metrics()` return schema

**Importer**: future per-paper pages (Tab 4 master tables) in Phases 1–3,
`tests/unit/test_metrics.py` in Phase 0.

## Signature

```python
def report_metrics(ret: np.ndarray) -> dict[str, float]
```

**Input**: 1-D numpy array of daily returns (decimal, not percentage). No
NaN/Inf in the input; the caller is responsible for cleaning. The function
does NOT validate input shape or content.

**Output**: dictionary with the nine keys listed below. All values are finite
Python `float`s.

## Output keys

| Key | Type | Original / NEW | Formula |
|---|---|---|---|
| `annual_ret` | `float` | Original | `np.mean(ret) * 252` |
| `annual_std` | `float` | Original | `np.std(ret) * np.sqrt(252)` |
| `annual_sharpe` | `float` | Original | `(np.mean(ret) / np.std(ret)) * np.sqrt(252)` |
| `downside_dev` | `float` | NEW | `np.std(ret[ret < 0]) * np.sqrt(252)` |
| `sortino` | `float` | NEW | `(np.mean(ret) / np.std(ret[ret < 0])) * np.sqrt(252)` |
| `max_drawdown` | `float` | NEW | `abs(drawdown.min())` where `drawdown = (equity - running_max) / running_max`, `equity = (1+ret).cumprod()`, `running_max = np.maximum.accumulate(equity)` |
| `calmar` | `float` | NEW | `annual_ret / max_drawdown` |
| `pct_positive` | `float` | NEW | `(ret > 0).mean()` |
| `avg_p_over_l` | `float` | NEW | `ret[ret > 0].mean() / abs(ret[ret < 0].mean())` |

## Invariants (Phase 0 acceptance gate)

1. **Original-keys regression invariant**: For any fixed-seed returns series,
   the three original keys (`annual_ret`, `annual_std`, `annual_sharpe`)
   MUST return values byte-identical to the values returned by the
   pre-extension version of `report_metrics`. (Principle V; FR-004.)
2. **Key-set invariant**: `set(report_metrics(ret).keys())` ⊇ `{annual_ret,
   annual_std, annual_sharpe, downside_dev, sortino, max_drawdown, calmar,
   pct_positive, avg_p_over_l}`. The function MAY add further keys in later
   phases (forward-compatible) but MUST NOT remove or rename any of these
   nine.
3. **Finite-value invariant**: every value is a finite `float` (no NaN, no
   ±Inf) for the canonical synthetic fixture
   `np.random.default_rng(0).normal(0.0005, 0.01, 1000)`. The function does
   not guarantee finiteness for pathological inputs (all-zero, all-positive,
   etc.); those are out-of-scope edge cases.
4. **Signature invariant**: parameter list and return type annotation MUST
   NOT change (Principle V).

## Test strategy (used by `tests/unit/test_metrics.py`)

The brief's §13.7 suggested **tolerance bands** ("Sharpe ≈ 0.79",
"max_drawdown ∈ [0.05, 0.20]", "hit_rate ≈ 0.52") that turn out to be the
**theoretical** values for the population (mean=0.0005, std=0.01) — NOT the
empirical values for a single N=1000 draw. Empirical Sharpe at N=1000 with
true Sharpe=0.79 has standard error ~0.5 (annualized), so a single draw can
fall anywhere in roughly [-0.5, 2.0]. The brief's bands are mathematically
optimistic about empirical convergence.

The Phase 0 tests sidestep the issue by using **formula-equality assertions**
instead of tolerance bands. For the fixture
`rng = np.random.default_rng(0); ret = rng.normal(0.0005, 0.01, 1000)`:

- **Principle V regression** (`test_original_keys_regression`): each of
  the three original keys is re-computed inline using the brief §13.7 /
  src.metrics formula and asserted equal to `report_metrics(ret)[k]` with
  `pytest.approx(rel=1e-12)`. This catches any change to the formula
  deterministically; no seed dependency.
- **Extension formulas** (one test per new key:
  `test_downside_dev_formula`, `test_sortino_formula`,
  `test_max_drawdown_formula`, `test_calmar_formula`,
  `test_pct_positive_formula`, `test_avg_p_over_l_formula`): same pattern —
  inline-recompute, assert `pytest.approx(rel=1e-12)`.
- **Key-set + finiteness** (`test_extended_keys_present`,
  `test_extended_keys_finite`): all nine keys present, all values finite.
- **Signature invariant** (`test_signature_unchanged`): `inspect.signature`
  confirms parameter list unchanged (Principle V).

A future test in Phase 1+ may add a tolerance-band check at a larger N
(e.g., N=100_000) where the empirical Sharpe DOES converge to within 5% of
theoretical — but that's a different sanity check, not a Principle V guard.
