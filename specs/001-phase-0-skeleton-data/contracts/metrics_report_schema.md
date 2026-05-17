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

## Synthetic-series reference values (used by `tests/unit/test_metrics.py`)

For `rng = np.random.default_rng(0); ret = rng.normal(0.0005, 0.01, 1000)`,
the expected values land in the tolerance bands documented in brief §13.7:

| Key | Expected range | Why |
|---|---|---|
| `annual_sharpe` | `0.74 ≤ x ≤ 0.84` | Brief §13.7 documents ≈ 0.79 |
| `max_drawdown` | `0.05 ≤ x ≤ 0.20` | Brief §13.7 documents that range |
| `pct_positive` | `0.47 ≤ x ≤ 0.57` | Brief §13.7 documents ≈ 0.52 |
| `annual_ret` | `0.10 ≤ x ≤ 0.16` | Derived: `0.0005 × 252 ≈ 0.126` |
| `annual_std` | `0.14 ≤ x ≤ 0.18` | Derived: `0.01 × √252 ≈ 0.159` |
| `downside_dev` | finite, > 0 | No specific band |
| `sortino` | finite | No specific band |
| `calmar` | finite, > 0 | No specific band |
| `avg_p_over_l` | `0.7 ≤ x ≤ 1.3` | Symmetric distribution → ratio ≈ 1 |

Tests assert membership in these bands AND that the value is finite. The
exact value is recorded as a comment in the test for reviewer eyeballing but
is NOT asserted (cross-platform float drift on `np.std` is real and the
bands give us room).
