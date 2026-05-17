# Research: Phase 1 — Momentum Page (Lim et al. 2019)

**Branch**: `002-phase-1-momentum`
**Date**: 2026-05-17
**Inputs**: `spec.md`, `plan.md`, `Project_brief.md` §§ 2.2, 6, 7, 11, 13.3,
13.5; `~/projects/QIS_Commodities/src/data/`, `src/signals/`,
`src/backtest/`

This document records the stack/library/algorithmic decisions reached during
Phase 1 planning. The four `/speckit-clarify` resolutions (OD-1 universe
scope, OD-2 copy-with-attribution + OD-2 follow-up F0 position, OD-3
60/40 split) are recorded in `spec.md` Clarifications session 2026-05-17
and are not re-litigated here.

---

## R1 — Continuous-futures construction: F0 ratio-adjustment

**Decision**: Use `~/projects/QIS_Commodities/src/data/term_structure.py:
build_ratio_adjusted_series(term_structure, position=0)` verbatim, copied
into `src/data/futures/term_structure.py` with attribution header. F0
front-month with the default `roll_offset_bdays = -5`.

**Rationale**: Already resolved in /clarify. Recapping:

- Paper-faithful (Lim et al. 2019 uses Pinnacle CLC, which is F0-based
  ratio-adjusted).
- Convention match with developer's existing QIS_Commodities work.
- Industry standard for trend-following.
- Expiry-week noise mitigated by 5-business-day pre-delivery roll.

The QIS algorithm in detail (line numbers as of 2026-05-17 in the source):

```python
# contracts.py:58 — build_roll_calendar(root, months, start_year, end_year, roll_offset_bdays=-5)
#   For each contract, roll_date = first-of-delivery-month + BDay(-5)
# contracts.py:96 — identify_active_contracts(roll_calendar, dates, n_contracts=13)
#   For each date, F0 = nearest contract with roll_date > date; F1, F2, …
# term_structure.py:129 — build_ratio_adjusted_series(term_structure, position=0)
#   prices = term_structure["F0"].dropna()
#   returns = prices.pct_change()              # roll-jump survives only on roll-day return
#   adjusted = (1 + returns).cumprod()         # rebuild from clean returns
#   adjusted = adjusted * (prices.iloc[-1] / adjusted.iloc[-1])  # anchor to latest
```

**Alternatives considered**: F1 (deferred for a future carry signal),
back-adjustment additive (can go negative on long histories), splice
(spurious roll jumps).

---

## R2 — Deep model output activation: Softsign

**Decision**: Both `DeepMomentumMLP` and `DeepMomentumLSTM` end with
`nn.Softsign()` for output. Maps the final scalar to `(-1, +1)` as the
continuous position size.

**Rationale**:

- Matches `notebooks/reference/03_deep_momentum_strategy.ipynb` (the
  developer's reference notebook for this paper) and brief §13.3's
  `DeepMomentumMLP` starter code.
- Paper Table 3 says "tanh / Softsign" output; Softsign is smoother and
  has lighter saturation than tanh, which the paper notes helps gradients
  through the Sharpe loss.
- The `(-1, +1)` range gives the model the full "long-or-short
  continuous-sized" position space the paper requires.

**Alternatives considered**: `tanh` (slightly more saturated; brief §13.3
chose Softsign explicitly), `clipped linear` (loses gradient outside
range, not paper-faithful).

---

## R3 — Train/test split: 60/40 chronological

**Decision**: Train 2010-06-06 → 2019-12-31; test 2020-01-01 → 2026-05-05.
Encoded as module-level constants in `src/data/futures/__init__.py`:

```python
TRAIN_START = date(2010, 6, 6)
TRAIN_END = date(2019, 12, 31)
TEST_START = date(2020, 1, 1)
TEST_END = None   # forward — uses whatever data exists at backtest time
```

**Rationale**: Resolved in /clarify (OD-3). Recapping: matches the brief's
"extension to recent regimes" framing in §4.1; test window covers COVID +
2022 rate-hike + recent commodity moves.

**Alternatives considered**: 70/30 (less test variety), walk-forward folds
(3× Modal compute, deferred to Phase 4 stretch).

---

## R4 — Hyperparameter defaults (MLP + LSTM training)

**Decision**: Adopt the developer's reference-notebook defaults for both
architectures. Paper Table 3 is the source of truth for hidden sizes;
optimizer + LR + batch are from the source notebooks.

| Hyperparameter | MLP | LSTM | Source |
|---|---|---|---|
| `hidden_size` | 20 | 10 | Paper Table 3 |
| `optimizer` | Adam | Adam | `notebooks/reference/03_deep_momentum_strategy.ipynb` |
| `lr` | 1e-3 | 1e-3 | Same notebook; paper §4.3 lists 1e-3 / 1e-4 range |
| `batch_size` | 128 | 128 | Same notebook |
| `epochs (cap)` | 200 | 200 | Cap; early-stopping triggers earlier |
| `early_stopper.patience` | 25 | 25 | `src/early_stopper.py` canonical signature |
| `early_stopper.min_delta` | 1e-4 | 1e-4 | Default in `EarlyStopping` |
| `seq_length` | 60 | 60 | EWMA-span equivalent; paper §4.3 |
| `n_features` | 8 | 8 | 5 normalized return horizons + 3 MACD timescales |

**Rationale**:

- Stay close to the developer's reference notebook so the Tab 4 numbers
  the page displays come from a well-understood configuration.
- Paper Table 3 hidden sizes are the only published architectural anchor;
  matching them keeps Principle I (paper-faithful replication) intact.

**Alternatives considered**: Hyperparameter search (out of scope per brief
§16 Non-Goals: "Hyperparameter-tune the pretrained models beyond what's
in the notebooks").

---

## R5 — Optimizer choice + training loss

**Decision**: `torch.optim.Adam`, loss is the canonical
`src/losses.py:SharpeLoss` used as-is (Constitution Principle V; FR-009).

**Rationale**:

- The developer's `Neg_Sharpe` and `SharpeLoss` are in `src/losses.py`
  byte-identical with the source notebooks; constitution forbids
  extension.
- For multi-asset uses (the BCOM 18-contract universe), the caller
  pre-aggregates `outputs * future_rets` into a per-day portfolio-return
  series before passing to `SharpeLoss`. The training script's `train()`
  body owns this aggregation per `src/losses.py` line 12–17 ("For the
  multi-asset Portfolio page where the model outputs a softmax weight
  vector per timestep, Claude Code should wrap or pre-aggregate the
  per-asset products into a portfolio return series before passing to
  `SharpeLoss`, rather than modifying the loss class."). The same
  guidance applies to multi-asset momentum (Phase 1) — pre-aggregate per
  day across the 18 contracts before the loss call.

**Alternatives considered**: SGD (slower convergence for this loss
geometry), AdamW (introduces weight-decay regularisation not in the
notebooks).

---

## R6 — Modal training script structure

**Decision**: Single file `src/training/train_deep_momentum.py` following
the canonical Modal template from `Project_brief.md` §7.2 (post-2026-05-17
amendment). Top of file declares `App` / `Image` / `Volume` /
`@app.function(gpu="T4")` + `@app.local_entrypoint()`. Bottom of file
defines plain Python `def train(...)` body that is device-agnostic and
importable for unit smoke tests.

Two invocations of the same file:

```bash
python -m src.training.train_deep_momentum --arch MLP --max-epochs 1   # CPU smoke
modal run src/training/train_deep_momentum.py --arch MLP               # GPU real
```

**Rationale**:

- Constitution v1.1.0 §"Training workflow (Modal)" mandates this exact
  pattern.
- Single file keeps `--arch` flag a one-line parameterisation; no
  notebook wrapper needed.
- CPU-only smoke run satisfies Constitution Principle II + VI (unit
  test coverage).

**Alternatives considered**: Separate `train_mlp.py` and `train_lstm.py`
(more file churn for no benefit), notebook orchestrator (forbidden by
v1.1.0).

---

## R7 — Unit-test fixtures (self-contained — no external repo dependency)

**Decision**: Tests run identically on any machine that has the Showcase
repo and a venv. No `sys.path` import of `~/projects/QIS_Commodities` or
any other external code. The signal-math reference is the brief's §13.5
+ the Lim et al. 2019 / Baz et al. 2015 papers; tests assert properties
of the in-repo implementation directly.

- `tests/unit/test_tsmom.py` — property-based + formula-equality
  assertions on `src/strategies/tsmom.py`, per FR-019:
    - Fixed-seed synthetic price series (1 contract × 1000 days):
      `100 * (1 + rng.normal(0, 0.01, 1000)).cumprod()` with
      `np.random.default_rng(0)`.
    - `long_only`: output is all `1.0`.
    - `sgn_returns`: on a monotonically increasing series, output `+1`
      past the lookback; monotonically decreasing → `-1`; constant → `0`.
    - `macd_signal`: constant input → output `0` (numerator collapses).
    - `macd_ensemble`: equals the elementwise mean of the three
      component MACDs (`np.allclose(..., atol=1e-12)`).
    - First `lookback` rows are NaN (no look-ahead leakage).
- `tests/unit/test_vol_targeting.py` uses the same fixture pattern;
  asserts realised vol of the scaled signal is within 5 % of σ_target.
- `tests/unit/test_checkpoint_smoke.py` instantiates each model with
  paper-Table-3 hidden sizes, loads the committed `.pt` from
  `data/pretrained/`, runs a single forward pass on a `(1, 60, 8)`
  zeros tensor, asserts `torch.isfinite(out).all()` and Softsign bounds.
- `tests/unit/test_train_smoke.py` calls the device-agnostic
  `train(...)` body directly with a truncated 5-contract × 100-day
  subset on CPU; asserts a `.pt` was written and is loadable.

**Rationale**:

- Self-containment per the developer's explicit principle: "the folder
  to be self-contained as much as possible" — tests must run on any
  clone without setting up adjacent repos.
- The brief's §13.5 IS the authoritative signal-math reference for this
  project. QIS_Commodities' `signals/trend.py` is informational only
  (different signal family — multi-lookback `sign(momentum)`, no MACD).
- Property + formula-equality tests are STRONGER than parity-against-
  external: they catch any change to the math directly (formula
  equality) AND prove invariants the math must satisfy (`long_only` is
  constant 1, `macd` of constant input is 0, etc.). The previous
  parity-against-QIS design would have missed a formula bug that
  coincidentally also broke the QIS reference.
- Realised-vol tolerance of 5 % is wider than 1e-12 because σ_target
  enforcement is an *output* property — EWMA smoothing produces a
  stochastic realised σ.

**Alternatives considered**:

- Import QIS at test time (`sys.path` insertion). Rejected: makes the
  repo non-self-contained; a clone without QIS at the assumed path
  fails the test suite.
- Hard-code reference values from a one-time QIS run. Rejected: test
  pretends to be a regression but doesn't catch formula drift in QIS;
  also still implicitly couples to a specific QIS version.

---

## R8 — Tab 4 metrics alignment with Phase 0 `src/metrics.py`

**Decision**: The Tab 4A / 4B metrics tables consume the output of
`src/metrics.py:report_metrics(ret)` directly. The 9 keys it returns map
to the paper's Exhibit 2/3 columns as follows:

| Paper column | `report_metrics` key | Notes |
|---|---|---|
| E[Return] | `annual_ret` | Annualized mean × 252 |
| Vol | `annual_std` | Annualized std × √252 |
| Downside Deviation | `downside_dev` | Phase 0 extension |
| MDD | `max_drawdown` | Phase 0 extension |
| Sharpe | `annual_sharpe` | Annualized ratio |
| Sortino | `sortino` | Phase 0 extension |
| Calmar | `calmar` | Phase 0 extension |
| % +ve Returns | `pct_positive` | Phase 0 extension |
| Ave. P / Ave. L | `avg_p_over_l` | Phase 0 extension |

**Rationale**: Phase 0 explicitly extended `metrics.py` with these six
new keys per brief §13.7 — Phase 1 is the first phase to consume them.
The bijection above keeps Constitution Principle V intact (no signature
change to `report_metrics`).

**Alternatives considered**: Adding a "paper-column-name" mapping table
in Tab 4 code (more flexible but couples the table renderer to the
column names). The pure-dict approach is simpler.
