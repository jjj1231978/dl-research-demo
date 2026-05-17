# Data Model: Phase 1 — Momentum Page

**Branch**: `002-phase-1-momentum`
**Date**: 2026-05-17
**Inputs**: `spec.md` §"Key Entities", `Project_brief.md` §§ 7.3, 8

Five entities. Per Constitution Principle IV all persistent state is
files on disk; entities below describe the **logical** shape of those
files and the in-memory dataclasses callers consume.

---

## E1 — `ContinuousFuturesContract`

A ratio-adjusted F0 continuous price series for one BCOM commodity root
(e.g., "CL" = WTI Crude, "ZC" = Corn). Constructed by
`scripts/fetch_futures.py` from raw individual-contract data in the
developer's databento lake.

**On-disk representation**: rows in
`${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet`. All 18 BCOM roots stored
in a single parquet long-format.

**Fields** (per FR-003):

| Field | Type | Required | Notes |
|---|---|---|---|
| `date` | `date32[day]` | ✅ | Trading day. UTC. |
| `contract` | `string` | ✅ | Continuous-series root (e.g., "CL", "ZC"). 18 distinct values for the 18 BCOM roots. |
| `asset_class` | `string` | ✅ | All rows have `"commodity"` in Phase 1 (per OD-1); column retained for forward compatibility. |
| `price` | `float64` | ✅ | Ratio-adjusted F0 price (paper-style). Anchored to most recent actual F0 close. |
| `return` | `float64` | ✅ | Daily simple return = `price.pct_change()`. NaN on the first row per contract. |

**Validation rules**:

- `len(df["contract"].unique()) == 18` (all BCOM roots present).
- For each contract, `df.set_index("date").index.is_monotonic_increasing`.
- The most recent `price` per contract MUST match the most recent F0
  close from the raw lake within 1e-9 (anchoring property of the QIS
  `build_ratio_adjusted_series`).

**State transitions**: immutable once written. Re-running
`scripts/fetch_futures.py` overwrites atomically (tempfile + rename
per Phase 0 pattern).

---

## E2 — `MomentumSignal`

A daily position-in-`(-1, +1)` series for one (strategy × contract) pair.
**Strategy** is one of: Long Only, Sgn(Returns), MACD, MLP-Sharpe,
LSTM-Sharpe. **Vol-scaling** is a boolean flag — when True, the raw
signal is rescaled to σ_target = 0.15 annualized via the EWMA-vol
helper in `src/strategies/vol_targeting.py`.

**On-disk representation**: rows in
`data/backtests/momentum_results.parquet` (see E4 below).

**In-memory representation**: pandas DataFrame indexed by date with a
single `position` column. Held transiently by the backtest engine and the
Tab 3 page renderer.

**Validation rules**:

- `(-1.0 - 1e-9) <= position <= (1.0 + 1e-9)` for every row (Softsign /
  Sgn output bound; small epsilon for floating-point slack).
- `position.isna().sum() <= seq_length` — the first `seq_length` rows
  may be NaN before the deep models have enough lookback.

**State transitions**: pure function output. No persistence between
runs.

---

## E3 — `DeepMomentumCheckpoint`

A trained `.pt` file plus its JSON sidecar. Two instances in Phase 1:
`data/pretrained/mlp_sharpe.{pt,json}` and
`data/pretrained/lstm_sharpe.{pt,json}`.

**`.pt` file**: `torch.save(state_dict, path)` output. Loadable on CPU
via `torch.load(path, map_location="cpu")`.

**Sidecar JSON schema** (per `Project_brief.md` §7.3 + constitution
v1.1.0 update):

```json
{
  "trained_on": "2026-05-18",
  "trained_with": "Modal T4 (image sha256:abc123…)",
  "torch_version": "2.x.x",
  "modal_app": "deep-finance-train-momentum",
  "arch": "MLP",
  "data_range": {"train_start": "2010-06-06", "train_end": "2019-12-31",
                  "test_start": "2020-01-01", "test_end": "2026-05-05"},
  "split": "chronological_60_40",
  "hyperparameters": {
    "hidden_size": 20, "lr": 1e-3, "batch_size": 128,
    "epochs_trained": 87, "patience": 25, "min_delta": 1e-4,
    "seq_length": 60, "n_features": 8
  },
  "final_metrics": {
    "val_neg_sharpe": -1.42,
    "test_annual_sharpe": 1.95,
    "test_max_drawdown": 0.12,
    "test_calmar": 2.10
  },
  "git_commit": "abc1234"
}
```

**Validation rules**:

- `arch ∈ {"MLP", "LSTM"}` and matches the `.pt` filename stem
  (`mlp_sharpe.pt` → `arch="MLP"`).
- `hyperparameters.hidden_size == 20` for MLP, `10` for LSTM (paper
  Table 3).
- `trained_with` MUST start with `"Modal "` (Constitution Principle III
  / v1.1.0 amendment).
- The `.pt` state-dict keys MUST match the `nn.Module`'s `state_dict()`
  keys when instantiated with the sidecar's hyperparameters — verified
  by `tests/unit/test_checkpoint_smoke.py`.

**State transitions**: produced by `src/training/train_deep_momentum.py`
on Modal, downloaded via `modal volume get`, committed to git, served by
the live app.

---

## E4 — `BacktestPanel`

The long-format DataFrame at `data/backtests/momentum_results.parquet`
that drives Tab 4. Pre-computed offline by `scripts/run_backtests.py`;
read by the page via Streamlit cache.

**Fields**:

| Field | Type | Required | Notes |
|---|---|---|---|
| `date` | `date32[day]` | ✅ | Trading day. |
| `contract` | `string` | ✅ | Continuous-series root. Same 18-value vocabulary as E1. |
| `strategy` | `string` | ✅ | One of `"long_only"`, `"sgn_returns"`, `"macd"`, `"mlp_sharpe"`, `"lstm_sharpe"`. |
| `vol_scaling` | `bool` | ✅ | `True` = signal rescaled to σ_target = 0.15; `False` = raw. |
| `daily_return` | `float64` | ✅ | Strategy P&L for that contract on that day. Used by Exhibit 5 box plots (per-asset metrics) and by the universe-aggregation step for Exhibit 4. |

**Row cardinality**: 18 contracts × 5 strategies × 2 vol-scaling × ~4000
trading days ≈ 720 k rows (well under 50 MB parquet).

**Validation rules**:

- `set(strategy.unique()) == {"long_only", "sgn_returns", "macd",
  "mlp_sharpe", "lstm_sharpe"}` (all 5 strategies present).
- `set(vol_scaling.unique()) == {True, False}` (both conditions present).
- For each `(strategy, vol_scaling, contract)` triple, the date set is
  monotonic increasing and non-empty.

**State transitions**: regenerated whenever models are retrained or the
fetch is refreshed. The script is idempotent — the output parquet is
fully deterministic given the same inputs.

---

## E5 — `TrainingRun`

A Modal function invocation that produces one `DeepMomentumCheckpoint`.
Not persisted as a project artifact — Modal's dashboard is the source of
truth — but the sidecar JSON of E3 includes enough provenance fields
(`modal_app`, `trained_with` image hash, `git_commit`) to identify the
specific run that produced any committed checkpoint.

**Fields (informational; no on-disk file owned by us)**:

| Field | Type | Notes |
|---|---|---|
| `app_name` | `string` | `"deep-finance-train-momentum"` per the Modal `App(...)` call at the top of `src/training/train_deep_momentum.py`. |
| `function_call_id` | `string` | Modal-assigned; visible in the Modal dashboard URL. Not stored in the sidecar (not addressable from the repo). |
| `image_hash` | `string` | Modal-assigned content hash of the container image (re-uses cache when `requirements-train.txt` hasn't changed). Stored in `trained_with`. |
| `gpu_type` | `string` | `"T4"` for Phase 1; future stretch may swap to `"A10G"`. Stored in `trained_with`. |
| `arch` | `string` | CLI arg, `"MLP"` or `"LSTM"`. Stored in the sidecar's `arch` field. |
| `started_at`, `ended_at` | ISO timestamps | Visible in Modal dashboard; sidecar records `trained_on` date only. |

**State transitions**: invoked by `modal run src/training/train_deep_momentum.py
--arch {MLP|LSTM}`; container spins up, runs `train(...)` body, writes
checkpoint to Modal Volume, container tears down. The developer pulls
artifacts with `modal volume get` (per quickstart) and commits them.
