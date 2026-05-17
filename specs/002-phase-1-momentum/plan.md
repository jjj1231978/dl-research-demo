# Implementation Plan: Phase 1 — Momentum Page (Lim et al. 2019)

**Branch**: `002-phase-1-momentum` | **Date**: 2026-05-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-phase-1-momentum/spec.md`

## Summary

Build the first per-paper page of the Deep Finance Showcase — a replication
of Lim, Zohren, Roberts (2019) Exhibits 2 / 3 / 4 / 5 on the developer's CME
futures dataset. Three user stories prioritise the work: researcher reads
the full page (P1), developer retrains via Modal (P2), reviewer audits the
paper-faithful replication (P3). Phase 1 lands the CME fetch script
(continuous-futures construction copied from `~/projects/QIS_Commodities`),
the Reference Model signals, the deep MLP + LSTM with Sharpe loss trained on
Modal T4, two pretrained checkpoints, and the fully built four-tab Momentum
page replacing the Phase 0 placeholder. Universe scope is the 18 BCOM
commodity roots (per OD-1); F0 ratio-adjusted continuous series (per OD-2
follow-up); 60/40 chronological train/test split (per OD-3).

## Technical Context

**Language/Version**: Python 3.11 (pinned in HF Dockerfile, Phase 0).
Local dev and Modal container also 3.11.

**Primary Dependencies** (deltas from Phase 0):

- Production runtime (`requirements.txt`) — unchanged. CPU torch, streamlit,
  pandas, pyarrow, numpy, scikit-learn, plotly, requests, python-dotenv,
  pytest, pytest-cov.
- Training runtime (`requirements-train.txt`) — extend the Phase 0 seed
  (single `torch` line) with the Modal-specific deps that get baked into
  the container image. Phase 1 adds: nothing required beyond `torch`
  itself; the Modal SDK is installed in the developer's local venv, not
  in the container image. (wandb stays out of Phase 1.)
- Developer-machine dev deps — `pip install modal` once. Not in any
  requirements file because the live HF app does not call Modal.

**Storage**:

- `${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet` — long-format
  18-commodity-root × ~16-year price + return series, F0
  ratio-adjusted, ~70 k rows (~5 MB).
- `data/pretrained/mlp_sharpe.pt` + `mlp_sharpe.json` sidecar.
- `data/pretrained/lstm_sharpe.pt` + `lstm_sharpe.json` sidecar.
- `data/backtests/momentum_results.parquet` — 5 strategies × 2
  vol-scaling conditions × per-asset daily returns, schema per FR-013.
- Modal Volume `dl-research-data` (already created in Phase 0 quickstart)
  — input data + intermediate training checkpoints.
- FMP per-ticker cache `~/data_lake/fmp/deep-finance/` — unchanged from
  Phase 0; Phase 1 does not add FMP fetches.

**Testing**:

- Unit: `pytest tests/unit/test_tsmom.py` (signal parity with
  QIS_Commodities to 1e-10 tolerance); `pytest
  tests/unit/test_vol_targeting.py` (σ_target realised-vol ≈ 0.15);
  `pytest tests/unit/test_checkpoint_smoke.py` (both `.pt` files load,
  non-NaN forward pass).
- Integration: `pytest tests/integration/test_momentum_page.py` —
  AppTest across {parquet-present, parquet-absent} × {MLP, LSTM, Both}
  sidebar states. Six parametrised cases.
- Modal training body MUST also be runnable as plain Python on CPU
  (`python -m src.training.train_deep_momentum --arch MLP --max-epochs 1`)
  for unit-test smoke runs — verified via a CPU-only path in
  `test_train_smoke.py`.

**Target Platform**:

- Production: HF Space CPU Basic (2 vCPU, 16 GB RAM), Linux container,
  Python 3.11 slim. Same as Phase 0.
- Training: Modal serverless GPU container (T4 default for Phase 1 per
  brief §7.1 cost estimate); same Python 3.11; auto-torn-down after
  function exit.

**Project Type**: Single project (Phase 0 layout retained). No
backend/frontend split. Phase 1 adds five new src/ subdirectories
(`src/data/futures/`, `src/strategies/`, `src/models/`, `src/training/`)
plus `scripts/fetch_futures.py` and a `scripts/run_backtests.py`
extension — all paths listed in brief §5.

**Performance Goals**:

- Momentum page warm render: < 3 s (constitution v1.1.0 §"Tech Stack &
  Runtime Constraints"; SC-001 reaches < 10 s including nav).
- Tab switches: < 500 ms.
- "Train your own" subsample: ≤ 30 s wall-clock on HF CPU Basic
  (constitution Principle III; FR-018; SC-005).
- Modal T4 full training: ~10 min MLP + ~15 min LSTM per brief §7.1
  (estimated ~$0.25 of Modal credit).
- `scripts/fetch_futures.py` wall-clock: best-effort; reads the existing
  lake parquet (~4 M rows already on disk) and outputs ~70 k rows; well
  under 30 s on the developer's machine.

**Constraints**:

- HF CPU Basic 16 GB RAM ceiling — `cme_futures.parquet` ~5 MB, both
  `.pt` checkpoints < 100 kB each (MLP ~10 k params, LSTM ~5 k params per
  brief §7.1), `momentum_results.parquet` < 50 MB long-format with 18
  contracts × 10 strategy series × ~16 years. Well within budget.
- No `.cuda()` calls anywhere in `src/` (Constitution Principle II).
- No `optimizer.step()` reachable from `streamlit_app.py` or
  `pages/1_📈_Momentum.py` except the explicit "Train your own" opt-in
  with 30-s hard cap (Constitution Principle III; FR-018).
- Canonical `SharpeLoss` from `src/losses.py` MUST be used as-is — no
  extension, no parameter additions (Constitution Principle V; FR-009;
  spec §"Scope discipline").
- Copied QIS_Commodities files MUST carry header comments naming the
  source path and date copied (FR-002; OD-2).

**Scale/Scope**:

- 18 BCOM commodity roots (per OD-1) — sourced from
  `~/projects/QIS_Commodities/src/data/contracts.py` constants.
- 5 trading strategies (Long Only, Sgn(Returns), MACD, MLP-Sharpe,
  LSTM-Sharpe) per FR-013.
- 2 deep-model architectures (MLP, LSTM) per FR-007.
- 1 training script (`src/training/train_deep_momentum.py`) parameterised
  by `--arch ∈ {MLP, LSTM}`.
- 1 fetch script (`scripts/fetch_futures.py`).
- ~3 unit-test modules + 1 integration-test module added.
- Files touched: ~25 new files; the `pages/1_📈_Momentum.py` placeholder
  from Phase 0 is fully overwritten.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against the six principles in `.specify/memory/constitution.md`
v1.1.0.

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | PASS | Page replicates named exhibits (2/3/4/5) from Lim et al. 2019. Reproduced cells in Tab 4 will carry `(reproduced here)` badging; the commodities-only substrate vs paper's Pinnacle CLC multi-asset-class is disclosed in Tab 1 and Tab 4D. MLP / LSTM architectures match paper Table 3 hidden sizes (20, 10). Continuous-futures construction uses the QIS_Commodities convention copied verbatim. |
| II. Device-Agnostic Torch (NON-NEGOTIABLE) | PASS | Models in `src/models/deep_momentum.py` use `nn.Module` with no `.cuda()` calls. Training body in `src/training/train_deep_momentum.py:def train(...)` detects device via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. The Modal `@app.function(gpu="T4")` decorator at the top of the file is the only GPU-specific glue. Unit smoke test (`test_train_smoke.py`) exercises the CPU path. |
| III. Two-Compute-Environment Discipline | PASS | Training script is a Modal app per constitution v1.1.0 §"Training workflow (Modal)". Committed `.pt` files produced offline; `trained_with` sidecar field includes Modal image hash. "Train your own" button on the Momentum page has 30-s hard cap on a single-contract subsample (FR-018). `requirements.txt` (CPU torch) and `requirements-train.txt` (GPU torch, baked into Modal image) remain separate files; Modal SDK not in either. |
| IV. Data-as-Artifact (parquet-not-DB) | PASS | `cme_futures.parquet`, `momentum_results.parquet`, `.pt` + JSON sidecars all on disk under `data/` or `${DEEP_FINANCE_DATA_DIR}`. No database, no cache server. |
| V. Pre-existing Canon | PASS | `src/losses.py` (Sharpe loss) used as-is — Phase 1 spec §"Scope discipline" explicitly forbids extension. `src/torch_data.py` `MyDataset` used as the dataloader. `src/early_stopper.py` `EarlyStopping` used in the training loop with `savepath` as first positional argument per its canonical signature. `src/metrics.py` `report_metrics()` consumed by Tab 4A/4B for the metric table (the 6 keys extended in Phase 0 — Sortino, Calmar, MDD, hit rate, avg P / avg L — are exactly what the paper's Exhibits 2/3 columns ask for). |
| VI. Test Critical Paths | PASS | Unit tests for new signal math (FR-019, FR-020), checkpoint-load smoke test (FR-021), integration test for page render across data states (FR-022). No UI snapshot tests; no browser-driven Selenium/Playwright tests. Matches "test critical paths, not everywhere" constitution mandate. |

**No violations. No Complexity Tracking entries required.**

The re-check after Phase 1 design (data-model + contracts) is recorded at
the end of this plan; if any design choice introduces friction with the
principles, it MUST be documented in the Complexity Tracking table below
with explicit justification.

## Project Structure

### Documentation (this feature)

```text
specs/002-phase-1-momentum/
├── plan.md              # This file (/speckit-plan command output)
├── spec.md              # /speckit-specify output (exists)
├── research.md          # Phase 0 output — 8 decision records
├── data-model.md        # Phase 1 output — 5 entities
├── quickstart.md        # Phase 1 output — operator walkthrough for US1/US2/US3
├── contracts/           # Phase 1 output — 4 interface contracts
│   ├── fetch_futures_cli.md          # scripts/fetch_futures.py contract
│   ├── deep_momentum_models.md       # src/models/deep_momentum.py public surface
│   ├── train_deep_momentum_cli.md    # src/training/train_deep_momentum.py (Modal + local)
│   └── momentum_page_ui.md           # pages/1_📈_Momentum.py sidebar + tab contracts
├── checklists/
│   └── requirements.md   # /speckit-specify validation checklist (exists)
└── tasks.md             # Phase 2 output (/speckit-tasks command — NOT created here)
```

### Source Code (repository root)

Per `Project_brief.md` §5 (authoritative). New paths in Phase 1 marked
`NEW`; paths from Phase 0 that get a body update marked `UPDATED`.

```text
deep-finance-showcase/
├── streamlit_app.py
├── pages/
│   ├── 1_📈_Momentum.py                       # UPDATED — replace Phase 0 placeholder with full 4-tab page
│   ├── 2_💼_Portfolio.py                      # unchanged (Phase 2)
│   └── 3_📖_Order_Book.py                     # unchanged (Phase 3)
├── src/
│   ├── __init__.py
│   ├── losses.py                              # unchanged (canonical)
│   ├── early_stopper.py                       # unchanged (canonical)
│   ├── torch_data.py                          # unchanged (canonical)
│   ├── metrics.py                             # unchanged (Phase 0 extended; consumed by Tab 4A/4B)
│   ├── data.py                                # unchanged (Phase 0 surface; Momentum page reads via load_universe)
│   ├── universes.py                           # unchanged (cme_futures universe definition already present)
│   ├── fmp.py                                 # unchanged (not used by Phase 1)
│   ├── data/                                  # NEW package — copied futures-loading machinery
│   │   ├── __init__.py                        #   exposes futures helpers + train/test split constants
│   │   └── futures/                           #   per FR-002 / OD-2 — copied from ~/projects/QIS_Commodities/src/data/
│   │       ├── __init__.py                    #     train/test constants (TRAIN_END = 2020-01-01)
│   │       ├── contracts.py                   #     copy of QIS contracts.py with attribution header
│   │       ├── term_structure.py              #     copy of QIS term_structure.py with attribution header
│   │       └── lake_loader.py                 #     extracted lake-loading code from QIS fetch.py
│   ├── strategies/                            # NEW package — Reference Model signals
│   │   ├── __init__.py
│   │   ├── tsmom.py                           #   Long Only / Sgn(Returns) / MACD ensemble (mirrors QIS trend.py)
│   │   └── vol_targeting.py                   #   σ_target = 15% EWMA vol scaling
│   ├── models/                                # NEW package
│   │   ├── __init__.py
│   │   └── deep_momentum.py                   #   DeepMomentumMLP + DeepMomentumLSTM
│   ├── training/                              # NEW package
│   │   ├── __init__.py
│   │   └── train_deep_momentum.py             #   Modal app + device-agnostic train() body
├── data/
│   ├── aapl.csv
│   ├── portfolio_data.csv
│   ├── etf_basket.parquet                     # produced by Phase 0 fetch_data.py
│   ├── sp500_20.parquet                       # produced by Phase 0 fetch_data.py
│   ├── cme_futures.parquet                    # NEW — produced by Phase 1 fetch_futures.py
│   ├── pretrained/                            # NEW directory
│   │   ├── mlp_sharpe.pt                      #   NEW — produced by Modal training
│   │   ├── mlp_sharpe.json                    #   NEW — sidecar (brief §7.3 + Modal image hash per v1.1.0)
│   │   ├── lstm_sharpe.pt                     #   NEW
│   │   └── lstm_sharpe.json                   #   NEW
│   └── backtests/                             # NEW directory
│       └── momentum_results.parquet           #   NEW — produced by scripts/run_backtests.py
├── scripts/
│   ├── fetch_data.py                          # unchanged (Phase 0)
│   ├── fetch_futures.py                       # NEW — 18 BCOM continuous F0 ratio-adjusted parquet
│   └── run_backtests.py                       # NEW — extended in Phase 2/3 for those pages too
├── notebooks/
│   └── reference/                             # unchanged
├── papers/
│   └── DeepMomentum.pdf                       # NEW (optional) — drop the arXiv PDF here for offline reference
├── tests/
│   ├── unit/
│   │   ├── test_metrics.py                    # unchanged
│   │   ├── test_data_root.py                  # unchanged
│   │   ├── test_tsmom.py                      # NEW — FR-019 parity check vs QIS
│   │   ├── test_vol_targeting.py              # NEW — FR-020 σ_target realised-vol check
│   │   ├── test_checkpoint_smoke.py           # NEW — FR-021 .pt files load + non-NaN forward
│   │   └── test_train_smoke.py                # NEW — CPU smoke run of train() body (constitution II)
│   └── integration/
│       ├── test_landing_page.py               # unchanged
│       └── test_momentum_page.py              # NEW — FR-022 AppTest across data + sidebar states
├── requirements.txt                           # unchanged
├── requirements-train.txt                     # unchanged from Phase 0 seed (no new deps for Phase 1)
└── README.md                                  # unchanged (Phase 4 polish)
```

**Structure Decision**: Same single-project Streamlit layout. Phase 1 adds
four new `src/` subpackages (`data/futures/`, `strategies/`, `models/`,
`training/`) plus `data/pretrained/` and `data/backtests/` artifact
directories. Every path is listed in brief §5; no layout deviation.

The `src/data/` directory deserves a small note: Phase 0 has `src/data.py`
as a flat module. Phase 1 introduces `src/data/futures/` as a subpackage.
To avoid a name conflict, Phase 1 will convert `src/data.py` → `src/data/__init__.py`
(carry forward the existing public surface — `data_root`, `BUNDLED_CSV_DIR`,
exceptions, `DataSnapshot`, `get_data_snapshot`, `load_universe`,
`render_data_status_sidebar`) and add `src/data/futures/` next to it.
Existing `from src.data import ...` lines keep working because Python's
package-init re-export semantics. This is a code-organisation tweak only;
no public-surface change to the contracts from Phase 0.

## Post-Design Constitution Re-Check

After producing `research.md`, `data-model.md`, `contracts/`, and
`quickstart.md`, re-evaluating the six principles:

| Principle | Status | Notes |
|---|---|---|
| I. Paper-Faithful Replication | PASS | `contracts/momentum_page_ui.md` requires every Tab 4 reproduced cell to carry an explicit `(reproduced here)` badge; `data-model.md` records the commodities-only substrate disclosure. |
| II. Device-Agnostic Torch | PASS | `contracts/deep_momentum_models.md` and `contracts/train_deep_momentum_cli.md` both forbid `.cuda()` calls in the importable surface; the train script's body is testable on CPU. |
| III. Two-Compute-Environment Discipline | PASS | `contracts/train_deep_momentum_cli.md` documents the dual-mode invocation (`python -m …` for CPU smoke, `modal run …` for GPU); sidecar JSON schema in `data-model.md` requires `trained_with` to include Modal image hash. |
| IV. Data-as-Artifact | PASS | All entities in `data-model.md` map to on-disk files. No DB. |
| V. Pre-existing Canon | PASS | `contracts/deep_momentum_models.md` and `contracts/train_deep_momentum_cli.md` both reference `SharpeLoss` from `src/losses.py` as-is and require the training script to import without modification. |
| VI. Test Critical Paths | PASS | `quickstart.md` walkthrough exercises all four pytest modules; no UI snapshot or browser-driven tests introduced. |

Re-check passes. No Complexity Tracking entries.

## Complexity Tracking

No Constitution Check violations in either gate. This table is
intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| _(none)_  | _(n/a)_    | _(n/a)_                             |
