# Feature Specification: Phase 1 — Momentum Page (Lim et al. 2019 replication)

**Feature Branch**: `002-phase-1-momentum`

**Created**: 2026-05-17

**Status**: Draft

**Input**: User description (paraphrased): Build the first per-paper page of
the Deep Finance Showcase — replication of Lim, Zohren, Roberts (2019)
Exhibits 2/3/4/5 on the developer's CME futures dataset. End-to-end deep
momentum: MLP + LSTM trained on Modal with the Sharpe loss already in
`src/losses.py`, two pretrained checkpoints committed, full 4-tab Momentum
page swapping in for the Phase 0 placeholder. Reference-model signals
(Long Only / Sgn(Returns) / MACD ensemble) mirror the conventions in
`~/projects/QIS_Commodities/src/signals/`.

## Clarifications

### Session 2026-05-17

- Q: How should we scope Phase 1's CME universe given that the data lake only has the 18 BCOM commodity roots? → A: **Ship Phase 1 with the 18 BCOM commodity roots only.** Page discloses the substrate mismatch with the paper (CME commodities vs the paper's Pinnacle CLC multi-asset-class). The qualitative-ordering claim (LSTM > MLP > MACD > Sgn > Long Only) is asserted on commodities-only. FI / equity-index / FX continuous futures are deferred to a future phase that would require expanding `~/data_lake/databento/futures/` first.
- Q: How should Phase 1 consume the continuous-futures construction logic that already exists in `~/projects/QIS_Commodities`? → A: **Copy the relevant ~200 lines (`src/data/contracts.py` + `src/data/term_structure.py` + the lake-loader portion of `src/data/fetch.py`) into the Showcase repo at `src/data/futures/`, with header comments crediting the QIS source.** Showcase repo stays self-contained for clone-and-go; future updates to QIS conventions require manual re-sync (tracked as a documentation note in the copied files). Submodule and rewrite-from-scratch alternatives rejected.
- Q: Which curve position should the ratio-adjusted continuous series use? → A: **F0 (front-month).** Matches the paper's Pinnacle CLC substrate, matches the QIS `build_ratio_adjusted_series(position=0)` default, matches industry convention for trend-following CTAs. Expiry-week noise mitigated by the QIS default `roll_offset_bdays = -5` (roll a business week before the delivery month). The `position` parameter stays exposed for a future stretch (e.g., a carry signal in a later phase).
- Q: How should we split the available 2010-06-06 → 2026-05-05 data into train/test given the paper's 1990–2010 / 1995–2015 windows aren't reachable? → A: **Chronological 60/40: train 2010-06-06 → 2019-12-31, test 2020-01-01 → 2026-05-05.** Aligns with the brief's "extension to recent regimes" framing in §4.1; test window covers COVID + 2022 rate-hike + recent commodity moves — rich out-of-sample variety. Walk-forward folds rejected for Phase 1 (3× Modal compute; can revisit in Phase 4 stretch).

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Researcher reads the Momentum story end-to-end (Priority: P1)

A researcher who arrived from the Lim et al. 2019 arXiv link clicks the
Momentum card on the landing page and lands on a fully working page. They
read the problem framing in Tab 1, see classical Reference Models in Tab 2,
explore the pretrained deep models in Tab 3, and find the headline
Exhibit 2/3/4/5 replication tables and charts in Tab 4. Every claim the
page makes is sourced to the paper; the math panel underneath shows the
Sharpe-loss derivation; the code panel shows the MLP/LSTM body verbatim.

**Why this priority**: This is the headline deliverable of Phase 1 — the
visible artifact that demonstrates the paper-faithful replication promise
from Constitution Principle I. Without it, Phase 1 ships nothing publicly
visible even if the data pipeline + training + backtests all work behind
the scenes.

**Independent Test**: Open the deployed app, click the Momentum card, walk
through all four tabs without errors. Verify (a) the Reference Models in
Tab 2 produce sensible cumulative-return overlays, (b) Tab 3 loads the
pretrained MLP and LSTM checkpoints and renders the equity curve, (c) the
Tab 4A and 4B metrics tables have 5 rows and the documented column set,
(d) Tab 4D shows three box plots aggregating across the futures universe.

**Acceptance Scenarios**:

1. **Given** the deployed app with `cme_futures.parquet` present and both
   pretrained checkpoints committed, **When** the researcher clicks the
   Momentum card on the landing page, **Then** `pages/1_📈_Momentum.py`
   loads within 3 seconds with all four tabs functional and no exception.
2. **Given** the running Momentum page, **When** the researcher selects
   Tab 4 → sub-tab 4A, **Then** a 5-row table appears (Long Only,
   Sgn(Returns), MACD, MLP-Sharpe, LSTM-Sharpe) with the column set
   E[Return] / Vol / Downside Deviation / MDD / Sharpe / Sortino / Calmar /
   % +ve Returns / Ave. P / Ave. L, all annualized, best-per-column bolded.
3. **Given** the running Momentum page, **When** the researcher toggles
   the deep-model selector to "Both overlaid" in the sidebar, **Then**
   Tab 3 renders one equity curve per model (MLP and LSTM) on the same
   chart with a clear legend.

---

### User Story 2 — Developer trains the deep models on Modal (Priority: P2)

The developer pushes a code change to `src/training/train_deep_momentum.py`
or `src/models/deep_momentum.py`, then runs `modal run
src/training/train_deep_momentum.py --arch MLP` from their VS Code
terminal. Training runs on a Modal T4 container (~10–15 minutes), saves
intermediate checkpoints to the Modal Volume, writes the final
`mlp_sharpe.pt` + JSON sidecar. The developer pulls the artifacts via
`modal volume get`, commits to `data/pretrained/`, and the next app start
serves the new model.

**Why this priority**: This is the production reproducibility path. Anyone
forking the project must be able to retrain the models on Modal without
operator intervention beyond the Modal CLI commands documented in the
quickstart. Without this, the committed checkpoints are black boxes.

**Independent Test**: From a clean checkout with `requirements-train.txt`
installed and `modal token new` already run, execute `modal run
src/training/train_deep_momentum.py --arch MLP`. Verify the container
spins up, the training loop runs to completion (or to a `--max-epochs`
cap for the smoke run), and the resulting `.pt` file is downloadable from
the Modal Volume via `modal volume get`.

**Acceptance Scenarios**:

1. **Given** the developer has authenticated with Modal and the
   `deep-finance-data` Modal Volume is populated with `cme_futures.parquet`,
   **When** they run `modal run src/training/train_deep_momentum.py
   --arch MLP --max-epochs 5`, **Then** the function executes on a T4
   container, prints per-epoch loss, writes the checkpoint to the Volume,
   and prints the `modal volume get` command for download.
2. **Given** the same developer running `--arch LSTM`, **Then** the same
   pipeline produces `lstm_sharpe.pt` with the LSTM architecture (verified
   by inspecting the saved state-dict keys).
3. **Given** the developer runs `python -m src.training.train_deep_momentum
   --arch MLP --max-epochs 1` locally on CPU, **Then** the same training
   body executes (slower, smaller batch) without GPU and produces a
   structurally valid `.pt` (Constitution Principle II — device agnostic).

---

### User Story 3 — Reviewer verifies paper-faithful replication (Priority: P3)

A reviewer (the developer's collaborator, or a future-them) audits the
page against the paper. They click into the math panel to verify the
Sharpe-loss derivation matches Eq. 4 of the paper, expand the code panel
to confirm the MLP architecture matches the paper's Table 3, and check
that Tab 4A's "MLP-Sharpe" row has reasonable Sharpe and that the
qualitative ordering of strategies (LSTM > MLP > MACD > Sgn > Long Only)
holds on the rescaled view. The unit-test suite passes; the
checkpoint-load smoke test passes; the data-pipeline parity test against
`QIS_Commodities`'s Sgn(Returns) and MACD backtests passes within
numerical tolerance.

**Why this priority**: Constitution Principle I (paper-faithful replication)
makes verifiability the test of credibility. A reviewer who can't trace a
displayed number back to a paper equation, or who finds a unit-test gap
on the new strategy module, has reason to distrust everything else.

**Independent Test**: From a clean checkout: (a) `pytest
tests/unit/test_tsmom.py tests/unit/test_vol_targeting.py
tests/unit/test_checkpoint_smoke.py` passes, (b) `pytest
tests/integration/test_momentum_page.py` passes, (c) the math panel
renders the volatility-scaling formula visually identical to paper Eq. 1,
(d) the code panel content matches `src/models/deep_momentum.py` and
`src/losses.py` byte-for-byte.

**Acceptance Scenarios**:

1. **Given** a clean checkout with venv installed, **When** the reviewer
   runs `pytest tests/unit/test_tsmom.py`, **Then** the Sgn(Returns) and
   MACD signal outputs on a fixed-seed price series match the
   corresponding `~/projects/QIS_Commodities/src/signals/trend.py`
   outputs within absolute tolerance 1e-10 (data-pipeline parity).
2. **Given** the same checkout, **When** the reviewer runs `pytest
   tests/unit/test_checkpoint_smoke.py`, **Then** both `mlp_sharpe.pt`
   and `lstm_sharpe.pt` load on CPU into their corresponding `nn.Module`
   and produce a non-NaN forward pass on a unit-shaped input.
3. **Given** the Tab 4A and 4B tables, **When** the reviewer compares the
   "Sharpe" column values for MLP-Sharpe and LSTM-Sharpe on the rescaled
   (σ_target = 15%) view, **Then** LSTM ≥ MLP ≥ MACD ≥ Sgn(Returns) ≥
   Long Only — the qualitative ordering from the paper's Exhibits 2/3
   holds.

---

### Edge Cases

- **CME parquet absent**: page MUST gracefully fall back to single-asset
  toy mode using `data/aapl.csv`; Tab 4 sub-tabs display a banner
  "single-asset toy mode — Exhibits 2/3/4/5 require
  `data/cme_futures.parquet` (run `scripts/fetch_futures.py` locally;
  fetches the 18 BCOM commodity roots from
  `~/data_lake/databento/futures/L0`)" rather than crashing.
- **Pretrained checkpoint missing**: Tab 3 displays a "checkpoint not yet
  committed — see `quickstart.md` §Train" notice rather than erroring
  out; Tab 2 (Reference Models) continues to work.
- **Pretrained checkpoint architecture mismatch**: e.g., loading a
  state-dict trained with hidden=10 into a hidden=20 MLP. Page MUST
  surface a clear error naming the mismatch rather than producing a
  corrupted forward pass.
- **Modal training container preempted mid-run**: the training script
  must resume from the most recent intermediate checkpoint in the Modal
  Volume, not start from scratch.
- **"Train your own" exceeds 30 s budget**: the opt-in button MUST hard-
  abort the training thread at 30 s and display the loss curve as-of the
  abort point, not let the user-facing UI hang.
- **σ_target slider extreme values** (e.g., 0.01 or 1.0): rescaled
  equity curves MUST remain finite and renderable; the metrics table MAY
  show inflated MDD / Calmar but MUST NOT crash.

## Requirements *(mandatory)*

### Functional Requirements

**Data pipeline**

- **FR-001**: A new fetch script MUST read the developer's local CME
  futures data at `~/data_lake/databento/futures/L0/ohlcv-1d/` and
  produce `${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet` covering **the
  18 BCOM commodity roots** (per OD-1; FI / equity / FX deferred). FR-022
  from Phase 0 honoured — output location is env-var-overridable.
- **FR-002**: The fetch script MUST construct **ratio-adjusted, F0
  (front-month) continuous** futures series from individual contracts
  using the QIS_Commodities algorithm. The implementation MUST be copied
  verbatim from `~/projects/QIS_Commodities/src/data/` into the Showcase
  repo at `src/data/futures/` (per OD-2): specifically
  `contracts.py` (roll calendar + active-contract identification),
  `term_structure.py` (term-structure assembly + `build_ratio_adjusted_series`
  with `position=0` for F0), and the lake-loader portion of `fetch.py`.
  Each copied file MUST carry a header comment naming the QIS source and
  the date copied. The default `roll_offset_bdays = -5` is retained
  (roll a business week before delivery-month start).
- **FR-003**: The output parquet schema is long-format with columns
  `date` (date32[day]), `contract` (string — continuous-series root, e.g.
  "CL", "ZC"), `asset_class` (string), `price` (float64, ratio-adjusted
  F0), `return` (float64, daily simple return per `pct_change`). For
  Phase 1 all rows have `asset_class = "commodity"`; the column is
  retained for forward compatibility with later phases that expand the
  universe.
- **FR-004**: The fetch script MUST be re-runnable: re-execution against
  unchanged source data produces a byte-identical output parquet
  (deterministic ordering of rows; no embedded timestamps in the data
  payload).
- **FR-004a**: The 60/40 chronological train/test split (per OD-3) MUST
  be encoded as module-level constants in `src/data/futures/__init__.py`
  (or equivalent), not duplicated across training script and backtest
  script: train cutoff = 2020-01-01 (exclusive in train, inclusive in
  test).

**Reference-Model signals**

- **FR-005**: `src/strategies/tsmom.py` MUST be a **self-contained**
  implementation of three signal functions per `Project_brief.md` §13.5
  (the project's authoritative signal-math reference) — no imports from
  `~/projects/QIS_Commodities` at module-load or test time:
  (a) **Long Only**: `X_t = 1` for every row.
  (b) **Sgn(Returns)**: `np.sign(price.pct_change(252))` per
       Moskowitz / Ooi / Pedersen 2012.
  (c) **MACD ensemble**: average of three vol-normalised MACDs across
       the (8/24), (16/48), (32/96) `(short_span, long_span)` timescales
       per Baz et al. 2015. Each MACD = `(ewm(short).mean() -
       ewm(long).mean()) / rolling(long).std()`.
  The QIS `signals/trend.py` is informational only — it implements a
  DIFFERENT signal family (multi-lookback `sign(momentum)`, no MACD).
  Phase 1's signal code does not depend on QIS at all.
- **FR-006**: `src/strategies/vol_targeting.py` MUST scale a raw signal
  to a target annualized volatility using EWMA-estimated realized
  volatility. Default σ_target = 0.15 (paper Eq. 1). σ_target and the
  EWMA span are function parameters.

**Deep Momentum models**

- **FR-007**: `src/models/deep_momentum.py` MUST expose two PyTorch
  `nn.Module` classes: `DeepMomentumMLP` (2-layer, hidden=20 per paper
  Table 3) and `DeepMomentumLSTM` (1-layer, hidden=10). Both accept
  inputs of shape `(batch, seq_length, n_features=8)` and output a
  scalar position in `(-1, +1)` via Softsign. Both MUST be
  device-agnostic per Constitution Principle II.
- **FR-008**: The feature vector at each timestep MUST be
  8-dimensional: five normalized return horizons (1d, 1m, 3m, 6m, 1y)
  plus three MACD indicator values at the (8/24), (16/48), (32/96)
  timescales. Construction conventions mirror
  `notebooks/reference/03_deep_momentum_strategy.ipynb`.

**Training (Modal)**

- **FR-009**: `src/training/train_deep_momentum.py` MUST declare a Modal
  `App` / `Image` / `Volume` / `@app.function(gpu="T4")` decorator at the
  top of the file per constitution v1.1.0 §"Training workflow (Modal)".
  The training function loss MUST be `SharpeLoss` from `src/losses.py`
  used as-is — no extension, no other loss families (Constitution
  Principle V; Phase 1 §"Out of Scope").
- **FR-010**: The training body MUST be runnable both ways: locally on
  CPU via `python -m src.training.train_deep_momentum --arch MLP
  --max-epochs N` (smoke test) and on Modal T4 via `modal run
  src/training/train_deep_momentum.py --arch MLP` (real training).
- **FR-011**: Training MUST checkpoint to the Modal Volume every N
  epochs and accept a `resume_from` argument so a container preemption
  does not destroy progress (Constitution Principle III).
- **FR-012**: The committed `.pt` files (`data/pretrained/mlp_sharpe.pt`,
  `lstm_sharpe.pt`) MUST be accompanied by JSON sidecars per brief §7.3
  schema, with `trained_with` naming the Modal image hash and GPU type
  (Constitution Principle III).

**Backtest precomputation**

- **FR-013**: `scripts/run_backtests.py` MUST produce
  `data/backtests/momentum_results.parquet` containing daily returns for
  all 5 strategies under both vol-scaled and unscaled conditions (10
  strategy series total). Per-asset daily returns also stored so
  Exhibit 5's box plots compute without re-running the deep models.

**Momentum page**

- **FR-014**: `pages/1_📈_Momentum.py` MUST replace the Phase 0
  placeholder and render four tabs (Problem & Data / Reference Models /
  Deep Method / Exhibits 2/3/4/5) plus a sidebar with the controls
  enumerated in the feature description.
- **FR-015**: Tab 4 sub-tabs (4A Exhibit 2, 4B Exhibit 3, 4C Exhibit 4,
  4D Exhibit 5) MUST follow the paper's structure: 4A and 4B are 5-row
  metrics tables (computed without / with vol scaling respectively); 4C
  is a single log-scale cumulative-return overlay (panel (a) of paper
  Exhibit 4); 4D is three Plotly box plots aggregating per-asset metrics
  across the futures universe.
- **FR-016**: The Math panel MUST render Eq. 1 (volatility scaling) and
  the Sharpe-loss derivation via `st.latex`. The Code panel MUST display
  `class DeepMomentumMLP` / `class DeepMomentumLSTM` from
  `src/models/deep_momentum.py` and `SharpeLoss` from `src/losses.py`
  using `st.code`.
- **FR-017**: The page MUST handle the missing-parquet edge case
  gracefully by falling back to single-asset toy mode using
  `data/aapl.csv` and surfacing a banner that names the fetch script.
- **FR-018**: The "Train your own (tiny subsample)" button MUST hard-cap
  the training thread at 30 seconds (Constitution Principle III) on the
  CPU Basic HF runtime and label the resulting display "illustrative
  only".

**Tests**

- **FR-019**: A unit-test module for the Reference-Model signals
  (`tests/unit/test_tsmom.py`) MUST use **property-based and
  formula-equality assertions** on the in-repo signal implementations
  (no `sys.path` import of QIS or any other external repo — keeps the
  repo self-contained). Required cases at minimum:
  (a) `long_only`: output is all `1.0`.
  (b) `sgn_returns`: monotonically increasing prices → output `+1`
       past the lookback window; monotonically decreasing → `-1`;
       constant → `0`.
  (c) `macd_signal`: constant prices → output `0` (numerator collapses).
  (d) `macd_ensemble`: equals the elementwise mean of the three
       component MACDs (`np.allclose(..., atol=1e-12)`).
  (e) All four functions: output shape equals input shape; first
       `lookback` rows are NaN (no look-ahead leakage).
- **FR-020**: A unit-test module for the vol-targeting helper
  (`tests/unit/test_vol_targeting.py`) MUST assert: (a) σ_target = 0.15
  produces scaled-signal realized volatility ≈ 0.15 within 5 % tolerance
  over a representative test window; (b) edge cases for σ_target ∈
  {0.01, 1.0} produce finite output.
- **FR-021**: A checkpoint-load smoke test
  (`tests/unit/test_checkpoint_smoke.py`) MUST load both committed `.pt`
  files on CPU into their corresponding `nn.Module` and produce a
  non-NaN forward pass on a unit-shaped input.
- **FR-022**: An integration test
  (`tests/integration/test_momentum_page.py`) MUST verify the page
  renders under `streamlit.testing.v1.AppTest` with parquet-present +
  parquet-absent + each of the three sidebar deep-model selector states.
  No exception raised; the four tab labels appear in the rendered output.

### Key Entities

- **ContinuousFuturesContract**: A ratio-adjusted continuous time series
  for a futures root (e.g., "CL" = WTI Crude, "ZC" = Corn). Attributes:
  root symbol, asset class, member individual contracts, roll calendar.
  Constructed by `scripts/fetch_futures.py` from raw individual-contract
  data in the developer's databento lake.
- **MomentumSignal**: A daily position-in-`(-1, +1)` series for a
  single contract under a single strategy. Strategy is one of {Long
  Only, Sgn(Returns), MACD, MLP-Sharpe, LSTM-Sharpe}. Optionally
  vol-scaled.
- **DeepMomentumCheckpoint**: A `.pt` file + JSON sidecar (per brief
  §7.3 schema). Sidecar fields: `trained_on` (date), `trained_with`
  (Modal image hash + GPU), `arch` (MLP or LSTM), `data_range`,
  `hyperparameters`, `final_metrics`, `git_commit`.
- **BacktestPanel**: The long-format DataFrame at
  `data/backtests/momentum_results.parquet` with columns `date`,
  `strategy`, `contract`, `daily_return`, `vol_scaling` (bool).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher landing on the Momentum page can navigate
  through all four tabs and find Tab 4's headline tables / charts within
  10 seconds of clicking the Momentum card on the landing page.
- **SC-002**: 100% of the Tab 4 sub-tabs (4A, 4B, 4C, 4D) render data
  consistent with their paper-referenced exhibit shape (5 rows × the
  documented column set in 4A/4B; single overlay chart in 4C; three box
  plots in 4D).
- **SC-003**: 100 % of the property-based + formula-equality test cases
  in `tests/unit/test_tsmom.py` pass on a clean checkout (per FR-019).
  No external repo dependency at test time — the test suite runs
  identically on any machine that has the Showcase repo and a venv.
- **SC-004**: On the rescaled (σ_target = 15%) view in Tab 4B, the
  ordering LSTM ≥ MLP ≥ MACD ≥ Sgn(Returns) ≥ Long Only holds for the
  Sharpe column on the 18 BCOM commodity universe (paper-qualitative-
  consistency check; commodities-only substrate per OD-1 disclosed on
  the page).
- **SC-005**: The "Train your own (tiny subsample)" button completes
  within 30 seconds wall-clock on HF CPU Basic without OOM (per FR-018,
  Constitution Principle III).
- **SC-006**: 100% of the four unit/integration tests
  (`test_tsmom.py`, `test_vol_targeting.py`, `test_checkpoint_smoke.py`,
  `test_momentum_page.py`) pass on a clean checkout.
- **SC-007**: A push of either pretrained-checkpoint file to GitHub
  results in the deployed Hugging Face Space serving the new model on
  the next page load.

## Assumptions

- Constitution v1.1.0 is the active governance. Modal is the GPU host;
  no Colab references in new code.
- Phase 0 deliverables are merged or merge-ready (parquet loader,
  bundled CSV fallbacks, extended `src/metrics.py`, `src/losses.py`
  canonical SharpeLoss). Phase 1 builds on top of them.
- `src/strategies/tsmom.py` is **self-contained** — math comes from
  `Project_brief.md` §13.5 (Long Only / Sgn(Returns) / MACD ensemble)
  and the Lim et al. 2019 + Baz et al. 2015 papers, NOT from
  `~/projects/QIS_Commodities`. The QIS `signals/trend.py` is informational
  only (it implements a different signal family — multi-lookback
  `sign(momentum)`, no MACD). Tests assert properties + formula equality
  on our own implementation; nothing in `tests/unit/` or `src/` reaches
  into the QIS repo.
- `~/projects/QIS_Commodities/src/data/` is the canonical source for the
  continuous-futures construction logic (roll calendar + ratio-adjustment
  + lake loader). Per OD-2, this code IS copied into the Showcase repo
  at `src/data/futures/` with attribution headers so the repo is also
  self-contained on the data-loading side.
- Modal credit budget for Phase 1: estimated ~$0.25 (T4 × 25 minutes
  for both checkpoints per brief §7.1). The developer pays this directly.
- Per OD-3: train/test split is chronological 60/40, train 2010-06-06 →
  2019-12-31, test 2020-01-01 → 2026-05-05 (and forward as new data
  lands).
- Per OD-1: Phase 1 universe is 18 BCOM commodity roots only; the page
  discloses the substrate mismatch with the paper's Pinnacle CLC
  multi-asset-class dataset.
- Per OD-2: continuous-futures construction code is copied verbatim from
  `~/projects/QIS_Commodities/src/data/` into `src/data/futures/` with
  attribution header comments. The algorithm is ratio-adjustment of the
  F0 (front-month) curve with a 5-business-day pre-delivery roll.
- Daily returns convention: simple returns (`pct_change`) — the paper
  does not specify but the existing source notebooks under
  `notebooks/reference/` use simple returns; matching for consistency.

## Out of Scope

- WaveNet architecture (developer scope says MLP + LSTM only).
- Loss-function comparison across {Sharpe, Average Returns, MSE, Binary}
  — Phase 1 ships only Sharpe per brief §6 Page 1 scope-discipline note.
- Transaction-cost sensitivity (Exhibit 7 of paper) — Phase 4 stretch.
- LIME / SHAP / gradient-sensitivity visualizations — out of scope for
  the Momentum page (Phase 2 Portfolio page has Figure 6 as its analog,
  also stretch).
- The live FMP refresh button (Phase 0 captured the input but didn't
  wire it; deferring to Phase 4).
- Portfolio page (Phase 2), Order Book page (Phase 3), HF deployment
  (Phase 4).
- Per OD-1: FI, equity-index, and FX continuous futures — deferred to a
  later phase that would first expand `~/data_lake/databento/futures/`
  with the relevant databento datasets.
- Per F0 lock-in (OD-2 follow-up): F1 / F2 / further-out continuous
  positions and carry-yield signals built on the F0–F1 spread — deferred;
  the `position` parameter remains exposed for a future stretch.
- The "futures universe selection" UI exposing all 18 commodity roots
  individually — Tab 4D aggregates over the universe; per-contract
  inspection in Tab 1 is single-asset only.

## Open Decisions

All four Phase-1-blocking decisions identified during specification are
resolved in the `## Clarifications` section above (OD-1 universe scope,
OD-2 continuous-futures construction, OD-2 follow-up F0 position, OD-3
train/test split, session 2026-05-17). The brief's §12 lists seven
further open decisions that do not affect Phase 1 scope; they will be
re-surfaced as `[NEEDS CLARIFICATION]` markers in later phases'
specifications where they are blocking.
