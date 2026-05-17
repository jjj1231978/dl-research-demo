---
description: "Task list for Phase 1 — Momentum Page (Lim et al. 2019)"
---

# Tasks: Phase 1 — Momentum Page (Lim et al. 2019 replication)

**Input**: Design documents from `/specs/002-phase-1-momentum/`

**Prerequisites**: `plan.md` (required), `spec.md` (required), `research.md`, `data-model.md`, `contracts/` (all present), `quickstart.md`

**Tests**: INCLUDED — constitution Principle VI + FR-019 / FR-020 / FR-021 / FR-022 mandate them.

**Organization**: Tasks grouped by user story. Setup (Phase 1) and Foundational (Phase 2) are shared infrastructure; Phases 3–5 are the three user stories in priority order (P1 → P2 → P3); Phase 6 is final polish.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Different file, no in-progress dependency — safe to parallelise.
- **[Story]**: Setup / Foundational / Polish have no story label; user-story tasks use US1 / US2 / US3.
- All paths are repo-root-relative.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Convert the flat `src/data.py` into a package so `src/data/futures/` can live next to it without name conflict; copy the QIS_Commodities futures-machinery files with attribution headers; create the new src/ subpackages; update `requirements-train.txt` if needed.

- [X] T001 Convert `src/data.py` → `src/data/__init__.py` so the directory can also host `src/data/futures/`. Move the existing `src/data.py` body verbatim into `src/data/__init__.py`. Update no imports — `from src.data import ...` continues to work because Python re-exports package `__init__.py` names by default. Delete the old `src/data.py` file.
- [X] T002 Create the new directory scaffold: `mkdir -p src/data/futures src/strategies src/models src/training data/pretrained data/backtests`
- [X] T003 [P] Create `src/data/futures/__init__.py` with the train/test split constants per FR-004a / research.md R3: `TRAIN_START = date(2010, 6, 6)`, `TRAIN_END = date(2019, 12, 31)`, `TEST_START = date(2020, 1, 1)`, `TEST_END = None`, plus a `BCOM_ROOTS` list constant pulled from `~/projects/QIS_Commodities/src/data/contracts.py` (whatever it names the 18 BCOM commodity roots — read that file first to find the exact list)
- [X] T004 [P] Copy `~/projects/QIS_Commodities/src/data/contracts.py` verbatim to `src/data/futures/contracts.py`. Add a 6-line attribution header at the top: `# Copied verbatim from ~/projects/QIS_Commodities/src/data/contracts.py on 2026-05-17.` plus a 1-line note that manual re-sync is the OD-2 maintenance policy. Update any internal imports if the QIS file imports from sibling QIS modules (e.g., if it does `from src.data.fetch import ...`, rewrite to a local relative import). Per FR-002.
- [X] T005 [P] Copy `~/projects/QIS_Commodities/src/data/term_structure.py` verbatim to `src/data/futures/term_structure.py` with the same attribution header pattern. Per FR-002.
- [X] T006 [P] Extract the lake-loading portion of `~/projects/QIS_Commodities/src/data/fetch.py` (the `_load_lake()` function, the `LAKE_PARQUET` constant, the `_lake_cache` / `_lake_lock` module-level state) into a new `src/data/futures/lake_loader.py`. Same attribution header. Do NOT copy the higher-level `fetch_futures_data` or `fetch_all_commodities` — those wrap the lake loader with QIS-project-specific logic; our `scripts/fetch_futures.py` will own that role. Per FR-002.
- [X] T007 [P] Create `src/strategies/__init__.py` (empty package marker).
- [X] T008 [P] Create `src/models/__init__.py` (empty package marker).
- [X] T009 [P] Create `src/training/__init__.py` (empty package marker).
- [X] T010 [P] Update `requirements-train.txt` if needed: add `numpy`, `pandas`, `pyarrow` so the Modal container has them available alongside torch (the Modal `Image.pip_install_from_requirements` builds a clean container; CPU runtime deps don't transfer). Verify the file still excludes `streamlit`, `scikit-learn`, `plotly`, `requests` (training-only image stays lean).
- [X] T011 [P] Add `papers/DeepMomentum.pdf` placeholder (developer drops the arXiv PDF in later) via `touch papers/DeepMomentum.pdf` AND mention it in the `.gitignore` exclusion list if the empty file shouldn't be committed; OR leave `papers/.gitkeep` as the marker and skip this task. (Cleanup task; not load-bearing.) — `papers/.gitkeep` already in place from Phase 0; skipped per task's alternate clause.

**Checkpoint**: `from src.data import ...` continues to work; new package directories exist; QIS copies in place with attribution; `requirements-train.txt` ready for Modal image build. Continue to Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the Reference Model signals, the deep models, and the training script. All three user stories depend on these landing first.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T012 [P] Implement `src/strategies/tsmom.py` per FR-005 — **fully self-contained** (no imports from `~/projects/QIS_Commodities`). Source: `Project_brief.md` §13.5 + Lim et al. 2019 + Baz et al. 2015. Three primary signals plus the MACD primitive:
  - `long_only(prices)` — output is all `1.0`, same shape as input.
  - `sgn_returns(prices, lookback_days=252)` — `np.sign(prices.pct_change(lookback_days))`.
  - `macd_signal(prices, short_span, long_span)` — `(prices.ewm(span=short_span).mean() - prices.ewm(span=long_span).mean()) / prices.rolling(long_span).std()`.
  - `macd_ensemble(prices, shorts=(8, 16, 32), longs=(24, 48, 96))` — mean across the three component MACDs computed via `macd_signal`.
  All four accept a long-format DataFrame (date × contract × price) OR a single price Series; behaviour vectorises naturally. Docstring cites brief §13.5 and the Lim / Baz papers (no QIS reference — QIS implements a different signal family).
- [X] T013 [P] Implement `src/strategies/vol_targeting.py` per FR-006: function `vol_target(signal, target_vol=0.15, ewma_span=60)` that estimates realised volatility via EWMA and rescales the signal to the target annualised vol. Returns the scaled signal in the same shape as input.
- [X] T014 [P] Implement `src/models/deep_momentum.py` per `contracts/deep_momentum_models.md` + FR-007: `DeepMomentumMLP(seq_length=60, n_features=8, hidden=20)` (Flatten → Linear → Softsign chain per brief §13.3) and `DeepMomentumLSTM(seq_length=60, n_features=8, hidden_size=10, num_layers=1)` (LSTM → Linear → Softsign on last time step). Both device-agnostic (Constitution Principle II — no `.cuda()` calls). Both end with `nn.Softsign` per research.md R2.
- [X] T015 Implement `src/training/train_deep_momentum.py` per `contracts/train_deep_momentum_cli.md` + FR-009/010/011/012 + constitution v1.1.0 §"Training workflow (Modal)". File structure: top declares Modal `App` / `Image` (with `pip_install_from_requirements("requirements-train.txt")` + `add_local_python_source("src")`) / `Volume("deep-finance-data")` / `@app.function(gpu="T4", timeout=3600)` + `@app.local_entrypoint()`. Bottom declares the device-agnostic `def train(data_dir, arch, device, checkpoint_dir, max_epochs=200, resume_from=None) -> dict` per contract. Body steps: load `cme_futures.parquet`, build 8-feature tensors (5 return horizons + 3 MACD per FR-008), wrap in `src.torch_data.MyDataset`, instantiate model per `arch`, train with `Adam(lr=1e-3, batch_size=128)` + canonical `src.losses.SharpeLoss` + canonical `src.early_stopper.EarlyStopping(patience=25, min_delta=1e-4)`. Multi-asset aggregation: pre-aggregate `outputs * future_rets` into per-day portfolio return before passing to SharpeLoss (per `src/losses.py` line 12–17 + research.md R5). Set `torch.manual_seed(42)` + `numpy.random.seed(42)` at function entry (research.md R6 reproducibility note). Save final `.pt` to `{checkpoint_dir}/{arch.lower()}_sharpe.pt` + sidecar JSON per `data-model.md` §E3 schema.
- [X] T016 Implement `scripts/fetch_futures.py` per `contracts/fetch_futures_cli.md` + FR-001/002/003/004. argparse skeleton with the documented flags; uses `src.data.futures.lake_loader._load_lake()` to read the source parquet; for each BCOM root, calls `src.data.futures.contracts.build_roll_calendar(...)` → `identify_active_contracts(...)` → `src.data.futures.term_structure.build_ratio_adjusted_series(position=0)` (F0 per OD-2 follow-up); concatenates into long format with `contract` / `asset_class="commodity"` / `price` / `return` columns; writes `cme_futures.parquet` atomically to `data_root()` per FR-022 (Phase 0); writes sidecar JSON per `contracts/fetch_futures_cli.md`.
- [X] T017 Extend `scripts/run_backtests.py` (was empty in Phase 0). Add a `--momentum` flag that produces `data/backtests/momentum_results.parquet` per FR-013 + `data-model.md` §E4: rows for all 5 strategies × 2 vol-scaling × per-contract daily returns. For each (strategy, vol_scaling, contract) triple, compute the daily P&L = position × next-day return. Use the pretrained `mlp_sharpe.pt` and `lstm_sharpe.pt` if present; print a clear warning and skip those two strategies if checkpoints are missing.

**Checkpoint**: `from src.strategies.tsmom import …` works; `from src.models.deep_momentum import …` works; `python -m src.training.train_deep_momentum --help` works; `python scripts/fetch_futures.py --help` works. The actual fetch + training are exercised in US1 (parquet via fetch_futures) and US2 (training on Modal).

---

## Phase 3: User Story 1 — Researcher reads the Momentum page (Priority: P1) 🎯 MVP

**Goal**: The 4-tab Momentum page replaces the Phase 0 placeholder and renders end-to-end with real data when parquet + checkpoints exist; falls back to single-asset toy mode gracefully when they don't.

**Independent Test**: From a clean checkout: (a) `python scripts/fetch_futures.py` produces `cme_futures.parquet`; (b) `streamlit run streamlit_app.py` → click Momentum card → all four tabs render without exception; (c) Tab 4 sub-tabs show the right shape (5-row tables in 4A/4B, single overlay in 4C, three box plots in 4D); (d) integration test passes all 6 parametrised cases.

### Tests for User Story 1

> Write the integration test before the page so the failing test proves the page implementation is needed.

- [ ] T018 [US1] Create `tests/integration/test_momentum_page.py` skeleton per `contracts/momentum_page_ui.md` §"Render assertions". Six parametrised cases: (parquet × deep_model) = {absent, present} × {MLP, LSTM, Both}. Each case uses monkeypatch + tmp_path for env isolation (mirror the Phase 0 `test_landing_page.py` fixture pattern). AppTest.from_file('pages/1_📈_Momentum.py').run() — assert no exception per FR-022.
- [ ] T019 [US1] Add `test_substrate_disclosure_visible` to `tests/integration/test_momentum_page.py`: assert the rendered Tab 1 markdown contains the literal substrings `"BCOM commodity roots"`, `"Pinnacle CLC"`, `"qualitative-ordering claims hold on commodities-only"` per `contracts/momentum_page_ui.md` §"Copy-text invariants".
- [ ] T020 [US1] Add `test_tab4_subtabs_shape` to `tests/integration/test_momentum_page.py` (parquet-present case): assert Tab 4 has 4 sub-tabs labelled 4A/4B/4C/4D; Tab 4A renders a dataframe with 5 rows and the 9 columns from `research.md` R8 (E[Return], Vol, Downside Deviation, MDD, Sharpe, Sortino, Calmar, % +ve Returns, Ave. P / Ave. L).
- [ ] T021 [US1] Add `test_fallback_banner_when_parquet_absent` to `tests/integration/test_momentum_page.py` (parquet-absent case): assert the page renders the `"single-asset toy mode"` banner containing `"scripts/fetch_futures.py"` per `contracts/momentum_page_ui.md` §"Copy-text invariants".

### Implementation for User Story 1

- [ ] T022 [US1] Rewrite `pages/1_📈_Momentum.py` per `contracts/momentum_page_ui.md`. Replace the Phase 0 placeholder body. Header (paper citation + arXiv); sidebar with the six widgets (asset_set, single_asset, vol_scaling, ewma_span, date_range, deep_model) using the documented `key` values; four tabs as a `st.tabs([...])` call; Math panel + Code panel expanders; footer. Reuse `render_data_status_sidebar` from `src.data`.
- [ ] T023 [US1] Tab 1 body — `pages/1_📈_Momentum.py` Tab 1 section. Universe summary (`st.metric` row for # contracts / date range / avg correlation); Plotly Gantt coverage timeline (one row per contract); single-asset price + return-distribution when asset_set=Single asset; substrate disclosure block per FR-014 + spec §"Tab 1 — Problem & Data" + the copy-text invariants from T019.
- [ ] T024 [US1] Tab 2 body — Reference Models. Multi-select for which models to show; Plotly cumulative-return overlay (log scale); inline 3-row mini-table. Uses `src.strategies.tsmom` signals from T012 and `src.strategies.vol_targeting` from T013.
- [ ] T025 [US1] Tab 3 body — Deep Method. Two-column layout (equity curve | position over time); model-card badge reading the JSON sidecar of the selected `arch`; "Train your own (tiny subsample)" expander with a 30-second-hard-cap button (FR-018 / Constitution Principle III) that runs `src.training.train_deep_momentum.train()` on a single-contract × 1-year subset locally on CPU. Display loss curve via `st.line_chart` updated each epoch.
- [ ] T026 [US1] Tab 4 body — sub-tabs 4A/4B/4C/4D per `contracts/momentum_page_ui.md` §"Sub-tab 4*". 4A and 4B render `st.dataframe` with the 9-column metrics table from `src.metrics.report_metrics` (per research.md R8) — best-per-column bolded via `column_config`; 4C renders Plotly cumulative-return overlay (log scale, σ_target rescaled); 4D renders three Plotly box plots aggregating per-asset metrics. Source data: rows in `data/backtests/momentum_results.parquet` (from T017).
- [ ] T027 [US1] Math panel + Code panel — Math panel uses `st.latex` for Sharpe-loss + vol-scaling formulas per FR-016. Code panel uses `inspect.getsource` to read `class DeepMomentumMLP` and `class DeepMomentumLSTM` from `src.models.deep_momentum` and `class SharpeLoss` from `src.losses` and renders via `st.code` per FR-016.
- [ ] T028 [US1] Manual fetch + acceptance: run `python scripts/fetch_futures.py` against the developer's databento lake — verify `cme_futures.parquet` lands at `${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet` with 18 distinct contracts × ~16 years of rows. Per FR-001/002/003.
- [ ] T029 [US1] Run `.venv/bin/python -m pytest tests/integration/test_momentum_page.py -v`. All 6 parametrised cases + the 3 named assertion tests pass. If any fail, fix the page; do NOT modify the test to pass.

**Checkpoint**: US1 is functional. The MVP — Momentum page rendering with fetched parquet — is shippable WITHOUT US2 (training) being done, because Tab 3 gracefully degrades when checkpoints are missing.

---

## Phase 4: User Story 2 — Developer retrains via Modal (Priority: P2)

**Goal**: `modal run src/training/train_deep_momentum.py --arch {MLP|LSTM}` produces both checkpoints reproducibly. Developer pulls them via `modal volume get` and commits to `data/pretrained/`.

**Independent Test**: With Modal CLI authenticated and the `deep-finance-data` Modal Volume populated, run `modal run …` twice (once per arch), pull both `.pt` + `.json` files back, verify they load correctly via the smoke test in US3.

### Implementation for User Story 2

These tasks are mostly **manual operator steps** (cannot fully automate without committing Modal credentials, which Constitution Principle III + brief §15.4 forbid). The training script itself was built in T015; this phase is its first real exercise.

- [ ] T030 [US2] Manual pre-flight: `pip install modal` on the developer's machine (not in any requirements file per `plan.md` §"Primary Dependencies"); `modal token new` (opens browser, writes `~/.modal.toml`); `modal volume create deep-finance-data` (idempotent if Phase 0 already created it).
- [ ] T031 [US2] Manual: populate the Modal Volume with the just-fetched parquet: `modal volume put deep-finance-data "$DEEP_FINANCE_DATA_DIR/cme_futures.parquet" /cme_futures.parquet`. Required because Modal containers can't reach the developer's local filesystem.
- [ ] T032 [US2] Train MLP: `modal run src/training/train_deep_momentum.py --arch MLP`. First run builds the Modal image (~2-5 min); subsequent runs reuse the cache (warm start <30 s). Per brief §7.1 estimate: ~10 min wall-clock on T4 for the MLP, ~$0.10 of Modal credit.
- [ ] T033 [US2] Train LSTM: `modal run src/training/train_deep_momentum.py --arch LSTM`. Per brief §7.1: ~15 min wall-clock on T4, ~$0.15 of Modal credit.
- [ ] T034 [US2] Pull checkpoints to repo: `modal volume get deep-finance-data /pretrained/mlp_sharpe.pt ./data/pretrained/mlp_sharpe.pt` (and the `.json` sidecar; repeat for LSTM). Verify the four files exist locally.
- [ ] T035 [US2] Inspect the sidecar JSON: open `data/pretrained/mlp_sharpe.json` and confirm the `trained_with` field starts with `"Modal "` (Constitution Principle III + v1.1.0 amendment via `data-model.md` §E3 validation rules); confirm `arch == "MLP"`, `hyperparameters.hidden_size == 20`.
- [ ] T036 [US2] Re-render the Momentum page: `.venv/bin/streamlit run streamlit_app.py` → click Momentum → Tab 3 now shows the model card with the real `trained_with` + `test_annual_sharpe` values pulled from the JSON sidecar.
- [ ] T037 [US2] Re-run `scripts/run_backtests.py --momentum` (T017): now that `mlp_sharpe.pt` and `lstm_sharpe.pt` exist, the script populates all 5 strategy rows in `momentum_results.parquet` (previously only 3 reference rows because deep checkpoints were missing). Tab 4 sub-tabs 4A/4B/4C/4D now show 5-row tables / 5-trace charts.

**Checkpoint**: US2 is functional. Both deep models are trained, committed, and being served by the live app. Tab 4 exhibits are populated.

---

## Phase 5: User Story 3 — Reviewer audits paper-faithful replication (Priority: P3)

**Goal**: All four unit-test modules pass on a clean checkout, proving signal parity, vol-targeting correctness, checkpoint loadability, and training-body device-agnosticism.

**Independent Test**: `pytest tests/unit/test_tsmom.py tests/unit/test_vol_targeting.py tests/unit/test_checkpoint_smoke.py tests/unit/test_train_smoke.py -v` exits 0.

### Tests for User Story 3

- [ ] T038 [P] [US3] Create `tests/unit/test_tsmom.py` per FR-019 + research.md R7. **Self-contained** — no imports from `~/projects/QIS_Commodities` or any external repo. Fixture: a 1-contract × 1000-day synthetic price series `100 * (1 + rng.normal(0, 0.01, 1000)).cumprod()` with `np.random.default_rng(0)`. Required test cases:
  - `test_long_only_is_constant_one` — `long_only(prices)` returns all 1.0.
  - `test_sgn_returns_monotonic_increasing` — for a linearly-increasing series, `sgn_returns` is `+1` past the lookback window.
  - `test_sgn_returns_monotonic_decreasing` — same shape, `-1`.
  - `test_sgn_returns_constant_input` — constant prices → output `0` (or NaN per `np.sign(0)` convention; assert the actual behaviour).
  - `test_macd_zero_for_constant_series` — `macd_signal(constant_prices, 8, 24)` returns all `0` past the warmup.
  - `test_macd_ensemble_equals_mean_of_components` — formula equality: `macd_ensemble == mean([macd_signal(s, l) for s, l in zip(shorts, longs)])` with `np.allclose(atol=1e-12)`.
  - `test_no_lookahead_leakage` — first `lookback` rows are NaN for `sgn_returns`; first `long_span` rows are NaN for `macd_signal`.
  - `test_output_shape_preserved` — all four functions return same shape as input.
- [ ] T039 [P] [US3] Create `tests/unit/test_vol_targeting.py` per FR-020. Three tests: (a) `test_vol_targeting_hits_target`: σ_target = 0.15 produces scaled-signal realised vol in [0.142, 0.158] (5 % tolerance band) on a 1000-day fixture; (b) `test_vol_targeting_extreme_low`: σ_target = 0.01 produces finite output; (c) `test_vol_targeting_extreme_high`: σ_target = 1.0 produces finite output. All use the same fixture pattern as test_tsmom.
- [ ] T040 [P] [US3] Create `tests/unit/test_checkpoint_smoke.py` per FR-021 + `contracts/deep_momentum_models.md` §"Forward-pass smoke test". Two tests: `test_mlp_checkpoint_loads_and_forwards`, `test_lstm_checkpoint_loads_and_forwards`. Each instantiates the model with default hyperparameters, loads the `.pt` from `data/pretrained/`, runs forward on a zeros tensor of shape `(1, 60, 8)`, asserts output shape `(1, 1)`, finite, in `[-1-1e-6, 1+1e-6]`. Skip the test (pytest.skip) if the `.pt` file doesn't exist — so the suite still passes between US1 ship and US2 ship.
- [ ] T041 [P] [US3] Create `tests/unit/test_train_smoke.py` per Constitution Principle II + plan §"Testing". One test: `test_train_runs_on_cpu_one_epoch` — call `train(data_dir=<tmp_path with a 5-contract × 100-day subset of cme_futures>, arch="MLP", device=torch.device("cpu"), checkpoint_dir=<tmp_path>, max_epochs=1)`, assert the function returns a dict with the expected keys, assert a `.pt` was written to the checkpoint_dir and is loadable.

### Run + verify for User Story 3

- [ ] T042 [US3] Run `.venv/bin/python -m pytest tests/unit/test_tsmom.py -v`. All 8 self-contained property + formula-equality tests pass (FR-019 / SC-003). No external repo dependency at test time.
- [ ] T043 [US3] Run `.venv/bin/python -m pytest tests/unit/test_vol_targeting.py -v`. 3 tests pass (FR-020).
- [ ] T044 [US3] Run `.venv/bin/python -m pytest tests/unit/test_checkpoint_smoke.py -v`. 2 tests pass (or skip if checkpoints not yet committed; will pass once US2 lands). FR-021.
- [ ] T045 [US3] Run `.venv/bin/python -m pytest tests/unit/test_train_smoke.py -v`. 1 test passes. Verifies Constitution Principle II — same training body runs on CPU.

**Checkpoint**: US3 is functional. All test surfaces backing FRs 019/020/021/022 + Principle II + Principle V are green.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T046 [P] Run `.venv/bin/python -m pytest -v` from repo root: full sweep across Phase 0 (21 tests) + Phase 1 (15 tests) ≈ 36 tests. All pass.
- [ ] T047 [P] Re-render `streamlit_app.py` and walk through `quickstart.md` §"Acceptance check" end-to-end. Verify A1–A11 from `spec.md` §"Acceptance criteria". Confirm SC-004 ordering (LSTM ≥ MLP ≥ MACD ≥ Sgn ≥ Long Only) holds on the rescaled view in Tab 4B; if it doesn't, investigate (likely a training-hyperparameter issue, not an implementation bug).
- [ ] T048 [P] Verify SC-007: copy a fresh `.pt` into `data/pretrained/`, restart the live app, confirm Tab 3 model card reflects the new metadata. No cache invalidation beyond `@st.cache_resource`'s normal TTL.
- [ ] T049 [P] Audit Tab 4 cells for `(reproduced here)` badging per Constitution Principle I; spec §"Tab 4" — every metrics cell + every chart trace MUST carry the badge or a `(paper-cited)` equivalent if any paper value is referenced.
- [ ] T050 Final review of `spec.md` against `tasks.md`: confirm every FR (FR-001 through FR-023) maps to at least one task and every SC (SC-001 through SC-007) is verified by either an automated test or a manual acceptance task. Update task IDs in `spec.md` if any FR is uncovered.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: independent — can start immediately after Phase 0 merge.
- **Foundational (Phase 2)**: depends on Setup completion. BLOCKS all user stories.
- **US1 (Phase 3, P1, MVP)**: depends on Foundational. Can ship even without US2 done (Tab 3 degrades gracefully when checkpoints missing).
- **US2 (Phase 4, P2)**: depends on Foundational (T015 — training script). Independent of US1's page implementation but the manual T036/T037 verification steps are more meaningful after US1 ships.
- **US3 (Phase 5, P3)**: depends on Foundational. T040 (checkpoint smoke) is most useful after US2 produces the `.pt` files but the test gracefully skips if absent.
- **Polish (Phase 6)**: depends on all three user stories being functional.

### Within Each User Story

- US1: T018–T021 (tests) before T022–T027 (page sections); T028 (manual fetch) before T029 (run integration tests).
- US2: sequential — pre-flight → put data → train MLP → train LSTM → pull → inspect → re-render → re-backtest.
- US3: T038–T041 (write tests in parallel — different files) then T042–T045 (run in parallel).

### Parallel Opportunities

- All `[P]` tasks in Phase 1 Setup (T003–T011) are independent files.
- T012 (tsmom) and T013 (vol_targeting) and T014 (deep_momentum models) are different files → parallelisable.
- T015 (training script) depends on T014 (uses the model classes); not parallel.
- T016 (fetch_futures) depends on T004/T005/T006 (uses the copied QIS modules); not parallel with those.
- US3 test files (T038–T041) are different files → fully parallelisable.

---

## Parallel Example: Phase 1 Setup

```bash
# After T001 (data package conversion) and T002 (mkdir scaffold), launch in parallel:
Task: "T003 — write src/data/futures/__init__.py constants"
Task: "T004 — copy QIS contracts.py with attribution header"
Task: "T005 — copy QIS term_structure.py with attribution header"
Task: "T006 — extract QIS lake_loader portion of fetch.py"
Task: "T007 — write src/strategies/__init__.py"
Task: "T008 — write src/models/__init__.py"
Task: "T009 — write src/training/__init__.py"
Task: "T010 — update requirements-train.txt"
Task: "T011 — papers/DeepMomentum.pdf placeholder"
```

---

## Implementation Strategy

### MVP First (US1)

1. Complete Phase 1 (Setup) → directory + QIS copies + train/test constants in place
2. Complete Phase 2 (Foundational) → src/strategies + src/models + src/training + scripts/fetch_futures.py + scripts/run_backtests.py extension
3. Complete Phase 3 (US1) → page rewrite + integration test passes; fetch runs once
4. **STOP and VALIDATE**: render the Momentum page; confirm all 4 tabs work; the Deep Method tab shows the "checkpoints not yet committed" notice until US2 lands
5. Shippable: the page is publicly visible with reference-models content in Tab 2, fallback in Tab 3

### Incremental Delivery

1. MVP (above) → branch checkpoint; could merge to main as "Phase 1 Part 1"
2. Add US2 (Modal training + commit `.pt` files) → branch checkpoint
3. Add US3 (unit tests green) → branch checkpoint
4. Polish (Phase 6) → final sweep + acceptance walkthrough → branch ready for merge

### Parallel Team Strategy

With two developers (overkill for a solo Phase 1, but applies in later phases):

1. Both complete Setup + Foundational together
2. Once Foundational is done:
   - Developer A: US1 (Phase 3 — page + integration tests)
   - Developer B: US2 (Phase 4 — Modal training) in parallel
3. Either developer picks up US3 (Phase 5) once their main story finishes

---

## Notes

- `[P]` tasks touch different files and have no in-progress dependency.
- `[Story]` label maps each task to the user story it serves; each story stays independently shippable.
- Manual acceptance tasks (T028, T030–T037, T047–T048) are not automatable but are gates that prove user-visible behaviour matches the spec.
- Commit cadence: one commit per checkpoint (end of Setup, end of Foundational, end of each user story, end of Polish). PR title: `feat: Phase 1 — Momentum Page (Lim et al. 2019)`.
- Once Phase 1 is merged, `/speckit-specify` for Phase 2 (Portfolio) begins on a new branch.
