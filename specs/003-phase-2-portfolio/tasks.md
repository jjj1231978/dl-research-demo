---
description: "Task list for Phase 2 — Portfolio Page (Zhang et al. 2020)"
---

# Tasks: Phase 2 — Portfolio Page (Zhang, Zohren, Roberts 2020 replication)

**Input**: Design documents from `/specs/003-phase-2-portfolio/`
**Prerequisites**: `plan.md` (required), `spec.md` (required),
`research.md`, `data-model.md`, `contracts/` (4 files), `quickstart.md`
**Tests**: INCLUDED — Constitution VI + FR-019/020/021/022.

**Organization**: Setup → Foundational → US1 (P1 MVP) → US2 (P2 Modal) →
US3 (P3 tests) → Polish.

---

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Fetch the FMP parquets: `python scripts/fetch_data.py --universe etf_basket --universe sp500_20 -v`. Verify `data/etf_basket.parquet` and `data/sp500_20.parquet` exist with shares_outstanding column populated. *(Already done in spec-commit 56eff45. Verification only.)*
- [ ] T002 [P] Confirm requirements.txt has scikit-learn (Phase 0 dep needed for `LedoitWolf` covariance). No edits expected. *(Already in.)*

**Checkpoint**: Both input parquets exist; sklearn importable.

---

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T003 [P] Implement `src/strategies/portfolios.py` per `contracts/classical_benchmarks_api.md`. Five functions: `equal_weight`, `min_variance`, `max_diversification`, `diversity_weighted`, `fixed_allocation`. Uses `sklearn.covariance.LedoitWolf` + `scipy.optimize.minimize(method='SLSQP')` for the optimisation methods. Self-contained (no `~/projects/QIS_Commodities` imports). Per FR-005, FR-006.
- [ ] T004 [P] Implement `src/models/deep_portfolio.py:DeepPortfolioMLP` per `contracts/deep_portfolio_model.md` + FR-007. Flatten → Linear(2NT, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, N) → Softmax. Device-agnostic (Principle II). Exposes `.n_assets`, `.lookback`, `.hidden_size` attributes.
- [ ] T005 Implement `src/training/train_deep_portfolio.py` per `contracts/train_deep_portfolio_cli.md` + FR-009/010/011. Modal app at top, device-agnostic `train()` body at bottom. Per-day portfolio aggregation via `(weights * yb).sum(dim=1)`. Reuses `_modal_image_id_or_local` helper imported from `src.training.train_deep_momentum` (DRY).
- [ ] T006 Extend `scripts/run_backtests.py` with `--portfolio` flag. Produces `data/backtests/portfolio_results.parquet` per data-model §E4. Per (universe, method, vol_scaling, cost_rate) tuple, compute the daily portfolio P&L net of turnover cost. Skip Deep Portfolio rows with a warning if checkpoints absent.

**Checkpoint**: `from src.strategies.portfolios import …` works; `from src.models.deep_portfolio import …` works; `python -m src.training.train_deep_portfolio --help` works; `python scripts/run_backtests.py --portfolio --help` shows the new flag.

---

## Phase 3: User Story 1 — Researcher reads the Portfolio page (Priority: P1) 🎯 MVP

**Goal**: 4-tab Portfolio page renders end-to-end with real parquets.

### Tests for US1

- [ ] T007 [US1] Create `tests/integration/test_portfolio_page.py` skeleton per `contracts/portfolio_page_ui.md` §"Render assertions". Six parametrised cases per the contract. Mirrors Phase 1 `test_momentum_page.py` fixture pattern.
- [ ] T008 [US1] Add `test_substrate_disclosure_visible` and `test_fallback_banner_when_parquet_absent` per the copy-text invariants in the contract.
- [ ] T009 [US1] Add `test_tab4_three_panels_shape` (parquet-present case for ETF): assert Tab 4A has 3 dataframes (one per cost/vol panel), each with the documented row/col shape.

### Implementation for US1

- [ ] T010 [US1] Rewrite `pages/2_💼_Portfolio.py` per `contracts/portfolio_page_ui.md`. Replace Phase 0 placeholder. Sidebar with 5 widgets, four tabs, math+code expanders, footer.
- [ ] T011 [US1] Tab 1 body — universe metrics, rolling-corr heatmap, substrate disclosure.
- [ ] T012 [US1] Tab 2 body — multi-select + cumulative-return overlay + weight-evolution heatmap. Uses `src.strategies.portfolios` from T003.
- [ ] T013 [US1] Tab 3 body — model-card badge + Deep Portfolio vs best-classical overlay + weight heatmap.
- [ ] T014 [US1] Tab 4 body — 3 panel dataframes (4A) and 3 side-by-side cumulative-return charts (4B). Filter by sidebar `date_range` (out-of-sample default — Phase 1 retrospective).
- [ ] T015 [US1] Math panel + Code panel. `st.latex` for softmax + Sharpe-loss + vol-scaling. `inspect.getsource` for `DeepPortfolioMLP` and `SharpeLoss`.
- [ ] T016 [US1] Run `pytest tests/integration/test_portfolio_page.py -v`. All cases pass. Fix the page if any fail; do NOT modify tests to pass.
- [ ] T017 [US1] Run `python scripts/run_backtests.py --portfolio -v`. Verify the parquet lands with the right shape (no Deep rows yet — those come after US2).

**Checkpoint**: MVP shippable. Tab 4 shows classical-only metrics; deep rows fill in after US2.

---

## Phase 4: User Story 2 — Developer retrains via Modal (Priority: P2)

> Developer-action: manual Modal commands. Auto-coder cannot launch
> training without permission (cost). Same handoff pattern as Phase 1 US2.

- [ ] T018 [US2] Manual: `modal volume put dl-research-data data/etf_basket.parquet /etf_basket.parquet` and same for sp500_20. Required because Modal containers can't read the developer's local filesystem.
- [ ] T019 [US2] Train ETF: `modal run src/training/train_deep_portfolio.py --universe etfs`. Warm image start ~30 s; training ~5-10 min; ~$0.08 of Modal credit.
- [ ] T020 [US2] Train 20-stock: `modal run src/training/train_deep_portfolio.py --universe 20stock`. ~10-15 min; ~$0.15 of Modal credit.
- [ ] T021 [US2] Pull all 4 files: `modal volume get …` per quickstart §US2.
- [ ] T022 [US2] Inspect sidecar JSONs: confirm `arch="DeepPortfolioMLP"`, `universe ∈ {"etfs","20stock"}`, `trained_with` starts with `"Modal "`.
- [ ] T023 [US2] Re-run `scripts/run_backtests.py --portfolio` to populate Deep Portfolio rows in the backtest panel.

**Checkpoint**: Both checkpoints + JSON sidecars committed; Tab 3 model card populates; Tab 4 shows Deep Portfolio rows.

---

## Phase 5: User Story 3 — Reviewer audits paper-faithful replication (Priority: P3)

- [ ] T024 [P] [US3] Create `tests/unit/test_classical_portfolios.py` per FR-019. Property + formula-equality assertions on the 5 functions in `src/strategies/portfolios.py`. Self-contained.
- [ ] T025 [P] [US3] Create `tests/unit/test_softmax_head.py` per FR-020. Random input → output sums to 1 ± 1e-6 + all weights in [0, 1].
- [ ] T026 [P] [US3] Create `tests/unit/test_portfolio_checkpoint_smoke.py` per FR-021. Two tests (etfs / 20stock), pytest.skip if `.pt` absent.
- [ ] T027 [P] [US3] Create `tests/unit/test_portfolio_train_smoke.py` per FR-021. One CPU smoke test on a synthetic fixture.
- [ ] T028 [US3] Run `pytest tests/unit/test_classical_portfolios.py -v`. All pass.
- [ ] T029 [US3] Run `pytest tests/unit/test_softmax_head.py -v`. Passes.
- [ ] T030 [US3] Run `pytest tests/unit/test_portfolio_checkpoint_smoke.py -v`. Two pass (or skip pre-US2).
- [ ] T031 [US3] Run `pytest tests/unit/test_portfolio_train_smoke.py -v`. Passes.

**Checkpoint**: Full unit-test sweep green.

---

## Phase 6: Polish & Cross-Cutting

- [ ] T032 [P] Run `pytest -v` from repo root: ~70 tests pass (49 Phase 0+1 + 21 Phase 2).
- [ ] T033 Verify acceptance scenarios A1–A5 from spec.md US1 against the live local Streamlit app.
- [ ] T034 Audit Tab 4 cells for `(reproduced here)` badging per Constitution Principle I.
- [ ] T035 Final review: every FR / SC covered. Update tasks.md status markers.
- [ ] T036 Merge `003-phase-2-portfolio` → main. Push to GitHub + HF. Wait for HF rebuild. Verify live page at https://huggingface.co/spaces/JJ-JIN12345/dl-research-demo.

---

## Dependencies

- **Setup (T001-T002)**: independent.
- **Foundational (T003-T006)**: depends on Setup. BLOCKS user stories.
- **US1 (T007-T017)**: depends on Foundational. Can ship without US2 (Tab 3 graceful-degrades).
- **US2 (T018-T023)**: depends on Foundational (T005 — training script). Manual operator.
- **US3 (T024-T031)**: depends on Foundational. T026 most useful after US2 but skips cleanly if absent.
- **Polish (T032-T036)**: after all three user stories.

## Parallel Opportunities

- T003 / T004 — different files, no shared state.
- T024 / T025 / T026 / T027 — four independent test files.

## MVP Scope

- Setup (T001-T002) → Foundational (T003-T006) → US1 (T007-T017): minimum for a shippable Portfolio page.
- US2 + US3 + Polish bring it to feature complete.
