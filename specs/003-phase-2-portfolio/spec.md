# Feature Specification: Phase 2 — Portfolio Page (Zhang et al. 2020 replication)

**Feature Branch**: `003-phase-2-portfolio`

**Created**: 2026-05-17

**Status**: Draft

**Input**: User description (paraphrased): Build the second per-paper page of
the Deep Finance Showcase — replication of Zhang, Zohren, Roberts (2020)
**Table 1** and **Figure 3** on two equity-and-multi-asset universes.
End-to-end deep portfolio optimization: an MLP with softmax output (long-only
weights summing to 1), trained on Modal with the Sharpe loss already in
`src/losses.py`, two pretrained checkpoints committed, full 4-tab Portfolio
page swapping in for the Phase 0 placeholder. Classical benchmarks (Equal
Weight, Minimum Variance, Maximum Diversification, Diversity-Weighted
Portfolio) computed in-page from the FMP-fetched parquets.

## Clarifications

### Session 2026-05-17

- Q: Which universes should Phase 2 cover? → A: **ETF basket (VTI/AGG/DBC/VIXY)
  + 20-stock S&P subset.** Two trained checkpoints (`deep_portfolio_etfs.pt`,
  `deep_portfolio_20stock.pt`). The 100-stock universe is deferred to a
  future phase pending resolution of OD-2 (the Phase 0 100-ticker
  confirmation flow is still in `proposed.json` state; Phase 2 does not
  resolve it).
- Q: Where should price data come from? → A: **Run `scripts/fetch_data.py`
  using the developer's FMP Premium key (already in local `.env`).** This is
  the first real exercise of the Phase 0 FMP fetcher. Output:
  `${DEEP_FINANCE_DATA_DIR}/etf_basket.parquet` and
  `${DEEP_FINANCE_DATA_DIR}/sp500_20.parquet`. Bundled
  `data/portfolio_data.csv` remains as a CSV fallback for offline / no-key
  runs (FR-022 / Phase 0 contract).
- Q: Loss-function scope? → A: **Sharpe only** (canonical
  `src/losses.py:SharpeLoss` used as-is, Constitution Principle V). One
  deep checkpoint per universe = 2 total. Paper's 4-loss comparison
  (Sharpe / Mean Returns / MSE / Binary) is out of scope. Page surfaces the
  choice in the Tab 3 model-card badge.
- Q: σ_target value? → A: **0.10 (10% annualized)**, matching paper §4.4 /
  Table 1. Differs from Phase 1's 0.15 because Phase 1 followed Lim et al.
  2019's choice; Phase 2 follows Zhang et al. 2020's choice.
- Q: Train/test split? → A: **Per-universe chronological with a common
  test_start of 2020-01-01** for ease of cross-universe comparison.
  Actual FMP-fetch ranges:
  - **etf_basket**: 2011-01-03 → 2026-05-15 (VIXY inception). Train:
    2011-01-03 → 2019-12-31 (~9 years); test: 2020-01-01 → 2026-05-15
    (~6.4 years).
  - **sp500_20**: 2006-06-30 → 2026-05-15 (FMP coverage limit for the
    oldest ticker — bundled portfolio_data.csv goes back to 1992 but FMP
    only returns this far). Train: 2006-06-30 → 2019-12-31 (~13.5 years);
    test: 2020-01-01 → 2026-05-15 (~6.4 years).
  Constants live in `src/training/train_deep_portfolio.py`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Researcher reads the Portfolio story end-to-end (Priority: P1) — MVP

A researcher who arrived from the Zhang et al. 2020 arXiv link clicks the
**💼 Portfolio** card on the landing page and lands on a fully working
page. They pick a universe in the sidebar (ETF basket by default), read
the problem framing in Tab 1 with the rolling-correlation heatmap, see
classical benchmarks (Equal Weight, Min Variance, Max Diversification,
DWP) in Tab 2, explore the pretrained deep model in Tab 3, and find the
headline **Table 1** and **Figure 3** replication in Tab 4. Every claim
the page makes is sourced to the paper; the math panel shows the softmax
+ Sharpe-loss derivation; the code panel shows the MLP body verbatim.

**Why this priority**: This is the headline deliverable of Phase 2 — the
visible artifact that demonstrates the paper-faithful replication promise
from Constitution Principle I. Without it, Phase 2 ships nothing publicly
visible.

**Independent Test**: From a clean checkout: (a) run `python
scripts/fetch_data.py --universe etf_basket --universe sp500_20` →
produces both parquets; (b) `streamlit run streamlit_app.py` → click
Portfolio card → all four tabs render without exception; (c) Tab 4A shows
a 3-row × 8-column Table 1 replication for the selected universe; (d) Tab
4B shows three side-by-side cumulative-return panels matching paper Fig 3
shape.

**Acceptance Scenarios**:

1. **Given** the deployed app with both parquets and both pretrained
   checkpoints present, **When** the researcher clicks the Portfolio
   card, **Then** `pages/2_💼_Portfolio.py` loads within 3 seconds with
   all four tabs functional and no exception.
2. **Given** the running Portfolio page with **ETF basket** selected,
   **When** Tab 4A renders, **Then** the table has rows for all 8 in-scope
   methods (Allocations 1-4, Equal Weight, Min Variance, Max
   Diversification, DWP, Deep Portfolio) — 9 rows total — with the 8
   metric columns from paper Table 1.
3. **Given** the running Portfolio page with **20-stock** selected,
   **When** Tab 4A renders, **Then** the table has rows for the universal
   methods only (Equal Weight, Min Variance, Max Diversification, DWP,
   Deep Portfolio) — 5 rows total — with the same 8 columns.
4. **Given** Tab 4B with either universe, **When** it renders, **Then**
   it shows three side-by-side Plotly cumulative-return panels
   (no-vol-scaling @ C=0.01% / vol-scaled @ C=0.01% / vol-scaled @
   C=0.10%) overlaying every in-scope method on log scale.
5. **Given** the parquet for the selected universe is absent, **When**
   the page renders, **Then** the page falls back to the bundled CSV
   (portfolio_data.csv for 20-stock; etf basket has no CSV fallback so
   shows a banner with `python scripts/fetch_data.py` instruction).

### User Story 2 — Developer retrains via Modal (Priority: P2)

A developer who wants to refresh the pretrained checkpoints runs `modal
run src/training/train_deep_portfolio.py --universe etfs` (and again with
`--universe 20stock`). Each run trains a single-universe MLP with Sharpe
loss, writes the `.pt` + `.json` sidecar to the `dl-research-data` Modal
Volume; the developer pulls them via `modal volume get` and commits.

**Why this priority**: P2 because US1 ships gracefully without it (the
page surfaces "checkpoint missing" notes; the user can still see
classical benchmarks).

**Independent Test**: `modal run src/training/train_deep_portfolio.py
--universe etfs` exits 0; the `.pt` + `.json` files are visible in the
Modal Volume; `modal volume get` retrieves them; the loaded `state_dict`
passes the smoke test in US3 on CPU.

### User Story 3 — Reviewer audits paper-faithful replication (Priority: P3)

A reviewer who wants to verify the page's claims runs `pytest tests/unit/`
and reads the four new test modules: `test_classical_portfolios.py`
(formula equality of Equal Weight, Min Variance, Max Diversification, DWP
against in-repo references), `test_softmax_head.py` (Deep Portfolio output
sums to 1 ± 1e-6 and all weights are in [0, 1]),
`test_portfolio_checkpoint_smoke.py` (both `.pt` files load on CPU and
produce non-NaN forward passes), and `test_portfolio_train_smoke.py` (the
device-agnostic `train()` body runs one epoch on a CPU subsample).

## Functional Requirements *(mandatory)*

**Data layer**:

- **FR-001**: `scripts/fetch_data.py` (Phase 0) MUST produce
  `etf_basket.parquet` and `sp500_20.parquet` under `data_root()` when
  invoked with `--universe etf_basket --universe sp500_20`. Both parquets
  follow the Phase 0 schema (FR-001 / data-model E1: `date`, `symbol`,
  `open`, `high`, `low`, `close`, `volume`, optional `shares_outstanding`).
- **FR-002**: `sp500_20` ticker set MUST be the 20 names already defined
  in `src/universes.py` (`AAPL, ABT, AEP, AXP, BAC, CI, GD, GE, HON, MMM,
  MO, MRK, NEM, NKE, NSC, PFE, PG, PTC, SNA, SO`) — unchanged from Phase 0.
- **FR-003**: The page MUST honor `DEEP_FINANCE_DATA_DIR` for the parquet
  source (Phase 0 FR-022) and fall back to bundled
  `data/portfolio_data.csv` for the 20-stock universe when no parquet is
  found.
- **FR-004**: The Modal training script MUST be invokable as `modal run
  src/training/train_deep_portfolio.py --universe {etfs|20stock}` and as
  `python -m src.training.train_deep_portfolio --universe etfs
  --max-epochs 1` for the device-agnostic CPU smoke (Constitution
  Principle II).
- **FR-004a**: Per-universe chronological 60/40 train/test split per the
  Clarifications session. Constants live in
  `src/training/train_deep_portfolio.py`.

**Classical benchmarks**:

- **FR-005**: `src/strategies/portfolios.py` MUST implement, **fully
  self-contained**:
  - `equal_weight(N)` — returns `(1/N, ..., 1/N)`.
  - `min_variance(returns, lookback=50, shrinkage="ledoit_wolf")` — Markowitz
    1952 minimum-variance weights using `sklearn.covariance.LedoitWolf` (or
    `ShrunkCovariance` if Ledoit-Wolf converges poorly) computed on the
    rolling `lookback`-day window per paper §4.2.
  - `max_diversification(returns, lookback=50)` — Choueifaty & Coignard
    2008 weights `w* = argmax (w' σ) / sqrt(w' Σ w)` where σ is the
    diagonal of per-asset stdevs.
  - `diversity_weighted(market_caps, p=0.5)` — Samo & Vervuurt 2016
    `w_i ∝ μ_i^p / Σ μ_j^p` where `μ` is normalized market cap. Skip
    with a warning when `shares_outstanding` is not in the parquet (FR-001
    optional field).
- **FR-006**: For the ETF basket only, also include the four fixed
  allocations from paper Table 1: 25/25/25/25, 50/10/20/20, 10/50/20/20,
  40/40/10/10 (order: shares/bonds/commodities/vol = VTI/AGG/DBC/VIXY).

**Deep model**:

- **FR-007**: `src/models/deep_portfolio.py:DeepPortfolioMLP` — MLP
  reading flattened `(lookback=50, 2N)` features (close + return per
  asset), producing a Softmax-normalized weight vector of length `N`.
  Hidden cascade per `notebooks/reference/02_deep_portfolio_optimization.ipynb`.
  Device-agnostic (Principle II).
- **FR-008**: Output MUST satisfy `w_i >= 0` and `sum(w_i) == 1` ± 1e-6
  (Softmax invariant). Asserted by `test_softmax_head.py`.

**Training**:

- **FR-009**: `src/training/train_deep_portfolio.py` is a Modal app per
  constitution v1.1.0 §"Training workflow (Modal)" with
  `@app.local_entrypoint()` taking `--universe ∈ {etfs, 20stock}`.
- **FR-010**: Loss = canonical `src/losses.py:Neg_Sharpe`. Multi-asset
  aggregation: per-day portfolio return = `(weights * next_day_returns).sum()`
  (NOT mean — weights already normalize). Then `Neg_Sharpe(portfolio_returns)`
  over the time series. Same per-day-Sharpe pattern as Phase 1's refactor.
- **FR-011**: Two `.pt` files + JSON sidecars committed to
  `data/pretrained/`:
  `deep_portfolio_etfs.{pt,json}`, `deep_portfolio_20stock.{pt,json}`.

**Page rendering**:

- **FR-012**: `pages/2_💼_Portfolio.py` replaces the Phase 0 placeholder.
  Header (paper citation + arXiv link), sidebar with: universe picker
  (ETF default), transaction-cost selector (C ∈ {0.01%, 0.10%}),
  vol-scaling toggle (default ON, σ_target=10%), rolling-window slider
  (default 50 days), backtest date range slider. Four tabs.
- **FR-013**: Tab 1 (Problem & Data): universe summary, rolling-correlation
  heatmap, return / vol / pairwise-correlation summary stats.
- **FR-014**: Tab 2 (Classical Benchmarks): multi-select toggle for
  methods, Plotly cumulative-return overlay (log scale, vol-scaling
  toggle controls panel), weight-evolution heatmap for the selected
  classical method.
- **FR-015**: Tab 3 (Deep Method): model-card badge from the
  `.json` sidecar (`trained_with`, `arch`, `final_metrics.test_annual_sharpe`),
  Deep Portfolio equity curve overlaid vs best classical benchmark,
  weight-over-time heatmap (same axes as Tab 2).
- **FR-016**: Tab 4 sub-tabs:
  - **4A** — Table 1 replication: a `st.dataframe` with 3 rows (no-vol /
    vol+0.01% / vol+0.10%) × all-in-scope-methods columns, 8 metric
    columns mapped from `src.metrics.report_metrics`. Best per-column
    bolded.
  - **4B** — Figure 3 replication: three side-by-side Plotly
    cumulative-return charts (log scale), one per (vol_scaling, cost)
    panel, all methods overlaid.
- **FR-017**: Math panel (`st.latex`): Softmax constraint, Sharpe loss,
  vol-scaling formula. Code panel: `inspect.getsource` on
  `DeepPortfolioMLP` + canonical `SharpeLoss`.
- **FR-018**: "Train your own (tiny subsample)" expander: surfaces the
  CLI command (Phase 1 precedent — in-page 30s training thread deferred).

**Tests**:

- **FR-019**: `tests/unit/test_classical_portfolios.py` — property +
  formula-equality tests for each of the four classical methods, using a
  fixed-seed synthetic return series. **Self-contained** (no external repo).
- **FR-020**: `tests/unit/test_softmax_head.py` — for a random 1×50×2N
  input, model output sums to 1 ± 1e-6 and every weight is in [0, 1].
- **FR-021**: `tests/unit/test_portfolio_checkpoint_smoke.py` — both
  `.pt` files load on CPU under the matching `DeepPortfolioMLP` instance
  and produce non-NaN forward passes. Skip if absent (US1 ships before US2).
- **FR-022**: `tests/integration/test_portfolio_page.py` — six
  parametrised cases (parquet × universe) × (parquet × deep_model), pages
  render without exception. Bundled-CSV fallback test asserts the
  banner appears when parquet absent.

## Key Entities

- **PortfolioWeights** — daily long-only weight vector for one (method,
  universe) pair. Per-day, length N, sums to 1, all entries in [0, 1].
  Held in memory; not persisted between runs.
- **DeepPortfolioCheckpoint** — `.pt` + JSON sidecar. Sidecar per
  Phase 1 `data-model.md` §E3 schema with `arch="DeepPortfolio"` and
  `universe ∈ {"etfs", "20stock"}` as a new field.
- **PortfolioBacktestPanel** — `data/backtests/portfolio_results.parquet`:
  long-format rows `(date, universe, method, vol_scaling, cost_rate,
  portfolio_return)`. Generated by `scripts/run_backtests.py --portfolio`.
- **UniverseDataset** — re-uses Phase 0's `DataSnapshot` / `load_universe`
  surface; no new entity needed.

## Success Criteria *(mandatory)*

- **SC-001**: Phase 0 FMP fetcher produces both parquets without errors
  on the developer's FMP key.
- **SC-002**: All 4 tabs of `pages/2_💼_Portfolio.py` render in ≤ 3 s on
  HF CPU Basic (first paint) for both universes.
- **SC-003**: Tab 4A Table 1 row ordering on the ETF basket shows
  qualitatively consistent ordering with paper Table 1 — **Deep Portfolio
  is in the top 3 by Sharpe across all three (vol_scaling, cost) panels**
  (the paper's strong claim). Specific Sharpe magnitudes will differ
  because our test window is 2020+ vs the paper's pre-2020.
- **SC-004**: Tab 4B Figure 3 panels are visually similar to paper Fig 3:
  Deep Portfolio's cumulative curve sits in the upper half of the
  overlay across the test window.
- **SC-005**: `pytest tests/unit/test_classical_portfolios.py
  tests/unit/test_softmax_head.py
  tests/unit/test_portfolio_checkpoint_smoke.py
  tests/integration/test_portfolio_page.py` exits 0 from a clean checkout.
- **SC-006**: Modal training run for each universe takes ≤ 15 min wall
  clock on T4; cost ≤ $0.15 per universe; total ≤ $0.30.

## Scope

**In scope**:
- ETF basket and 20-stock universes
- Four classical methods (Equal Weight, Min Variance, Max
  Diversification, Diversity-Weighted Portfolio) for both universes
- Four fixed allocations for ETF basket only
- Deep Portfolio MLP with Softmax + Sharpe loss
- Tab 4A (Table 1) and Tab 4B (Figure 3) replication
- Math panel + Code panel
- Two pretrained checkpoints committed
- Tests for classical methods, softmax invariant, checkpoint smoke,
  integration

**Out of scope** (deferred to Phase 4 or future):
- 100-stock universe (pending OD-2 resolution)
- LSTM variant of Deep Portfolio (paper's actual choice — paper-faithful
  MLP from `02_deep_portfolio_optimization.ipynb` is sufficient for
  Phase 2 acceptance)
- Tab 4C — Figure 6 (gradient-based sensitivity heatmap)
- Tab 4D — Figure 5 (COVID stress test)
- Four-loss comparison (paper compares Sharpe / Mean Returns / MSE /
  Binary)
- Transaction-cost sensitivity beyond the two-rate sweep (paper §4.5)

## Critical References

- Paper: Zhang, Zohren, Roberts (2020). *Deep Learning for Portfolio
  Optimization*. arXiv:2005.13665.
- Brief: `Project_brief.md` §§ 6 Page 2, 11 Phase 2.
- Constitution: v1.1.0.
- Phase 0 contracts:
  `specs/001-phase-0-skeleton-data/contracts/fetch_data_cli.md`,
  `specs/001-phase-0-skeleton-data/contracts/data_loader_api.md`.
- Phase 1 lessons learned:
  - Per-day portfolio Sharpe-loss aggregation (commit 93578ca)
  - Tab 4 must honor sidebar date_range to show out-of-sample only
  - Backtest parquet must be committed to git for live-app rendering

## Open Decisions

- **OD-1** (deferred): 100-stock universe ticker list. Phase 2 ships
  without it. When the developer is ready, run `scripts/fetch_data.py
  --universe sp500_100` once to seed the `proposed.json`, review/edit,
  rename to `confirmed.json`, then re-run.
- **OD-2** (Phase 2 internal): When the FMP `shares-outstanding`
  endpoint is not available on the developer's tier, DWP is skipped with
  a warning per Phase 0 OD-3 (warn-and-skip). Already-committed Phase 0
  resolution; no action needed in Phase 2.
