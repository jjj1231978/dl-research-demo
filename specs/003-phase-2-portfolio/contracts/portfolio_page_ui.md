# Contract: `pages/2_💼_Portfolio.py`

**Type**: Streamlit page; replaces Phase 0 placeholder.
**Renders on**: HF Space CPU Basic + local dev.
**Consumed by**: `tests/integration/test_portfolio_page.py`.

Mirrors Phase 1's Momentum page structure: header → sidebar → four tabs →
math/code expanders → footer.

## Page header

- Title: `💼 Portfolio — Zhang, Zohren, Roberts (2020)`
- Subtitle: *"Deep Learning for Portfolio Optimization"*, arXiv link
- Elevator pitch (≤ 2 lines): "A long-only portfolio with softmax output,
  trained to maximize Sharpe directly — no covariance matrix, no
  expected-return forecast."

## Sidebar (top to bottom)

| Widget | `key` | Default | Notes |
|---|---|---|---|
| `st.selectbox` "Universe" | `portfolio_universe` | `"ETF basket"` | `"ETF basket"` / `"20-stock"`. Maps to `etfs` / `20stock` internally. |
| `st.selectbox` "Cost rate" | `portfolio_cost_rate` | `"0.01%"` | `"0.01%"` (= 0.0001) or `"0.10%"` (= 0.0010). Drives the "active" cost panel highlighting in Tab 2/4. |
| `st.checkbox` "Volatility scaling (σ_target = 10%)" | `portfolio_vol_scaling` | `True` | Toggles between vol-scaled and raw views. |
| `st.slider` "Rolling window for classical methods (days)" | `portfolio_lookback` | `50` | Used by Min Variance / Max Diversification rolling cov estimate. |
| `st.slider` "Backtest date range" | `portfolio_date_range` | `(2020-01-01, today)` | Filters Tab 2 + Tab 4 panels to honour the paper's out-of-sample convention. |
| `st.divider` | | | |
| `render_data_status_sidebar` from Phase 0 | | | Re-uses the data-status component. |

## Tabs

`st.tabs(["1. Problem & Data", "2. Classical Benchmarks", "3. Deep Method",
"4. Table 1 / Figure 3"])`

### Tab 1 — Problem & Data

- Paragraph: Markowitz 1952 + Zhang 2020 contribution
- Rolling-correlation heatmap (annual avg) for the selected universe
- `st.metric` row: # assets, date range, avg pairwise corr, # methods
  available (depends on whether DWP can run)
- Substrate disclosure: "ETF basket replicates the paper's exact universe
  (VTI/AGG/DBC/VIXY); 20-stock substitutes a sector-spread S&P 500
  subset. The paper's qualitative claim (Deep Portfolio in top 3 by
  Sharpe) is asserted on these substrates; absolute Sharpe values differ
  from the paper because our test window is 2020+ vs the paper's
  pre-2020."

### Tab 2 — Classical Benchmarks

- Multi-select for which benchmark methods to show (default: all
  available for the universe)
- Plotly cumulative-return chart (log scale), one trace per selected
  method, controlled by sidebar `vol_scaling` + `cost_rate`
- Weight-evolution heatmap for the (multi-)selected method (assets × time)
- Inline 4-row mini-table (Sharpe, ann. return, MDD, vol) for the
  selected methods

### Tab 3 — Deep Method

- Model-card badge: `trained_on`, `trained_with`, `universe`,
  `final_metrics.test_annual_sharpe` from the JSON sidecar (or "checkpoint
  missing — run modal training" warning)
- Two columns:
  - Left: equity curve of Deep Portfolio overlaid with best-Sharpe
    classical method
  - Right: weight-over-time heatmap (one row per asset)
- "Train your own (tiny subsample)" expander surfacing the CLI command
  (Phase 1 precedent — no in-page training thread)

### Tab 4 — Table 1 / Figure 3 Replication

`st.tabs(["4A. Table 1 (Performance)", "4B. Figure 3 (Cumulative Returns)"])`

**Sub-tab 4A — Table 1**:
- `st.dataframe` with 3 rows × 9 cols:
  - Row 1: "No vol scaling, C = 0.01%"
  - Row 2: "Vol scaling, C = 0.01%"
  - Row 3: "Vol scaling, C = 0.10%"
  - Wait — paper's Table 1 has rows = methods, columns = scenarios.
    Our table has rows = (vol_scaling × cost_rate) panels, cols = methods.
    Actually NO — re-read the paper. Table 1 has rows = (vol_scaling ×
    cost) panels, cols = methods, and per-method cells contain a comma-
    separated `(E[R], Std(R), Sharpe, …)` 8-metric tuple. Streamlit can't
    render that easily.

  Renderer: render as one `st.dataframe` per panel (3 panels), with rows
  = methods and cols = 8 metrics. Best per column bolded via Streamlit's
  `column_config` formatting. Section headers `### Panel: no vol scaling,
  C = 0.01%` etc. separate the three.

**Sub-tab 4B — Figure 3**:
- Three Plotly cumulative-return charts side-by-side (`st.columns(3)`):
  - **Left**: no vol scaling, C = 0.01%
  - **Middle**: vol scaling (σ_target=10%), C = 0.01%
  - **Right**: vol scaling (σ_target=10%), C = 0.10%
- All in-scope methods overlaid in each panel, log scale
- Same `date_range` filter as Tab 2

## Math panel (`st.expander("📐 Math", expanded=False)`)

- Softmax constraint (long-only, sums to 1)
- Sharpe loss (same as Phase 1 Math panel)
- Volatility scaling formula
- One-sentence intuitions

## Code panel (`st.expander("💻 Code", expanded=False)`)

- `class DeepPortfolioMLP` via `inspect.getsource`
- `Neg_Sharpe` from `src/losses.py` via `inspect.getsource`

## Footer

- arXiv badge → 2005.13665
- GitHub deep-link to `pages/2_💼_Portfolio.py`
- BibTeX block

## Required widget keys (used by integration tests)

- `portfolio_universe`
- `portfolio_cost_rate`
- `portfolio_vol_scaling`
- `portfolio_lookback`
- `portfolio_date_range`

## Copy-text invariants (asserted by integration test)

- Substrate-disclosure substring in Tab 1: `"qualitative claim (Deep
  Portfolio in top 3 by Sharpe)"`.
- Fallback banner when ETF parquet absent: `"single-universe toy mode"`
  and `"scripts/fetch_data.py"`.

## Render assertions (6 parametrised cases per FR-022)

| Case | universe | parquet | Assertion |
|---|---|---|---|
| A | etfs    | absent  | Falls back to toy banner; no exception |
| B | etfs    | present | All 4 tabs render; Tab 4A has 3 dataframes (one per panel) |
| C | 20stock | absent  | Toy banner; no exception |
| D | 20stock | present | All 4 tabs render; Tab 4A dataframes have 5 rows (5 methods only — no alloc_*) |
| E | etfs    | present + checkpoint missing | Tab 3 shows "checkpoint missing" warning instead of model card |
| F | etfs    | present + checkpoint present | Tab 3 model card populates |
