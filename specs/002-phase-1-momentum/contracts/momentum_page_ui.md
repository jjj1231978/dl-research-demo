# Contract: `pages/1_📈_Momentum.py`

**Type**: Streamlit page, replaces the Phase 0 placeholder.
**Renders on**: HF Space CPU Basic + local dev.
**Consumed by**: `tests/integration/test_momentum_page.py` (AppTest
assertions).

## Page header

- Title: `📈 Momentum — Lim, Zohren, Roberts (2019)`
- Subtitle: *"Enhancing Time Series Momentum Strategies Using Deep
  Neural Networks"*, Journal of Financial Data Science.
- arXiv link: <https://arxiv.org/abs/1904.04912>
- Elevator pitch (≤ 2 lines): "Instead of forecasting returns then sizing
  positions, train a neural network end-to-end to directly output
  position sizes that maximize Sharpe ratio."

## Sidebar (left rail)

In order, top to bottom:

| Widget | `key` | Default | Notes |
|---|---|---|---|
| `st.selectbox` "Asset set" | `momentum_asset_set` | `"All futures"` | Options: "All futures", "Subset by class", "Single asset". Phase 1: only "All futures" and "Single asset" wired; "Subset by class" reserved for the asset_class expansion in a later phase. |
| `st.selectbox` "Single asset" (visible only when asset_set = "Single asset") | `momentum_single_asset` | `"CL"` | Dropdown over 18 BCOM roots. |
| `st.checkbox` "Volatility scaling (σ_target = 15%)" | `momentum_vol_scaling` | `True` | Toggles between Tab 4A (off) and Tab 4B (on) views as the "active" condition; both sub-tabs stay populated regardless. |
| `st.number_input` "EWMA span (vol lookback, days)" | `momentum_ewma_span` | `60` | Used by `src/strategies/vol_targeting.py`. |
| `st.slider` "Backtest range" | `momentum_date_range` | `(2020-01-01, max_date)` | Test-window default per OD-3. |
| `st.selectbox` "Deep model" | `momentum_deep_model` | `"MLP"` | Options: "MLP", "LSTM", "Both overlaid". |
| `st.divider` |   |   |   |
| `render_data_status_sidebar(...)` from Phase 0 | (same as landing) | — | Reuses the Phase 0 data-status component. |

## Main area: four tabs

`st.tabs(["1. Problem & Data", "2. Reference Models", "3. Deep Method",
"4. Exhibits 2/3/4/5"])`

### Tab 1 — Problem & Data

- One paragraph: TSMOM problem (Moskowitz et al. 2012) + Lim et al. 2019
  contribution.
- Universe summary (when asset_set = "All futures"):
  - `st.metric` row showing: # of contracts (18), date range, avg
    inter-contract correlation.
  - Plotly Gantt-style coverage timeline (one row per contract).
- Single-asset drill-down (when asset_set = "Single asset"):
  - Plotly price chart (log scale).
  - Plotly return distribution (histogram + KDE).
- Substrate disclosure block: "This page uses the 18 BCOM commodity roots
  from the developer's databento lake. Lim et al. 2019 used Pinnacle
  CLC's multi-asset-class dataset (commodities + fixed income + equities
  + FX). Qualitative-ordering claims hold on commodities-only; absolute
  Sharpe values differ from the paper."

### Tab 2 — Reference Models

- Multi-select toggle for which reference models to display (default:
  all 3 — Long Only, Sgn(Returns), MACD).
- Plotly cumulative-return chart (log scale), one trace per selected
  model. Vol-scaling toggle from the sidebar controls which series is
  shown.
- Inline metrics summary (3-row mini-table; full Exhibit 2/3 tables are
  in Tab 4).

### Tab 3 — Deep Method

- Two columns:
  - **Left**: equity curve for the selected `momentum_deep_model`
    architecture. When "Both overlaid", two traces with a legend.
  - **Right**: position-over-time plot for the contract selected in the
    sidebar (or the most-traded contract if asset_set = "All futures").
- Model-card badge above the columns: read `mlp_sharpe.json` or
  `lstm_sharpe.json` (per selected model) and display
  `trained_on`, `trained_with`, `arch`, `final_metrics.test_annual_sharpe`.
- "Train your own (tiny subsample)" expander (collapsed by default):
  contains a button that runs `train()` on a single-contract × 1-year
  subset for ≤ 30 s wall-clock (Constitution Principle III; FR-018) and
  shows the live loss curve via `st.line_chart` updated each epoch.
  Hard-abort thread at 30 s.

### Tab 4 — Exhibits 2 / 3 / 4 / 5 Replication

`st.tabs(["4A. Exhibit 2 (Raw)", "4B. Exhibit 3 (Rescaled)",
         "4C. Exhibit 4 (Cum. Returns)", "4D. Exhibit 5 (Per-Asset)"])`

**Sub-tab 4A — Exhibit 2 (Raw Signal Outputs)**:
- `st.dataframe` with 5 rows (Long Only, Sgn(Returns), MACD, MLP-Sharpe,
  LSTM-Sharpe) and the 9 columns mapped from `src/metrics.py:report_metrics`
  per `research.md` R8.
- Computed from rows in `momentum_results.parquet` where
  `vol_scaling == False`.
- Best-per-column bolded via `column_config` formatting.
- `(reproduced here)` badge above the table (Constitution Principle I).

**Sub-tab 4B — Exhibit 3 (Rescaled to σ_target = 15%)**:
- Same table shape as 4A, computed from rows where `vol_scaling == True`.
- Same badging.

**Sub-tab 4C — Exhibit 4 (Cumulative Returns)**:
- Single Plotly chart on log scale, all 5 strategies overlaid, rescaled
  to σ_target. Corresponds to panel (a) of paper Exhibit 4.

**Sub-tab 4D — Exhibit 5 (Per-Asset Performance)**:
- Three Plotly box plots side-by-side:
  - Sharpe per asset, x-axis = strategy (5 boxes).
  - Average return per asset, x-axis = strategy.
  - Volatility per asset, x-axis = strategy.
- Source data: per-asset rows in `momentum_results.parquet`.

## Math panel (`st.expander("📐 Math", expanded=False)`)

- Sharpe-loss formula (Eq. 4 of paper) via `st.latex`.
- Volatility-scaling formula (Eq. 1 of paper) via `st.latex`.
- Softsign output mapping to (-1, +1).
- One-sentence intuitions below each equation.

## Code panel (`st.expander("💻 Code", expanded=False)`)

- `st.code` block showing `class DeepMomentumMLP` and
  `class DeepMomentumLSTM` from `src/models/deep_momentum.py` (paste
  verbatim at render time via `inspect.getsource` so the panel cannot
  drift from the code).
- `st.code` block showing `SharpeLoss` from `src/losses.py`.

## Footer

- arXiv badge linking to the PDF.
- GitHub deep-link to `pages/1_📈_Momentum.py` source.
- BibTeX copy block (Plotly's clipboard icon).

## Render assertions (integration test)

`tests/integration/test_momentum_page.py` runs **six parameterised
cases** (per FR-022):

| Case | parquet | Deep model | Assertion |
|---|---|---|---|
| A | absent | MLP | Banner "single-asset toy mode" visible; AppTest.run() no exception. |
| B | absent | LSTM | Same as A; Deep Method tab handles missing checkpoint gracefully. |
| C | absent | Both | Same as A; selector renders both options even in toy mode. |
| D | present | MLP | All four tab labels appear; Tab 4A renders 5-row table; no exception. |
| E | present | LSTM | Same as D; Tab 3 model card shows LSTM checkpoint metadata. |
| F | present | Both | Same as D; Tab 3 shows two equity curves overlaid. |

All six cases pass when `AppTest.run()` raises no exception and the
specific assertion above succeeds.

## Required widget keys (used by integration tests)

- `momentum_asset_set`
- `momentum_single_asset`
- `momentum_vol_scaling`
- `momentum_ewma_span`
- `momentum_date_range`
- `momentum_deep_model`

Tests grep the rendered widget list for these `key` values to confirm
the sidebar is wired correctly.

## Copy-text invariants

The substrate disclosure block in Tab 1 MUST literally contain the
substring `"BCOM commodity roots"` AND `"Pinnacle CLC"` AND
`"qualitative-ordering claims hold on commodities-only"`. Verified by the
integration test.

The CSV-fallback / parquet-absent banner MUST literally contain
`"single-asset toy mode"` and `"scripts/fetch_futures.py"`.
