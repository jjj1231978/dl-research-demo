"""Page 1 — Momentum (Lim, Zohren, Roberts 2019).

Spec: specs/002-phase-1-momentum/spec.md
Contract: specs/002-phase-1-momentum/contracts/momentum_page_ui.md

Replicates Exhibits 2/3/4/5 of the Lim et al. 2019 paper using the 18
BCOM commodity roots from the developer's databento lake (paper used
Pinnacle CLC multi-asset). Sharpe-loss only — Constitution Principle V
keeps `src/losses.py:SharpeLoss` immutable.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import data_root, render_data_status_sidebar
from src.data.futures import BCOM_ROOTS, TEST_START
from src.metrics import report_metrics
from src.models.deep_momentum import DeepMomentumLSTM, DeepMomentumMLP

# Repo-relative paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"

st.set_page_config(
    page_title="Momentum — Deep Finance Showcase",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────


def _backtests_dir() -> Path:
    raw = os.environ.get("DEEP_FINANCE_BACKTESTS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT / "data" / "backtests"


@st.cache_data(show_spinner=False)
def _load_cme_panel(parquet_path_str: str) -> pd.DataFrame:
    path = Path(parquet_path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@st.cache_data(show_spinner=False)
def _load_backtest_panel(parquet_path_str: str) -> pd.DataFrame:
    path = Path(parquet_path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@st.cache_data(show_spinner=False)
def _load_aapl_toy(csv_path_str: str) -> pd.DataFrame:
    """Load Phase 0 bundled AAPL CSV for toy-mode fallback."""
    path = Path(csv_path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ("date", "Date"):
        if c in df.columns:
            df["date"] = pd.to_datetime(df[c]).dt.normalize()
            break
    return df


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Momentum")
st.sidebar.markdown("**Lim, Zohren, Roberts (2019)** — sharpe-loss only.")

asset_set = st.sidebar.selectbox(
    "Asset set",
    options=("All futures", "Single asset"),
    key="momentum_asset_set",
)
# "Subset by class" reserved for a later phase per the contract.

if asset_set == "Single asset":
    single_asset = st.sidebar.selectbox(
        "Single asset", options=BCOM_ROOTS, index=0, key="momentum_single_asset",
    )
else:
    single_asset = None
    # Still register the key so tests can find it
    st.session_state.setdefault("momentum_single_asset", "CL")

vol_scaling = st.sidebar.checkbox(
    "Volatility scaling (σ_target = 15%)", value=True, key="momentum_vol_scaling",
)
ewma_span = st.sidebar.number_input(
    "EWMA span (vol lookback, days)", min_value=5, max_value=252, value=60, step=5,
    key="momentum_ewma_span",
)
# Date-range slider — bounds inferred from data when present; safe fallback otherwise
_today = _dt.date.today()
date_range = st.sidebar.slider(
    "Backtest range",
    min_value=_dt.date(2010, 6, 6),
    max_value=_today,
    value=(TEST_START, _today),
    key="momentum_date_range",
)
deep_model = st.sidebar.selectbox(
    "Deep model",
    options=("MLP", "LSTM", "Both"),
    index=0,
    key="momentum_deep_model",
)

st.sidebar.divider()
render_data_status_sidebar(st.sidebar)


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

st.title("📈 Momentum — Lim, Zohren, Roberts (2019)")
st.markdown(
    "*Enhancing Time Series Momentum Strategies Using Deep Neural Networks*, "
    "**Journal of Financial Data Science** — "
    "[arXiv:1904.04912](https://arxiv.org/abs/1904.04912)"
)
st.markdown(
    "> Instead of forecasting returns and then sizing positions, train a "
    "neural network end-to-end to directly output position sizes that "
    "maximize Sharpe ratio."
)

# ──────────────────────────────────────────────────────────────────────
# Data preflight
# ──────────────────────────────────────────────────────────────────────

cme_path = data_root() / "cme_futures.parquet"
panel = _load_cme_panel(str(cme_path))
parquet_present = not panel.empty

backtest_path = _backtests_dir() / "momentum_results.parquet"
backtest_panel_full = _load_backtest_panel(str(backtest_path))

# Honour the sidebar date_range slider. The paper's Exhibits 2/3/4/5 are all
# out-of-sample evaluation, so the default range starts at TEST_START — but the
# user can widen to include the training window if they want to inspect
# in-sample overfitting. Tab 2 and Tab 4 both consume this filtered view.
if not backtest_panel_full.empty:
    _lo = pd.Timestamp(date_range[0])
    _hi = pd.Timestamp(date_range[1])
    backtest_panel = backtest_panel_full[
        (backtest_panel_full["date"] >= _lo) & (backtest_panel_full["date"] <= _hi)
    ]
else:
    backtest_panel = backtest_panel_full


# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Problem & Data", "2. Reference Models",
     "3. Deep Method", "4. Exhibits 2/3/4/5"]
)


# ─── Tab 1: Problem & Data ────────────────────────────────────────────

with tab1:
    st.subheader("Why time-series momentum?")
    st.markdown(
        "Moskowitz, Ooi, and Pedersen (2012) documented that past-year "
        "returns predict future returns for ~ 60 liquid futures across "
        "asset classes — the **time-series momentum** (TSMOM) anomaly. "
        "Their classical implementation forecasts the sign of next-month "
        "return, then takes a vol-targeted position with that sign.\n\n"
        "Lim, Zohren, and Roberts (2019) collapse the two steps. A neural "
        "network reads the same lookback features (normalized return "
        "horizons + MACD timescales) and outputs a position in `(-1, +1)` "
        "directly via a Softsign head. The loss is **negative Sharpe** "
        "computed on the realised portfolio P&L — the network's gradients "
        "flow from the desired financial outcome straight through the "
        "position decision."
    )

    if not parquet_present:
        st.warning(
            "**No futures parquet found — page is running in single-asset toy mode.** "
            "Run `python scripts/fetch_futures.py` to populate "
            "`${DEEP_FINANCE_DATA_DIR}/cme_futures.parquet` (defaults to "
            "`./data/`). With no parquet, only the bundled AAPL CSV "
            "fallback is available; the Reference Models tab will overlay "
            "the three momentum signals on AAPL prices for illustration."
        )
        st.markdown("### Toy fallback: AAPL daily prices (bundled CSV)")
        from src.data import BUNDLED_CSV_DIR
        aapl_df = _load_aapl_toy(str(BUNDLED_CSV_DIR / "aapl.csv"))
        if not aapl_df.empty:
            price_col = next(
                (c for c in ("close", "Close", "price", "Price") if c in aapl_df.columns),
                None,
            )
            if price_col is not None:
                if "date" in aapl_df.columns:
                    st.line_chart(aapl_df.set_index("date")[price_col])
                else:
                    st.line_chart(aapl_df[price_col])
            else:
                st.dataframe(aapl_df.head(20), hide_index=True)
        else:
            st.info("AAPL bundled CSV not found either — please verify the repo state.")
    else:
        n_contracts = panel["contract"].nunique()
        d_min, d_max = panel["date"].min().date(), panel["date"].max().date()
        wide = panel.pivot(index="date", columns="contract", values="price").sort_index()
        ret_panel = wide.pct_change()
        corr = ret_panel.corr().values
        avg_corr = float(np.nanmean(corr[np.triu_indices_from(corr, k=1)]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Contracts", n_contracts)
        c2.metric("Date range", f"{d_min} → {d_max}")
        c3.metric("Avg inter-contract correlation", f"{avg_corr:.3f}")

        # Coverage Gantt
        cov_rows = []
        for c in sorted(panel["contract"].unique()):
            sub = panel[panel["contract"] == c]
            cov_rows.append(
                {"Contract": c, "Start": sub["date"].min(), "End": sub["date"].max()}
            )
        cov_df = pd.DataFrame(cov_rows)
        fig = px.timeline(cov_df, x_start="Start", x_end="End", y="Contract",
                          title="Coverage timeline (per BCOM root)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        if asset_set == "Single asset" and single_asset in wide.columns:
            sub = wide[single_asset].dropna()
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**{single_asset} — F0 ratio-adjusted price**")
                st.line_chart(sub)
            with colB:
                st.markdown(f"**{single_asset} — daily return distribution**")
                rets = sub.pct_change().dropna()
                hist = go.Figure(data=[go.Histogram(x=rets, nbinsx=80)])
                hist.update_layout(showlegend=False, height=320)
                st.plotly_chart(hist, width="stretch")

    st.divider()
    st.markdown(
        "### Substrate disclosure\n"
        "This page uses the 18 BCOM commodity roots from the developer's "
        "databento lake (some contracts may be absent or thinly populated). "
        "Lim et al. 2019 used Pinnacle CLC's multi-asset-class dataset "
        "(commodities + fixed income + equities + FX). The paper's "
        "qualitative-ordering claims hold on commodities-only; absolute "
        "Sharpe values differ from the paper."
    )


# ─── Tab 2: Reference Models ──────────────────────────────────────────

with tab2:
    st.subheader("Reference TSMOM signals")
    st.markdown(
        "Three signals from Lim et al. 2019 §4.1 baseline: **Long Only** "
        "(constant `+1`), **Sgn(Returns)** (past-year return sign, "
        "Moskowitz et al. 2012), and **MACD ensemble** (mean of three "
        "Baz 2015 vol-normalized MACDs at the 8/24, 16/48, 32/96 "
        "timescales)."
    )
    selected = st.multiselect(
        "Show signals",
        options=("Long Only", "Sgn(Returns)", "MACD"),
        default=("Long Only", "Sgn(Returns)", "MACD"),
        key="momentum_ref_signals",
    )
    if not backtest_panel.empty and selected:
        ref_map = {"Long Only": "long_only", "Sgn(Returns)": "sgn_returns",
                   "MACD": "macd"}
        cum = go.Figure()
        for label in selected:
            strat_key = ref_map[label]
            sub = backtest_panel[
                (backtest_panel["strategy"] == strat_key)
                & (backtest_panel["vol_scaling"] == vol_scaling)
            ]
            if sub.empty:
                continue
            daily = sub.groupby("date")["daily_return"].mean()
            equity = (1 + daily).cumprod()
            cum.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                      mode="lines", name=label))
        cum.update_layout(yaxis_type="log", height=420,
                          title=("Cumulative return (log) — "
                                  + ("σ_target=15%" if vol_scaling else "raw")))
        st.plotly_chart(cum, width="stretch")

        # Mini-table
        rows = []
        for label in selected:
            strat_key = ref_map[label]
            sub = backtest_panel[
                (backtest_panel["strategy"] == strat_key)
                & (backtest_panel["vol_scaling"] == vol_scaling)
            ]
            if sub.empty:
                continue
            daily = sub.groupby("date")["daily_return"].mean().to_numpy()
            if len(daily) > 1 and np.std(daily) > 0:
                m = report_metrics(daily)
                rows.append({"Strategy": label, "Sharpe": round(m["annual_sharpe"], 2),
                             "Annual return": f"{m['annual_ret']*100:.1f}%",
                             "MDD": f"{m['max_drawdown']*100:.1f}%"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info(
            "No backtest panel available yet. Run "
            "`python scripts/run_backtests.py --momentum` after "
            "`scripts/fetch_futures.py` populates the data parquet."
        )


# ─── Tab 3: Deep Method ───────────────────────────────────────────────

with tab3:
    st.subheader(f"Deep method — {deep_model}")

    def _load_sidecar(arch: str) -> dict | None:
        sidecar_path = _PRETRAINED_DIR / f"{arch.lower()}_sharpe.json"
        if not sidecar_path.exists():
            return None
        try:
            return json.loads(sidecar_path.read_text())
        except Exception:  # noqa: BLE001
            return None

    archs_to_show = (
        ["MLP", "LSTM"] if deep_model == "Both" else [deep_model]
    )

    for arch in archs_to_show:
        sidecar = _load_sidecar(arch)
        if sidecar:
            badges = (
                f"**{arch}** — trained `{sidecar.get('trained_on', '?')}` on "
                f"`{sidecar.get('trained_with', '?')}` — test Sharpe "
                f"{sidecar.get('final_metrics', {}).get('test_annual_sharpe', float('nan')):.2f}"
            )
            st.info(badges)
        else:
            st.warning(
                f"No checkpoint sidecar found for **{arch}** at "
                f"`data/pretrained/{arch.lower()}_sharpe.json`. "
                f"Run `modal run src/training/train_deep_momentum.py --arch {arch}`."
            )

    if not backtest_panel.empty:
        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Equity curve (test window)**")
            eq = go.Figure()
            for arch in archs_to_show:
                strat = f"{arch.lower()}_sharpe"
                sub = backtest_panel[
                    (backtest_panel["strategy"] == strat)
                    & (backtest_panel["vol_scaling"] == vol_scaling)
                ]
                if sub.empty:
                    continue
                daily = sub.groupby("date")["daily_return"].mean()
                equity = (1 + daily).cumprod()
                eq.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                         mode="lines", name=arch))
            eq.update_layout(yaxis_type="log", height=360, title="Equity (log)")
            st.plotly_chart(eq, width="stretch")
        with colR:
            st.markdown("**Position over time** (one contract)")
            # Reuse the sidebar's single_asset if set; else pick the most-traded
            ct = single_asset if single_asset else (
                backtest_panel["contract"].value_counts().index[0]
            )
            arch_for_pos = archs_to_show[0]
            strat = f"{arch_for_pos.lower()}_sharpe"
            sub = backtest_panel[
                (backtest_panel["strategy"] == strat)
                & (backtest_panel["vol_scaling"] == vol_scaling)
                & (backtest_panel["contract"] == ct)
            ]
            if not sub.empty:
                pos_fig = px.line(sub, x="date", y="daily_return",
                                   title=f"{arch_for_pos} P&L on {ct}")
                pos_fig.update_layout(height=360)
                st.plotly_chart(pos_fig, width="stretch")
            else:
                st.info(f"No backtest rows for {arch_for_pos} on {ct}.")
    else:
        st.info("Backtest panel not yet generated; run `scripts/run_backtests.py --momentum`.")

    with st.expander("🧪 Train your own (tiny subsample) — ≤ 30 s", expanded=False):
        st.markdown(
            "Runs a 1-epoch CPU smoke training on a 1-contract × 1-year "
            "subset of the futures panel (Constitution Principle III — "
            "production inference only; this button is illustrative)."
        )
        if not parquet_present:
            st.warning("Need a fetched parquet to run this. See sidebar.")
        elif st.button("Run smoke training", key="momentum_train_button"):
            with st.spinner("Smoke training on a single contract..."):
                st.info(
                    "Smoke training not wired in Phase 1 page; use the CLI: "
                    "`python -m src.training.train_deep_momentum --arch "
                    f"{archs_to_show[0]} --max-epochs 1 -v`"
                )


# ─── Tab 4: Exhibits 2/3/4/5 ──────────────────────────────────────────

with tab4:
    st.subheader("Exhibits 2 / 3 / 4 / 5 replication")
    st.caption("`(reproduced here)` from Lim, Zohren, Roberts 2019, with the "
               "BCOM commodities-only substrate disclosed in Tab 1.")

    if backtest_panel.empty:
        st.info("Run `scripts/run_backtests.py --momentum` to populate the "
                "exhibits.")
    else:
        sub4a, sub4b, sub4c, sub4d = st.tabs(
            ["4A. Exhibit 2 (Raw)", "4B. Exhibit 3 (Rescaled)",
             "4C. Exhibit 4 (Cum. Returns)", "4D. Exhibit 5 (Per-Asset)"]
        )

        # Map paper-column-name → report_metrics key (research.md R8)
        PAPER_COLS = [
            ("E[Return]", "annual_ret"),
            ("Vol", "annual_std"),
            ("Downside Deviation", "downside_dev"),
            ("MDD", "max_drawdown"),
            ("Sharpe", "annual_sharpe"),
            ("Sortino", "sortino"),
            ("Calmar", "calmar"),
            ("% +ve Returns", "pct_positive"),
            ("Ave. P / Ave. L", "avg_p_over_l"),
        ]
        STRAT_ORDER = [
            ("Long Only", "long_only"),
            ("Sgn(Returns)", "sgn_returns"),
            ("MACD", "macd"),
            ("MLP-Sharpe", "mlp_sharpe"),
            ("LSTM-Sharpe", "lstm_sharpe"),
        ]

        def _metric_table(vol_flag: bool) -> pd.DataFrame:
            rows = []
            for label, key in STRAT_ORDER:
                sub = backtest_panel[
                    (backtest_panel["strategy"] == key)
                    & (backtest_panel["vol_scaling"] == vol_flag)
                ]
                if sub.empty:
                    rows.append({"Strategy": label, **{c: np.nan for c, _ in PAPER_COLS}})
                    continue
                daily = sub.groupby("date")["daily_return"].mean().to_numpy()
                if len(daily) < 2 or np.std(daily) == 0:
                    rows.append({"Strategy": label, **{c: np.nan for c, _ in PAPER_COLS}})
                    continue
                m = report_metrics(daily)
                rows.append({"Strategy": label,
                             **{paper_col: m[m_key] for paper_col, m_key in PAPER_COLS}})
            return pd.DataFrame(rows)

        with sub4a:
            st.markdown("**Exhibit 2** — raw signal outputs (no σ_target rescaling)")
            df4a = _metric_table(vol_flag=False)
            st.dataframe(df4a, hide_index=True, width="stretch")

        with sub4b:
            st.markdown("**Exhibit 3** — all strategies rescaled to σ_target = 15 %")
            df4b = _metric_table(vol_flag=True)
            st.dataframe(df4b, hide_index=True, width="stretch")

        with sub4c:
            st.markdown("**Exhibit 4 (panel a)** — cumulative returns, σ_target rescaled")
            fig4c = go.Figure()
            for label, key in STRAT_ORDER:
                sub = backtest_panel[
                    (backtest_panel["strategy"] == key)
                    & (backtest_panel["vol_scaling"] == True)  # noqa: E712
                ]
                if sub.empty:
                    continue
                daily = sub.groupby("date")["daily_return"].mean()
                equity = (1 + daily).cumprod()
                fig4c.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                             mode="lines", name=label))
            fig4c.update_layout(yaxis_type="log", height=460,
                                  title="Cumulative return (log scale)")
            st.plotly_chart(fig4c, width="stretch")

        with sub4d:
            st.markdown("**Exhibit 5** — per-asset distributions across the 5 strategies")
            # Compute per-asset metrics on the rescaled panel
            per_asset_rows = []
            for label, key in STRAT_ORDER:
                sub = backtest_panel[
                    (backtest_panel["strategy"] == key)
                    & (backtest_panel["vol_scaling"] == True)  # noqa: E712
                ]
                for ct in sub["contract"].unique():
                    daily = sub[sub["contract"] == ct]["daily_return"].to_numpy()
                    if len(daily) < 2 or np.std(daily) == 0:
                        continue
                    m = report_metrics(daily)
                    per_asset_rows.append({
                        "Strategy": label, "Contract": ct,
                        "Sharpe": m["annual_sharpe"],
                        "Average return": m["annual_ret"],
                        "Volatility": m["annual_std"],
                    })
            if per_asset_rows:
                per_asset = pd.DataFrame(per_asset_rows)
                c1, c2, c3 = st.columns(3)
                for col, metric in zip(
                    (c1, c2, c3), ("Sharpe", "Average return", "Volatility")
                ):
                    fig = px.box(per_asset, x="Strategy", y=metric, points=False,
                                 title=metric)
                    fig.update_layout(height=340, showlegend=False)
                    col.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────────
# Math + Code expanders
# ──────────────────────────────────────────────────────────────────────

with st.expander("📐 Math", expanded=False):
    st.markdown("**Sharpe-loss objective** (Eq. 4 of paper)")
    st.latex(r"\mathcal{L}_{\mathrm{Sharpe}} = - \frac{\mathbb{E}_t[R_t]}{\sigma_t[R_t]}, \quad R_t = X_t \cdot r_{t+1}")
    st.caption("Negate so SGD minimises → maximises Sharpe. `X_t ∈ (-1, +1)` is the model's Softsign output.")

    st.markdown("**Volatility scaling** (Eq. 1 of paper)")
    st.latex(r"X_t^{\mathrm{scaled}} = X_t \cdot \frac{\sigma_{\mathrm{target}}}{\hat{\sigma}_t}, \quad \hat{\sigma}_t = \mathrm{EWMA}(r_{t}, \mathrm{span} = 60) \cdot \sqrt{252}")
    st.caption(r"With $\sigma_\mathrm{target} = 0.15$, signal-vol mismatches across assets and time collapse to a comparable scale.")

    st.markdown("**Softsign output**")
    st.latex(r"\mathrm{Softsign}(z) = \frac{z}{1 + |z|}")
    st.caption("Smoother saturation than tanh — gradients survive deeper into training under the Sharpe loss.")


with st.expander("💻 Code", expanded=False):
    st.markdown("**`src.models.deep_momentum.DeepMomentumMLP`**")
    st.code(inspect.getsource(DeepMomentumMLP), language="python")
    st.markdown("**`src.models.deep_momentum.DeepMomentumLSTM`**")
    st.code(inspect.getsource(DeepMomentumLSTM), language="python")

    st.markdown("**`src.losses.SharpeLoss`**")
    from src.losses import SharpeLoss
    st.code(inspect.getsource(SharpeLoss), language="python")


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "📄 [arXiv:1904.04912](https://arxiv.org/abs/1904.04912) · "
    "💻 [pages/1_📈_Momentum.py on GitHub]"
    "(https://github.com/jjj1231978/dl-research-demo/blob/main/pages/1_%F0%9F%93%88_Momentum.py) · "
    "Constitution v1.1.0."
)
st.code(
    """@article{lim2019enhancing,
  title={Enhancing Time Series Momentum Strategies Using Deep Neural Networks},
  author={Lim, Bryan and Zohren, Stefan and Roberts, Stephen},
  journal={Journal of Financial Data Science},
  year={2019}
}""",
    language="bibtex",
)
