"""Page 2 — Portfolio (Zhang, Zohren, Roberts 2020).

Spec: specs/003-phase-2-portfolio/spec.md
Contract: specs/003-phase-2-portfolio/contracts/portfolio_page_ui.md

Replicates Table 1 + Figure 3 of the paper on two universes (ETF basket
+ sp500_20). Sharpe-loss only per Constitution Principle V.
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
from src.metrics import report_metrics
from src.models.deep_portfolio import DeepPortfolioMLP

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRETRAINED_DIR = _REPO_ROOT / "data" / "pretrained"

_UNIVERSE_KEYS = {"ETF basket": "etfs", "20-stock": "20stock"}
_PARQUET_MAP = {"etfs": "etf_basket", "20stock": "sp500_20"}

_COST_RATES = {"0.01%": 0.0001, "0.10%": 0.0010}
_PANELS = (
    {"vol_scaling": False, "cost_rate": 0.0001, "label": "No vol scaling, C = 0.01%"},
    {"vol_scaling": True,  "cost_rate": 0.0001, "label": "Vol scaling (σ=10%), C = 0.01%"},
    {"vol_scaling": True,  "cost_rate": 0.0010, "label": "Vol scaling (σ=10%), C = 0.10%"},
)

_METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Min Variance",
    "max_diversification": "Max Diversification",
    "diversity_weighted": "Diversity Weighted",
    "alloc_25_25_25_25": "Alloc 25/25/25/25",
    "alloc_50_10_20_20": "Alloc 50/10/20/20",
    "alloc_10_50_20_20": "Alloc 10/50/20/20",
    "alloc_40_40_10_10": "Alloc 40/40/10/10",
    "deep_portfolio": "Deep Portfolio",
}

# 8 metric columns mapped to src.metrics.report_metrics (research.md R8)
_PAPER_COLS = [
    ("E[Return]", "annual_ret"),
    ("Vol", "annual_std"),
    ("Downside Deviation", "downside_dev"),
    ("MDD", "max_drawdown"),
    ("Sharpe", "annual_sharpe"),
    ("Sortino", "sortino"),
    ("% +ve Returns", "pct_positive"),
    ("Ave. P / Ave. L", "avg_p_over_l"),
]

st.set_page_config(
    page_title="Portfolio — Deep Finance Showcase",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _backtests_dir() -> Path:
    raw = os.environ.get("DEEP_FINANCE_BACKTESTS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT / "data" / "backtests"


@st.cache_data(show_spinner=False)
def _load_universe_parquet(parquet_path_str: str) -> pd.DataFrame:
    p = Path(parquet_path_str)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@st.cache_data(show_spinner=False)
def _load_backtest_panel(parquet_path_str: str) -> pd.DataFrame:
    p = Path(parquet_path_str)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────

st.sidebar.title("💼 Portfolio")
st.sidebar.markdown("**Zhang, Zohren, Roberts (2020)** — Sharpe-loss only.")

universe_display = st.sidebar.selectbox(
    "Universe",
    options=list(_UNIVERSE_KEYS),
    key="portfolio_universe",
)
universe = _UNIVERSE_KEYS[universe_display]

cost_label = st.sidebar.selectbox(
    "Transaction cost rate",
    options=list(_COST_RATES),
    key="portfolio_cost_rate",
)
selected_cost = _COST_RATES[cost_label]

vol_scaling = st.sidebar.checkbox(
    "Volatility scaling (σ_target = 10%)",
    value=True,
    key="portfolio_vol_scaling",
)

lookback = st.sidebar.slider(
    "Rolling window (classical, days)",
    min_value=20, max_value=252, value=50, step=5,
    key="portfolio_lookback",
)

_today = _dt.date.today()
date_range = st.sidebar.slider(
    "Backtest range (Tab 2 / Tab 4)",
    min_value=_dt.date(2006, 6, 30),
    max_value=_today,
    value=(_dt.date(2020, 1, 1), _today),
    key="portfolio_date_range",
)

st.sidebar.divider()
render_data_status_sidebar(st.sidebar)


# ── Header ────────────────────────────────────────────────────────────

st.title("💼 Portfolio — Zhang, Zohren, Roberts (2020)")
st.markdown(
    "*Deep Learning for Portfolio Optimization* — "
    "[arXiv:2005.13665](https://arxiv.org/abs/2005.13665)"
)
st.markdown(
    "> A long-only portfolio with **softmax** output, trained by gradient "
    "ascent to maximize Sharpe directly — no covariance matrix, no "
    "expected-return forecast."
)

# ── Data preflight ───────────────────────────────────────────────────

parquet_path = data_root() / f"{_PARQUET_MAP[universe]}.parquet"
panel_raw = _load_universe_parquet(str(parquet_path))
parquet_present = not panel_raw.empty

backtest_full = _load_backtest_panel(str(_backtests_dir() / "portfolio_results.parquet"))
if not backtest_full.empty:
    _lo = pd.Timestamp(date_range[0])
    _hi = pd.Timestamp(date_range[1])
    backtest_panel = backtest_full[
        (backtest_full["universe"] == universe)
        & (backtest_full["date"] >= _lo)
        & (backtest_full["date"] <= _hi)
    ]
else:
    backtest_panel = backtest_full


# ── Tabs ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Problem & Data", "2. Classical Benchmarks",
     "3. Deep Method", "4. Table 1 / Figure 3"]
)


with tab1:
    st.subheader("Why portfolio optimization?")
    st.markdown(
        "Markowitz (1952) framed portfolio choice as a quadratic program: "
        "given expected returns μ and covariance Σ, find weights that "
        "maximize μ'w − λ(w'Σw). The classical two-step pipeline forecasts "
        "μ and Σ first, then solves the QP.\n\n"
        "Zhang, Zohren, Roberts (2020) collapse the two steps. A neural "
        "network reads the lookback window of prices and returns and "
        "outputs a long-only weight vector directly via a **Softmax** "
        "head. The loss is **negative Sharpe** computed on the realized "
        "portfolio P&L — gradients flow from the financial outcome "
        "straight to allocation decisions."
    )

    if not parquet_present:
        st.warning(
            f"**No {universe_display} parquet found — page is running in "
            "single-universe toy mode.** Run "
            "`python scripts/fetch_data.py --universe etf_basket --universe sp500_20` "
            "to populate `${DEEP_FINANCE_DATA_DIR}/`. With no parquet the "
            "classical and deep methods cannot run; Tab 2 and Tab 4 will be empty."
        )
    else:
        close_wide = panel_raw.pivot(index="date", columns="symbol", values="close").sort_index()
        close_wide = close_wide.ffill().dropna(how="all")
        rets_wide = close_wide.pct_change(1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Assets", len(close_wide.columns))
        c2.metric("Date range",
                   f"{close_wide.index.min().date()} → {close_wide.index.max().date()}")
        corr_mat = rets_wide.corr().values
        avg_corr = float(np.nanmean(corr_mat[np.triu_indices_from(corr_mat, k=1)]))
        c3.metric("Avg pairwise corr", f"{avg_corr:.3f}")

        st.markdown("**Pairwise correlation of daily returns**")
        heat = rets_wide.corr()
        fig_corr = px.imshow(
            heat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, width="stretch")

    st.divider()
    st.markdown(
        "### Substrate disclosure\n"
        "**ETF basket** replicates the paper's exact universe (VTI / AGG / "
        "DBC / VIXY). **20-stock** substitutes a sector-spread S&P 500 "
        "subset (the 20 names from `src/universes.py`). The paper's "
        "qualitative claim (Deep Portfolio in top 3 by Sharpe) is asserted "
        "on these substrates; absolute Sharpe values differ from the paper "
        "because our test window is 2020+ vs the paper's pre-2020. "
        "Diversity-Weighted Portfolio is skipped when FMP's "
        "`shares-outstanding` endpoint returns sparse data (typical for "
        "ETFs)."
    )


with tab2:
    st.subheader("Classical benchmarks")
    if backtest_panel.empty:
        st.info(
            "No backtest panel available. Run "
            "`python scripts/run_backtests.py --portfolio` after "
            "`scripts/fetch_data.py` populates the universe parquet."
        )
    else:
        available_methods = sorted(backtest_panel["method"].unique())
        classical_methods = [m for m in available_methods if m != "deep_portfolio"]
        selected = st.multiselect(
            "Show methods",
            options=classical_methods,
            default=classical_methods,
            format_func=lambda m: _METHOD_LABELS.get(m, m),
            key="portfolio_methods_select",
        )
        active_panel = backtest_panel[
            (backtest_panel["vol_scaling"] == vol_scaling)
            & (backtest_panel["cost_rate"] == selected_cost)
        ]
        cum = go.Figure()
        for m in selected:
            sub = active_panel[active_panel["method"] == m]
            if sub.empty:
                continue
            daily = sub.set_index("date")["portfolio_return"]
            equity = (1 + daily).cumprod()
            cum.add_trace(go.Scatter(
                x=equity.index, y=equity.values, mode="lines",
                name=_METHOD_LABELS.get(m, m),
            ))
        cum.update_layout(
            yaxis_type="log", height=420,
            title=f"Cumulative return (log) — "
                  f"{'σ=10%' if vol_scaling else 'raw'}, C={cost_label}",
        )
        st.plotly_chart(cum, width="stretch")

        rows = []
        for m in selected:
            sub = active_panel[active_panel["method"] == m]
            if sub.empty:
                continue
            r = sub["portfolio_return"].to_numpy()
            if len(r) > 1 and np.std(r) > 0:
                mt = report_metrics(r)
                rows.append({
                    "Method": _METHOD_LABELS.get(m, m),
                    "Sharpe": round(mt["annual_sharpe"], 2),
                    "Ann. return": f"{mt['annual_ret'] * 100:.1f}%",
                    "Ann. vol": f"{mt['annual_std'] * 100:.1f}%",
                    "MDD": f"{mt['max_drawdown'] * 100:.1f}%",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


with tab3:
    st.subheader(f"Deep Portfolio — {universe_display}")

    sidecar_path = _PRETRAINED_DIR / f"deep_portfolio_{universe}.json"
    sidecar_data = None
    if sidecar_path.exists():
        try:
            sidecar_data = json.loads(sidecar_path.read_text())
        except Exception:  # noqa: BLE001
            sidecar_data = None

    if sidecar_data:
        sharpe = sidecar_data.get("final_metrics", {}).get(
            "test_annual_sharpe", float("nan")
        )
        epochs = sidecar_data.get("hyperparameters", {}).get("epochs_trained", "?")
        st.info(
            f"**Deep Portfolio ({universe})** — trained "
            f"`{sidecar_data.get('trained_on', '?')}` on "
            f"`{sidecar_data.get('trained_with', '?')}` — "
            f"test Sharpe **{sharpe:.2f}** — epochs {epochs}"
        )
    else:
        st.warning(
            f"No checkpoint sidecar found at `data/pretrained/deep_portfolio_{universe}.json`. "
            f"Run `modal run src/training/train_deep_portfolio.py --universe {universe}`."
        )

    if not backtest_panel.empty:
        deep_rows = backtest_panel[
            (backtest_panel["method"] == "deep_portfolio")
            & (backtest_panel["vol_scaling"] == vol_scaling)
            & (backtest_panel["cost_rate"] == selected_cost)
        ]
        if deep_rows.empty:
            st.info(
                "Deep Portfolio rows not yet in the backtest panel. "
                "Re-run `scripts/run_backtests.py --portfolio` after the "
                "checkpoint lands in `data/pretrained/`."
            )
        else:
            colL, colR = st.columns(2)
            with colL:
                st.markdown("**Equity vs. best classical**")
                active = backtest_panel[
                    (backtest_panel["vol_scaling"] == vol_scaling)
                    & (backtest_panel["cost_rate"] == selected_cost)
                ]
                best_method = None
                best_sharpe = -np.inf
                for m in active["method"].unique():
                    if m == "deep_portfolio":
                        continue
                    r = active[active["method"] == m]["portfolio_return"].to_numpy()
                    if len(r) > 1 and np.std(r) > 0:
                        s = float(np.mean(r) / np.std(r) * np.sqrt(252))
                        if s > best_sharpe:
                            best_sharpe = s
                            best_method = m
                fig_eq = go.Figure()
                deep_daily = deep_rows.set_index("date")["portfolio_return"]
                fig_eq.add_trace(go.Scatter(
                    x=deep_daily.index, y=(1 + deep_daily).cumprod().values,
                    mode="lines", name="Deep Portfolio",
                ))
                if best_method:
                    bm = active[active["method"] == best_method]
                    bm_daily = bm.set_index("date")["portfolio_return"]
                    fig_eq.add_trace(go.Scatter(
                        x=bm_daily.index, y=(1 + bm_daily).cumprod().values,
                        mode="lines",
                        name=f"Best classical: {_METHOD_LABELS.get(best_method, best_method)}",
                    ))
                fig_eq.update_layout(yaxis_type="log", height=400,
                                       title="Equity (log)")
                st.plotly_chart(fig_eq, width="stretch")
            with colR:
                st.markdown("**Per-day Deep Portfolio P&L**")
                fig_pnl = px.line(deep_rows, x="date", y="portfolio_return",
                                    title="Daily portfolio return")
                fig_pnl.update_layout(height=400)
                st.plotly_chart(fig_pnl, width="stretch")

    with st.expander("🧪 Train your own (tiny subsample) — ≤ 30 s", expanded=False):
        st.markdown(
            "Smoke training not wired in the page (Constitution Principle "
            "III). Use the CLI instead:\n\n"
            f"```bash\npython -m src.training.train_deep_portfolio "
            f"--universe {universe} --max-epochs 1 -v\n```"
        )


with tab4:
    st.subheader("Table 1 / Figure 3 replication")
    st.caption("`(reproduced here)` from Zhang, Zohren, Roberts 2020.")

    if backtest_panel.empty:
        st.info("Run `scripts/run_backtests.py --portfolio` to populate.")
    else:
        sub4a, sub4b = st.tabs(["4A. Table 1 (Performance)",
                                   "4B. Figure 3 (Cumulative Returns)"])

        def _panel_metric_table(p):
            rows = []
            for m in sorted(backtest_panel["method"].unique()):
                sub = backtest_panel[
                    (backtest_panel["method"] == m)
                    & (backtest_panel["vol_scaling"] == p["vol_scaling"])
                    & (backtest_panel["cost_rate"] == p["cost_rate"])
                ]
                if sub.empty:
                    continue
                r = sub["portfolio_return"].to_numpy()
                if len(r) < 2 or np.std(r) == 0:
                    continue
                mt = report_metrics(r)
                rows.append({"Method": _METHOD_LABELS.get(m, m),
                              **{col: mt[key] for col, key in _PAPER_COLS}})
            return pd.DataFrame(rows)

        with sub4a:
            for p in _PANELS:
                st.markdown(f"### Panel: {p['label']}")
                df = _panel_metric_table(p)
                if df.empty:
                    st.info("(no rows for this panel)")
                else:
                    st.dataframe(df, hide_index=True, width="stretch")

        with sub4b:
            cols = st.columns(3)
            for col, p in zip(cols, _PANELS):
                with col:
                    st.markdown(f"**{p['label']}**")
                    fig = go.Figure()
                    for m in sorted(backtest_panel["method"].unique()):
                        sub = backtest_panel[
                            (backtest_panel["method"] == m)
                            & (backtest_panel["vol_scaling"] == p["vol_scaling"])
                            & (backtest_panel["cost_rate"] == p["cost_rate"])
                        ]
                        if sub.empty:
                            continue
                        daily = sub.set_index("date")["portfolio_return"]
                        eq = (1 + daily).cumprod()
                        fig.add_trace(go.Scatter(
                            x=eq.index, y=eq.values, mode="lines",
                            name=_METHOD_LABELS.get(m, m),
                        ))
                    fig.update_layout(yaxis_type="log", height=420,
                                       showlegend=True)
                    st.plotly_chart(fig, width="stretch")


with st.expander("📐 Math", expanded=False):
    st.markdown("**Softmax output (long-only simplex)**")
    st.latex(r"w_i = \frac{e^{\tilde w_i}}{\sum_j e^{\tilde w_j}}, \quad w_i \in [0,1], \; \sum_i w_i = 1")
    st.caption("Long-only allocation by construction; no projection step at inference.")

    st.markdown("**Sharpe-loss objective** (paper Eq. 4 + Phase 1's per-day aggregation)")
    st.latex(r"\mathcal{L} = -\frac{\mathbb{E}_t[R_t]}{\sigma_t[R_t]}, \quad R_t = \sum_i w_{i,t} \cdot r_{i,t+1}")
    st.caption("Per-day portfolio P&L via Σ across assets, then Sharpe over the time series.")

    st.markdown("**Volatility scaling** (paper §4.4)")
    st.latex(r"R_t^{\mathrm{scaled}} = R_t \cdot \frac{\sigma_{\mathrm{target}}}{\hat \sigma_t}")
    st.caption(r"$\sigma_\mathrm{target} = 0.10$ per Table 1.")


with st.expander("💻 Code", expanded=False):
    st.markdown("**`src.models.deep_portfolio.DeepPortfolioMLP`**")
    st.code(inspect.getsource(DeepPortfolioMLP), language="python")
    st.markdown("**`src.losses.Neg_Sharpe`**")
    from src.losses import Neg_Sharpe
    st.code(inspect.getsource(Neg_Sharpe), language="python")


st.divider()
st.caption(
    "📄 [arXiv:2005.13665](https://arxiv.org/abs/2005.13665) · "
    "💻 [pages/2_💼_Portfolio.py on GitHub]"
    "(https://github.com/jjj1231978/dl-research-demo/blob/main/pages/2_%F0%9F%92%BC_Portfolio.py) · "
    "Constitution v1.1.0."
)
st.code(
    """@article{zhang2020deep,
  title={Deep Learning for Portfolio Optimization},
  author={Zhang, Zihao and Zohren, Stefan and Roberts, Stephen},
  journal={Journal of Financial Data Science},
  year={2020}
}""",
    language="bibtex",
)
