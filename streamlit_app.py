"""Deep Finance Showcase — landing page (Page 0).

Lands the visitor with the three paper cards (DeepLOB, Deep Momentum
Networks, Deep Portfolio Optimization), a Common Thread section, and the
data-status sidebar.

Spec: specs/001-phase-0-skeleton-data/spec.md §"User Story 1"
Contract: specs/001-phase-0-skeleton-data/contracts/landing_page_sidebar.md
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import render_data_status_sidebar
from src.data.futures import TEST_START

# -- Page config (port and headless mode come from .streamlit/config.toml) ----
st.set_page_config(
    page_title="Deep Finance Showcase",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

from src.ui.theme import (
    ACCENT,
    apply_page_chrome,
    page_footer,
    page_header,
    plot,
    stat_row,
)

apply_page_chrome()

# -- Sidebar (data status, FMP key input, links) -----------------------------
render_data_status_sidebar(st.sidebar)

# -- Hero ---------------------------------------------------------------------
st.title("Deep Learning for Quantitative Finance")
st.markdown(
    "**An interactive showcase of three influential papers from the "
    "Oxford-Man Institute of Quantitative Finance, applying deep learning "
    "to canonical quant-finance problems.**"
)

st.divider()

# -- Headline outcomes --------------------------------------------------------
# The cards described the papers but promised nothing, so a visitor had to
# click through to find out whether any of it worked. These are read from the
# same pre-computed panels the pages themselves use, rather than typed in, so
# a card cannot drift away from the page it links to.

_BACKTESTS = Path(
    os.environ.get("DEEP_FINANCE_BACKTESTS_DIR",
                   str(Path(__file__).resolve().parent / "data" / "backtests"))
)


def _sharpe(returns) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 2 or np.std(r) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r) * np.sqrt(252))


@st.cache_data(show_spinner=False)
def _headlines() -> dict[str, str]:
    out: dict[str, str] = {}

    # Momentum: the out-of-sample window only. The panel also spans the
    # 2010-2019 training window, where the MLP's memorised Sharpe is not a
    # claim anyone should put on a landing page.
    try:
        m = pd.read_parquet(_BACKTESTS / "momentum_results.parquet")
        m = m[(m["date"] >= pd.Timestamp(TEST_START)) & (m["vol_scaling"])]
        best = max(
            (_sharpe(m[m["strategy"] == k].groupby("date")["daily_return"].mean())
             for k in ("mlp_sharpe", "lstm_sharpe")),
            default=float("nan"),
        )
        if np.isfinite(best):
            out["momentum"] = f"Sharpe {best:.2f} out of sample"
    except Exception:  # noqa: BLE001 — a card without a number still renders
        pass

    # Portfolio: the paper's own four-ETF basket, scaled, at 1bp costs.
    try:
        p_ = pd.read_parquet(_BACKTESTS / "portfolio_results.parquet")
        p_ = p_[(p_["universe"] == "etfs") & (p_["vol_scaling"])
                & (p_["cost_rate"] == 0.0001)]
        deep = _sharpe(p_[p_["method"] == "deep_portfolio"]["portfolio_return"])
        rest = [
            _sharpe(p_[p_["method"] == k]["portfolio_return"])
            for k in p_["method"].unique() if k != "deep_portfolio"
        ]
        rest = [x for x in rest if np.isfinite(x)]
        if np.isfinite(deep) and rest:
            out["portfolio"] = f"Sharpe {deep:.2f} vs {max(rest):.2f} best classical"
    except Exception:  # noqa: BLE001
        pass

    # Order book: macro-F1 at k=10, the paper's own headline metric.
    try:
        l_ = pd.read_parquet(_BACKTESTS / "lob_results.parquet")
        row = l_[(l_["method"] == "deeplob") & (l_["source"] == "reproduced_here")]
        if not row.empty:
            out["lob"] = f"Macro-F1 {float(row['f1_macro'].iloc[0]) * 100:.1f}%"
    except Exception:  # noqa: BLE001
        pass
    return out


_HEADLINES = _headlines()


def _outcome(key: str) -> None:
    """One number, one label, in the accent colour. No chart."""
    value = _HEADLINES.get(key)
    if not value:
        return
    st.markdown(
        f'<p style="color:{ACCENT};font-weight:650;font-size:.95rem;'
        f'margin:-.4rem 0 .9rem;">{value}</p>',
        unsafe_allow_html=True,
    )


# -- Three paper cards --------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
### Deep Momentum Networks

*Lim, Zohren, Roberts (2019)*
*Journal of Financial Data Science*

[arXiv:1904.04912](https://arxiv.org/abs/1904.04912)

> *Enhancing Time Series Momentum Strategies Using Deep Neural Networks.*
> Train a neural network end-to-end to directly output position sizes that
> maximize Sharpe ratio — bypassing the forecast-then-size two-step.

"""
    )
    _outcome("momentum")
    st.page_link("pages/1_📈_Momentum.py", label="Explore Momentum")

with col2:
    st.markdown(
        """
### Deep Portfolio Optimization

*Zhang, Zohren, Roberts (2020)*

[arXiv:2005.13665](https://arxiv.org/abs/2005.13665)

> *Deep Learning for Portfolio Optimization.*
> A long-only portfolio with softmax output, trained by gradient ascent to
> maximize Sharpe directly — no covariance matrix, no expected-return
> forecast.

"""
    )
    _outcome("portfolio")
    st.page_link("pages/2_💼_Portfolio_Optimization.py", label="Explore Portfolio Optimization")

with col3:
    st.markdown(
        """
### DeepLOB

*Zhang, Zohren, Roberts (2019)*
*IEEE Transactions on Signal Processing*

[arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

> *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.*
> A CNN + Inception + LSTM architecture that learns universal features of
> limit order book microstructure, transferable across instruments.

"""
    )
    _outcome("lob")
    st.page_link("pages/3_📖_Limit_Order_Book.py", label="Explore Order Book")

st.divider()

# -- Common Thread ------------------------------------------------------------
st.markdown(
    """
## Common Thread

All three papers share the same conceptual move: **bypass the prediction
step and optimise the financial objective end-to-end** via deep neural
networks with appropriate output activations.

The classical pattern is two-step: forecast a quantity (next-period
return, mid-price direction, expected-return vector), then convert the
forecast into a decision (position size, signal sign, portfolio weights).
Each step has its own loss; the second step's gradients never flow back to
the first. The deep-learning move is to **collapse the two steps into one
network** whose output *is* the decision — a continuous position in
`(-1, +1)` (Momentum, via Softsign), a long-only weight vector summing to
one (Portfolio, via Softmax), or a class probability over mid-price
movement directions (Order Book, via 3-way Softmax). Training optimises
the **financial objective** directly: negative Sharpe for Momentum and
Portfolio, cross-entropy on the smoothed class labels for Order Book. No
explicit forecast is ever produced — there is nothing to "forecast then
size" because the network's output is the size.

This bypass-and-optimise pattern is the conceptual contribution that
unifies the three papers. Each page walks through how it manifests in its
domain, compares against the classical baselines, and lets you scrub
through the live model on the paper's canonical data.
"""
)

# -- Footer -------------------------------------------------------------------
page_footer()
