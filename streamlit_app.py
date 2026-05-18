"""Deep Finance Showcase — landing page (Page 0).

Lands the visitor with the three paper cards (DeepLOB, Deep Momentum
Networks, Deep Portfolio Optimization), a Common Thread section, and the
data-status sidebar.

Spec: specs/001-phase-0-skeleton-data/spec.md §"User Story 1"
Contract: specs/001-phase-0-skeleton-data/contracts/landing_page_sidebar.md
"""
import streamlit as st

from src.data import render_data_status_sidebar

# -- Page config (port and headless mode come from .streamlit/config.toml) ----
st.set_page_config(
    page_title="Deep Finance Showcase",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Sidebar (data status, FMP key input, links) -----------------------------
render_data_status_sidebar(st.sidebar)

# -- Hero ---------------------------------------------------------------------
st.title("📈 Deep Learning for Quantitative Finance")
st.markdown(
    "**An interactive showcase of three influential papers from the "
    "Oxford-Man Institute of Quantitative Finance, applying deep learning "
    "to canonical quant-finance problems.**"
)

st.divider()

# -- Three paper cards --------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
### 📈 Deep Momentum Networks

*Lim, Zohren, Roberts (2019)*
*Journal of Financial Data Science*

[arXiv:1904.04912](https://arxiv.org/abs/1904.04912)

> *Enhancing Time Series Momentum Strategies Using Deep Neural Networks.*
> Train a neural network end-to-end to directly output position sizes that
> maximize Sharpe ratio — bypassing the forecast-then-size two-step.

"""
    )
    st.page_link("pages/1_📈_Momentum.py", label="Explore Momentum →", icon="▶")

with col2:
    st.markdown(
        """
### 💼 Deep Portfolio Optimization

*Zhang, Zohren, Roberts (2020)*

[arXiv:2005.13665](https://arxiv.org/abs/2005.13665)

> *Deep Learning for Portfolio Optimization.*
> A long-only portfolio with softmax output, trained by gradient ascent to
> maximize Sharpe directly — no covariance matrix, no expected-return
> forecast.

"""
    )
    st.page_link("pages/2_💼_Portfolio.py", label="Explore Portfolio →", icon="▶")

with col3:
    st.markdown(
        """
### 📖 DeepLOB

*Zhang, Zohren, Roberts (2019)*
*IEEE Transactions on Signal Processing*

[arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

> *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.*
> A CNN + Inception + LSTM architecture that learns universal features of
> limit order book microstructure, transferable across instruments.

"""
    )
    st.page_link("pages/3_📖_Order_Book.py", label="Explore Limit Order Book →", icon="▶")

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
movement directions (Limit Order Book, via 3-way Softmax). Training optimises
the **financial objective** directly: negative Sharpe for Momentum and
Portfolio, cross-entropy on the smoothed class labels for Limit Order Book. No
explicit forecast is ever produced — there is nothing to "forecast then
size" because the network's output is the size.

This bypass-and-optimise pattern is the conceptual contribution that
unifies the three papers. Each page walks through how it manifests in its
domain, compares against the classical baselines, and lets you scrub
through the live model on the paper's canonical data.
"""
)

# -- Footer -------------------------------------------------------------------
st.divider()
st.caption(
    "Source code on [GitHub](https://github.com/jjj1231978/dl-research-demo). "
    "Deployed on Hugging Face Spaces. Constitution v1.1.0."
)
