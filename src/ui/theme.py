"""src/ui/theme.py: one visual system for the whole showcase.

Import at the top of streamlit_app.py and each page, right after
st.set_page_config:

    from src.ui.theme import apply_page_chrome, page_header, page_footer
    from src.ui.theme import plot, stat_row, correlation_heatmap

    apply_page_chrome()

Everything in this module is presentation. No model, data or metric code
belongs here.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
INK = "#1A1D24"          # body text
MUTED = "#5B6472"        # captions, axis labels, secondary values
RULE = "#E2E6EB"         # gridlines, borders, dividers
ACCENT = "#1F4E79"       # primary, and "the deep model" in every chart

# Categorical series, ordered so the first colour is the deep model and the
# rest are baselines. Lightness varies as well as hue, so the series stay
# separable in greyscale and under a red-green deficiency.
CATEGORICAL: tuple[str, ...] = (
    "#1F4E79",  # deep blue, the paper's model
    "#C77B3C",  # amber, primary benchmark
    "#4C8C7A",  # teal, secondary benchmark
    "#8A6FA0",  # violet
    "#9C4F4F",  # brick
    "#6E7B8B",  # slate, long-only / buy-and-hold
)

# Semantic assignment. Prefer these over positional colours whenever a chart
# compares the paper's model against classical methods, so the deep model is
# the same colour on every page and the reader learns it once.
SEMANTIC: Mapping[str, str] = {
    "deep": "#1F4E79",
    "benchmark": "#C77B3C",
    "classical": "#4C8C7A",
    "baseline": "#6E7B8B",
    "long_only": "#6E7B8B",
}

# Diverging scale for correlations and signed weights. Symmetric in lightness
# around a near-white midpoint, so zero reads as "nothing here".
DIVERGING: tuple[tuple[float, str], ...] = (
    (0.00, "#2C5F8A"),
    (0.25, "#7FA8C4"),
    (0.50, "#F4F1EC"),
    (0.75, "#D79A6A"),
    (1.00, "#A65A34"),
)

SEQUENTIAL: tuple[tuple[float, str], ...] = (
    (0.00, "#F4F6F8"),
    (0.50, "#7FA8C4"),
    (1.00, "#1F4E79"),
)

FONT_STACK = (
    '"Source Sans 3", "Source Sans Pro", -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", Helvetica, Arial, sans-serif'
)


# ----------------------------------------------------------------------
# Plotly template
# ----------------------------------------------------------------------
def _build_template() -> go.layout.Template:
    axis = dict(
        showgrid=False,
        zeroline=False,
        linecolor=RULE,
        linewidth=1,
        ticks="outside",
        tickcolor=RULE,
        ticklen=4,
        tickfont=dict(size=12, color=MUTED),
        title=dict(font=dict(size=12, color=MUTED), standoff=10),
        automargin=True,
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_STACK, size=13, color=INK),
            # Chart titles live in the markdown heading above the figure, not
            # inside it. Two titles is the most common Streamlit chart smell.
            # When a figure does carry one, left-align it to the axis.
            title=dict(
                font=dict(size=14, color=INK),
                x=0,
                xanchor="left",
                y=0.97,
                pad=dict(b=10),
            ),
            colorway=list(CATEGORICAL),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=8, r=8, t=28, b=8),
            xaxis={**axis},
            # Horizontal gridlines only. The reader compares values, not dates.
            yaxis={**axis, "showgrid": True, "gridcolor": RULE, "griddash": "dot"},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="left",
                x=0,
                title_text="",
                font=dict(size=12, color=MUTED),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor=RULE,
                font=dict(family=FONT_STACK, size=12, color=INK),
            ),
            hovermode="x unified",
            colorscale=dict(
                diverging=list(DIVERGING),
                sequential=list(SEQUENTIAL),
                sequentialminus=list(SEQUENTIAL),
            ),
            coloraxis=dict(
                colorbar=dict(
                    outlinewidth=0,
                    ticks="outside",
                    ticklen=4,
                    tickfont=dict(size=11, color=MUTED),
                    thickness=10,
                    len=0.8,
                )
            ),
        )
    )


pio.templates["dfs"] = _build_template()
pio.templates.default = "plotly_white+dfs"


# A showcase chart is a finished exhibit, not a workbench. The modebar's
# zoom / pan / lasso / autoscale cluster reads as leftover tooling.
PLOTLY_CONFIG: dict = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
}

# Where zoom genuinely helps (the order-book tick slice), use this instead.
PLOTLY_CONFIG_INTERACTIVE: dict = {
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def plot(fig: go.Figure, *, interactive: bool = False, height: int | None = None) -> None:
    """Render a figure with the house config.

    Use this everywhere instead of calling st.plotly_chart directly, so the
    modebar, sizing and template stay consistent.
    """
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        width="stretch",  # use_container_width is deprecated as of Streamlit 1.49
        config=PLOTLY_CONFIG_INTERACTIVE if interactive else PLOTLY_CONFIG,
        theme=None,  # our template wins; Streamlit's would override the colorway
    )


# ----------------------------------------------------------------------
# Page chrome
# ----------------------------------------------------------------------
_CSS = """
<style>
/* 1. Measure. layout="wide" with no cap gives a ~120-character line at
      1440px and an unbounded one on a 27-inch display. Cap the prose and
      leave charts, tables and dataframes at full width. */
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 3.25rem;
    padding-bottom: 6rem;
}
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] blockquote {
    max-width: 74ch;
}

/* 2. Headings. Tighter tracking and leading than the Streamlit default,
      which is tuned for dashboards rather than for reading. */
[data-testid="stMainBlockContainer"] h1 {
    font-size: 2.1rem;
    line-height: 1.18;
    letter-spacing: -0.018em;
    margin-bottom: 0.15rem;
}
[data-testid="stMainBlockContainer"] h2 {
    font-size: 1.35rem;
    letter-spacing: -0.012em;
    margin-top: 2.75rem;
    padding-top: 1.25rem;
    border-top: 1px solid #E2E6EB;
}
[data-testid="stMainBlockContainer"] h3 {
    font-size: 1.05rem;
    letter-spacing: -0.006em;
    margin-top: 1.75rem;
}

/* 3. Tabs. Give the active tab a weight change, not just a colour change. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1.75rem;
    border-bottom: 1px solid #E2E6EB;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    padding-left: 0;
    padding-right: 0;
    font-size: 0.9rem;
    color: #5B6472;
}
[data-testid="stTabs"] [aria-selected="true"] {
    font-weight: 650;
    color: #1A1D24;
}

/* 4. Sidebar. Separate navigation from controls from status. */
[data-testid="stSidebar"] { border-right: 1px solid #E2E6EB; }
[data-testid="stSidebarNav"] { padding-bottom: 0.5rem; }
[data-testid="stSidebar"] h2 {
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5B6472;
    border: none;
    margin-top: 1.9rem;
    margin-bottom: 0.4rem;
    padding-top: 0;
}
[data-testid="stSidebar"] label { font-size: 0.82rem; }

/* 5. Calm the running indicator on a page that recomputes on every
      interaction. */
[data-testid="stStatusWidget"] { font-size: 0.75rem; }

/* 6. Tabular figures wherever numbers are stacked. */
[data-testid="stMetricValue"],
[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
</style>
"""


def apply_page_chrome() -> None:
    """Inject the shared CSS. Call once per page, after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(*, title: str, citation: str, arxiv_id: str, standfirst: str) -> None:
    """A consistent masthead for the three paper pages.

    The citation belongs under the title, not inside it. An H1 reading
    "Portfolio Optimization - Zhang, Zohren, Roberts (2020)" wraps to two
    lines and buries the subject behind the authors.
    """
    st.markdown(f"# {title}")
    st.markdown(
        f'<p style="margin-top:-.35rem;color:{MUTED};font-size:.92rem;">'
        f"{citation} &nbsp;&middot;&nbsp; "
        f'<a href="https://arxiv.org/abs/{arxiv_id}" style="color:{MUTED};">'
        f"arXiv:{arxiv_id}</a></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="max-width:70ch;font-size:1.05rem;line-height:1.55;color:{INK};'
        f'border-left:3px solid {ACCENT};padding-left:1rem;margin:1.4rem 0 2rem;">'
        f"{standfirst}</p>",
        unsafe_allow_html=True,
    )


_REPO = "https://github.com/jjj1231978/dl-research-demo"
_PAPERS = (
    "Lim, Zohren &amp; Roberts (2019) &middot; "
    "Zhang, Zohren &amp; Roberts (2020) &middot; "
    "Zhang, Zohren &amp; Roberts (2019)"
)


def page_footer(updated: str | None = None) -> None:
    """One footer for all four pages, so they cannot drift apart.

    Replaces the per-page caption that ended with "Constitution v1.1.0".
    """
    stamp = updated or _dt.date.today().isoformat()
    st.divider()
    st.markdown(
        f'<p style="color:{MUTED};font-size:.8rem;line-height:1.6;">'
        f'<a href="{_REPO}" style="color:{MUTED};">Source on GitHub</a>'
        f" &nbsp;&middot;&nbsp; Papers: {_PAPERS}"
        f" &nbsp;&middot;&nbsp; Updated {stamp}</p>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------
def stat_row(stats: Sequence[tuple[str, ...]]) -> None:
    """A row of label/value pairs that never truncates.

    st.metric clips its value to the column width with an ellipsis, which is
    how "2011-01-03 to 2026-05-04" renders as "2011-01-03 -> 2...".

    Each item is (label, value) or (label, value, hint).
    """
    cols = st.columns(len(stats), gap="large")
    for col, item in zip(cols, stats):
        label, value = item[0], item[1]
        hint = item[2] if len(item) > 2 else ""
        with col:
            st.markdown(
                f'<div style="font-size:.7rem;font-weight:650;letter-spacing:.07em;'
                f'text-transform:uppercase;color:{MUTED};margin-bottom:.25rem;">'
                f"{label}</div>"
                f'<div style="font-size:1.45rem;font-weight:600;line-height:1.15;'
                f'color:{INK};font-variant-numeric:tabular-nums;white-space:nowrap;">'
                f"{value}</div>"
                + (
                    f'<div style="font-size:.75rem;color:{MUTED};margin-top:.2rem;">'
                    f"{hint}</div>"
                    if hint
                    else ""
                ),
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def correlation_heatmap(corr, *, labels: Iterable[str] | None = None) -> go.Figure:
    """Correlation matrix without the two mistakes the current one makes.

    1. The unit diagonal saturates the colour scale, so every informative
       off-diagonal value collapses to near-white. Mask it.
    2. A symmetric matrix drawn in full says everything twice. Keep the
       lower triangle.

    Cells are annotated: with 4 to 20 assets the numbers are more useful
    than a colourbar, and they survive being printed in greyscale.
    """
    z = np.asarray(corr, dtype=float).copy()
    n = z.shape[0]
    names = list(labels) if labels is not None else list(
        getattr(corr, "columns", range(n))
    )

    z[np.triu_indices(n, k=0)] = np.nan  # drop the diagonal and upper triangle
    finite = z[np.isfinite(z)]
    bound = max(float(np.abs(finite).max()) if finite.size else 1.0, 0.05)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-bound,
            zmax=bound,
            colorscale=list(DIVERGING),
            xgap=2,
            ygap=2,
            texttemplate="%{z:.2f}",
            textfont=dict(size=11),
            hovertemplate="%{y} / %{x}<br>corr %{z:.3f}<extra></extra>",
            colorbar=dict(
                thickness=10,
                len=0.7,
                outlinewidth=0,
                ticks="outside",
                tickfont=dict(size=11, color=MUTED),
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=90 + 52 * n,
        xaxis=dict(showgrid=False, ticks="", title_text="", side="bottom"),
        yaxis=dict(showgrid=False, ticks="", title_text="", autorange="reversed"),
    )
    return fig


def series_by_role(fig: go.Figure, roles: Mapping[str, str]) -> go.Figure:
    """Recolour traces by semantic role.

    Example: series_by_role(fig, {"Deep Portfolio": "deep",
                                  "Equal Weight": "baseline"})

    Keeps the deep model the same colour on every page and every tab, and
    gives it a heavier stroke so it reads as the subject of the chart.
    """
    for trace in fig.data:
        role = roles.get(getattr(trace, "name", ""))
        if not role or role not in SEMANTIC:
            continue
        colour = SEMANTIC[role]
        if getattr(trace, "line", None) is not None:
            trace.line.color = colour
            trace.line.width = 2.4 if role == "deep" else 1.6
        if getattr(trace, "marker", None) is not None:
            trace.marker.color = colour
    return fig
