"""Integration tests for the Phase 2 Portfolio page (FR-022).

Six parametrised cases per `contracts/portfolio_page_ui.md` §"Render
assertions": (universe × parquet) = {etfs, 20stock} × {absent, present}
plus two checkpoint-presence variants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE_FILE = str(REPO_ROOT / "pages" / "2_💼_Portfolio_Optimization.py")


def _make_minimal_etf_parquet(path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for sym in ("VTI", "AGG", "DBC", "VIXY"):
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        for d, p in zip(pd.date_range("2019-01-02", periods=400, freq="B"), prices):
            rows.append({
                "date": d.date(), "symbol": sym,
                "open": float(p), "high": float(p), "low": float(p),
                "close": float(p), "volume": 1_000_000,
            })
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


def _make_minimal_sp500_20_parquet(path: Path) -> None:
    rng = np.random.default_rng(7)
    rows = []
    syms = ["AAPL", "ABT", "AEP", "AXP", "BAC", "CI", "GD", "GE", "HON", "MMM",
            "MO", "MRK", "NEM", "NKE", "NSC", "PFE", "PG", "PTC", "SNA", "SO"]
    for sym in syms:
        prices = 50.0 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        for d, p in zip(pd.date_range("2019-01-02", periods=400, freq="B"), prices):
            rows.append({
                "date": d.date(), "symbol": sym,
                "open": float(p), "high": float(p), "low": float(p),
                "close": float(p), "volume": 1_000_000,
            })
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


def _make_minimal_backtest_parquet(path: Path) -> None:
    rng = np.random.default_rng(11)
    rows = []
    methods_per_universe = {
        "etfs": ["equal_weight", "min_variance", "max_diversification",
                 "alloc_25_25_25_25", "alloc_50_10_20_20",
                 "alloc_10_50_20_20", "alloc_40_40_10_10"],
        "20stock": ["equal_weight", "min_variance", "max_diversification"],
    }
    for universe, methods in methods_per_universe.items():
        for method in methods:
            for vol_scaling in (False, True):
                for cost_rate in (0.0001, 0.0010):
                    for d in pd.date_range("2020-04-01", periods=200, freq="B"):
                        rows.append({
                            "date": d.date(), "universe": universe,
                            "method": method, "vol_scaling": vol_scaling,
                            "cost_rate": cost_rate,
                            "portfolio_return": float(rng.normal(0, 0.005)),
                        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


@pytest.fixture
def page_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              request: pytest.FixtureRequest) -> dict:
    """Parametrized by (universe, parquet_present).

    Sets DEEP_FINANCE_DATA_DIR + DEEP_FINANCE_BACKTESTS_DIR to tmp_path,
    optionally seeds the requested universe's parquet + the backtest panel.
    """
    universe, parquet_present = request.param
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    backtests_dir = tmp_path / "backtests"
    backtests_dir.mkdir()
    if parquet_present:
        if universe == "etfs":
            _make_minimal_etf_parquet(data_dir / "etf_basket.parquet")
        else:
            _make_minimal_sp500_20_parquet(data_dir / "sp500_20.parquet")
        _make_minimal_backtest_parquet(backtests_dir / "portfolio_results.parquet")
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DEEP_FINANCE_BACKTESTS_DIR", str(backtests_dir))
    return {"universe": universe, "parquet_present": parquet_present}


def _run_app(universe: str) -> AppTest:
    at = AppTest.from_file(PAGE_FILE, default_timeout=60)
    at.session_state["portfolio_universe"] = (
        "ETF basket" if universe == "etfs" else "20-stock"
    )
    at.run()
    return at


def _all_markdown(at: AppTest) -> str:
    parts = [md.value for md in at.main.markdown]
    for tab in getattr(at, "tabs", []) or []:
        parts.extend(md.value for md in tab.markdown)
        for inner in getattr(tab, "tabs", []) or []:
            parts.extend(md.value for md in inner.markdown)
    return "\n".join(parts)


def _all_warnings(at: AppTest) -> str:
    parts = [w.value for w in at.warning]
    for tab in getattr(at, "tabs", []) or []:
        parts.extend(w.value for w in tab.warning)
    return "\n".join(parts)


def _all_dataframes(at: AppTest):
    out = list(at.dataframe)
    for tab in getattr(at, "tabs", []) or []:
        out.extend(tab.dataframe)
        for inner in getattr(tab, "tabs", []) or []:
            out.extend(inner.dataframe)
    return out


_MATRIX = [
    pytest.param(("etfs",    False), id="A_etfs_absent"),
    pytest.param(("etfs",    True),  id="B_etfs_present"),
    pytest.param(("20stock", False), id="C_20stock_absent"),
    pytest.param(("20stock", True),  id="D_20stock_present"),
]


@pytest.mark.parametrize("page_env", _MATRIX, indirect=True)
def test_portfolio_page_renders(page_env):
    """All 4 cases: AppTest.run() raises no exception (FR-022)."""
    at = _run_app(page_env["universe"])
    assert not at.exception, (
        f"Portfolio page raised in case (universe={page_env['universe']}, "
        f"parquet_present={page_env['parquet_present']}): "
        f"{[str(e) for e in at.exception]}"
    )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param(("etfs", True), id="etfs_present")],
    indirect=True,
)
def test_substrate_disclosure_visible(page_env):
    """Tab 1 contains the copy-text invariant from the contract."""
    at = _run_app(page_env["universe"])
    assert not at.exception, str([str(e) for e in at.exception])
    md = _all_markdown(at)
    needle = "qualitative claim (Deep Portfolio in top 3 by Sharpe)"
    assert needle in md, (
        f"Substrate-disclosure substring {needle!r} not found.\n"
        f"Actual markdown (truncated):\n{md[:1500]}"
    )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param(("etfs", False), id="etfs_absent")],
    indirect=True,
)
def test_fallback_banner_when_parquet_absent(page_env):
    """Toy-mode banner appears when parquet missing."""
    at = _run_app(page_env["universe"])
    assert not at.exception, str([str(e) for e in at.exception])
    haystack = _all_markdown(at) + "\n" + _all_warnings(at)
    for needle in ("single-universe toy mode", "scripts/fetch_data.py"):
        assert needle in haystack, (
            f"Fallback-banner substring {needle!r} not found.\n"
            f"Markdown+Warnings: {haystack[:1500]}"
        )


@pytest.mark.parametrize(
    "page_env",
    [pytest.param(("etfs", True), id="etfs_present")],
    indirect=True,
)
def test_tab4a_has_three_panel_dataframes(page_env):
    """Tab 4A renders 3 dataframes (one per cost/vol panel)."""
    at = _run_app(page_env["universe"])
    assert not at.exception, str([str(e) for e in at.exception])
    rendered_dfs = _all_dataframes(at)
    # The page may render additional helper tables (Tab 2 mini-table etc.)
    # so we look for 3+ dataframes total with the 8-metric column set.
    expected_cols = {
        "E[Return]", "Vol", "Downside Deviation", "MDD",
        "Sharpe", "Sortino", "% +ve Returns", "Ave. P / Ave. L",
    }
    panel_dfs = [
        d for d in rendered_dfs
        if d.value is not None and expected_cols.issubset(set(d.value.columns))
    ]
    assert len(panel_dfs) >= 3, (
        f"Expected 3+ Tab 4A panel dataframes with the 8 metric columns; "
        f"got {len(panel_dfs)}. All rendered df col sets: "
        f"{[list(d.value.columns) if d.value is not None else None for d in rendered_dfs]}"
    )
