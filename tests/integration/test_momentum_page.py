"""Integration tests for the Phase 1 Momentum page (FR-022 + the contract
specs/002-phase-1-momentum/contracts/momentum_page_ui.md §"Render assertions").

Six parametrised cases: (parquet × deep_model) = {absent, present} × {MLP, LSTM, Both}.

Each test renders `pages/1_📈_Momentum.py` via streamlit.testing.v1.AppTest
inside an env-isolated `tmp_path` (mirroring the Phase 0 fixture pattern).
The tests assert:
  - no exception during AppTest.run()  (all 6 cases)
  - substrate disclosure copy is visible              (test_substrate_disclosure_visible)
  - Tab 4 sub-tab shape (5×9 dataframe in 4A)         (test_tab4_subtabs_shape)
  - fallback banner copy when parquet absent          (test_fallback_banner_when_parquet_absent)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE_FILE = str(REPO_ROOT / "pages" / "1_📈_Momentum.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_cme_parquet(path: Path) -> None:
    """Tiny `cme_futures.parquet` with 3 contracts × 400 trading days.

    Just enough rows that the deep model's 60-day rolling window emits some
    output, and that Tab 4's per-asset box plots have multiple boxes.
    """
    rng = np.random.default_rng(42)
    rows = []
    for sym in ("CL", "ZC", "GC"):
        # Generate a synthetic price path
        rets = rng.normal(0, 0.012, 400)
        prices = 50.0 * np.cumprod(1 + rets)
        for d, p, r in zip(
            pd.date_range("2020-01-02", periods=400, freq="B"), prices, rets
        ):
            rows.append({
                "date": d.date(),
                "contract": sym,
                "asset_class": "commodity",
                "price": float(p),
                "return": float(r),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow")


def _make_minimal_backtest_parquet(path: Path) -> None:
    """Tiny `momentum_results.parquet` so Tab 4 has data to render.

    5 strategies × 2 vol_scaling × 3 contracts × 200 days = 6000 rows.
    """
    rng = np.random.default_rng(7)
    rows = []
    for strategy in ("long_only", "sgn_returns", "macd", "mlp_sharpe", "lstm_sharpe"):
        for vol_scaling in (False, True):
            for contract in ("CL", "ZC", "GC"):
                for d in pd.date_range("2020-04-01", periods=200, freq="B"):
                    rows.append({
                        "date": d.date(),
                        "contract": contract,
                        "strategy": strategy,
                        "vol_scaling": vol_scaling,
                        "daily_return": float(rng.normal(0, 0.005)),
                    })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


@pytest.fixture
def parquet_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> bool:
    """When True: writes cme_futures.parquet AND momentum_results.parquet
    into a tmp dir; points DEEP_FINANCE_DATA_DIR there. When False: points
    DEEP_FINANCE_DATA_DIR at an empty dir.
    """
    flag: bool = request.param
    data_dir = tmp_path / "data_root"
    data_dir.mkdir()
    backtests_dir = tmp_path / "backtests_root"
    backtests_dir.mkdir()
    if flag:
        _make_minimal_cme_parquet(data_dir / "cme_futures.parquet")
        _make_minimal_backtest_parquet(backtests_dir / "momentum_results.parquet")
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DEEP_FINANCE_BACKTESTS_DIR", str(backtests_dir))
    return flag


@pytest.fixture
def deep_model(request: pytest.FixtureRequest) -> str:
    """Pre-selects the sidebar's deep-model widget via Streamlit's
    session_state. The page reads `momentum_deep_model` from session_state
    to honour the test's choice without manual click simulation.
    """
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_app(deep_model: str) -> AppTest:
    """Render the Momentum page with the chosen deep-model selection."""
    at = AppTest.from_file(PAGE_FILE, default_timeout=60)
    at.session_state["momentum_deep_model"] = deep_model
    at.run()
    return at


def _all_markdown(at: AppTest) -> str:
    """Concatenate markdown from main + every tab (recursive). Tabs are
    not surfaced under at.main.markdown by Streamlit's AppTest."""
    parts = [md.value for md in at.main.markdown]
    for tab in getattr(at, "tabs", []) or []:
        parts.extend(md.value for md in tab.markdown)
        for inner_tab in getattr(tab, "tabs", []) or []:
            parts.extend(md.value for md in inner_tab.markdown)
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
        for inner_tab in getattr(tab, "tabs", []) or []:
            out.extend(inner_tab.dataframe)
    return out


# ---------------------------------------------------------------------------
# Parameterised matrix — six cases
# ---------------------------------------------------------------------------

_MATRIX = [
    pytest.param(False, "MLP",  id="A_absent_MLP"),
    pytest.param(False, "LSTM", id="B_absent_LSTM"),
    pytest.param(False, "Both", id="C_absent_Both"),
    pytest.param(True,  "MLP",  id="D_present_MLP"),
    pytest.param(True,  "LSTM", id="E_present_LSTM"),
    pytest.param(True,  "Both", id="F_present_Both"),
]


@pytest.mark.parametrize(
    ("parquet_present", "deep_model"), _MATRIX, indirect=["parquet_present"]
)
def test_momentum_page_renders(parquet_present, deep_model):
    """All six cases: AppTest.run() returns without exception (FR-022)."""
    at = _run_app(deep_model)
    assert not at.exception, (
        f"Momentum page raised in case "
        f"(parquet_present={parquet_present}, deep_model={deep_model}): "
        f"{[str(e) for e in at.exception]}"
    )


# ---------------------------------------------------------------------------
# Targeted assertion tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parquet_present", "deep_model"),
    [pytest.param(True, "MLP", id="present_MLP")],
    indirect=["parquet_present"],
)
def test_substrate_disclosure_visible(parquet_present, deep_model):
    """T019: Tab 1 contains the literal substrings from the copy-text invariants
    (contracts/momentum_page_ui.md §"Copy-text invariants")."""
    at = _run_app(deep_model)
    assert not at.exception, str([str(e) for e in at.exception])

    all_markdown = _all_markdown(at)
    for needle in (
        "BCOM commodity roots",
        "Pinnacle CLC",
        "qualitative-ordering claims hold on commodities-only",
    ):
        assert needle in all_markdown, (
            f"Substrate-disclosure substring {needle!r} not found.\n"
            f"Actual page markdown (truncated):\n{all_markdown[:2000]}"
        )


@pytest.mark.parametrize(
    ("parquet_present", "deep_model"),
    [pytest.param(True, "MLP", id="present_MLP")],
    indirect=["parquet_present"],
)
def test_tab4_subtabs_shape(parquet_present, deep_model):
    """T020: Tab 4A shows a 5-row × 9-column dataframe with the columns from
    research.md R8.
    """
    at = _run_app(deep_model)
    assert not at.exception, str([str(e) for e in at.exception])

    # Walk both main + tabs so we find dataframes nested in Tab 4 sub-tabs
    rendered_dfs = _all_dataframes(at)
    assert rendered_dfs, "No st.dataframe widgets rendered — Tab 4A is missing the table"

    expected_cols = {
        "E[Return]", "Vol", "Downside Deviation", "MDD",
        "Sharpe", "Sortino", "Calmar", "% +ve Returns", "Ave. P / Ave. L",
    }
    matched = False
    for df_widget in rendered_dfs:
        df = df_widget.value
        if df is None:
            continue
        # Accept df.columns as a superset of expected_cols, with 5 rows.
        if expected_cols.issubset(set(df.columns)) and len(df) == 5:
            matched = True
            break
    assert matched, (
        "No rendered dataframe had 5 rows AND the expected 9 columns. "
        f"Found columns: {[list(df.value.columns) if df.value is not None else None for df in rendered_dfs]}"
    )


@pytest.mark.parametrize(
    ("parquet_present", "deep_model"),
    [pytest.param(False, "MLP", id="absent_MLP")],
    indirect=["parquet_present"],
)
def test_fallback_banner_when_parquet_absent(parquet_present, deep_model):
    """T021: parquet-absent case shows the fallback banner with the operator-action
    substring (per copy-text invariants)."""
    at = _run_app(deep_model)
    assert not at.exception, str([str(e) for e in at.exception])

    all_markdown = _all_markdown(at)
    all_warnings = _all_warnings(at)
    haystack = all_markdown + "\n" + all_warnings
    for needle in ("single-asset toy mode", "scripts/fetch_futures.py"):
        assert needle in haystack, (
            f"Fallback-banner substring {needle!r} not found.\n"
            f"Markdown: {all_markdown[:1000]}\n"
            f"Warnings: {all_warnings[:1000]}"
        )
