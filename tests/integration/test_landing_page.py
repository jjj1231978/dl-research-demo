"""Integration tests for the landing page (Phase 0 US1 / FR-021 / SC-005).

Each test renders streamlit_app.py via streamlit.testing.v1.AppTest and
asserts no exception was raised. Parametrized across the four
(parquet_present, fmp_key_set) combinations from
contracts/landing_page_sidebar.md §"Render assertions".
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
APP_FILE = str(REPO_ROOT / "streamlit_app.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_etf_parquet(path: Path) -> None:
    """Write a 4-symbol etf_basket.parquet that satisfies
    contracts/fetch_data_cli.md §"Output schema" minimally so
    get_data_snapshot resolves to 'parquet' source_kind.
    """
    rows = []
    for sym in ("VTI", "AGG", "DBC", "VIXY"):
        for d in pd.date_range("2024-01-02", periods=10, freq="B"):
            rows.append({
                "date": d.date(),
                "symbol": sym,
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 1_000_000,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow")


@pytest.fixture
def parquet_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> bool:
    """Boolean fixture, parameterized via indirect=True.

    When True: writes etf_basket.parquet into a tmp dir and points
    DEEP_FINANCE_DATA_DIR at it (so get_data_snapshot resolves to parquet
    for at least one universe).
    When False: points DEEP_FINANCE_DATA_DIR at an empty tmp dir (no
    parquets exist there, so the loader falls back to CSV / missing).
    """
    flag: bool = request.param
    data_dir = tmp_path / "data_root"
    data_dir.mkdir()
    if flag:
        _make_minimal_etf_parquet(data_dir / "etf_basket.parquet")
    monkeypatch.setenv("DEEP_FINANCE_DATA_DIR", str(data_dir))
    return flag


@pytest.fixture
def fmp_key_env(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> bool:
    """Boolean fixture, parameterized via indirect=True.

    When True: sets FMP_API_KEY in the env.
    When False: deletes FMP_API_KEY from the env.
    """
    flag: bool = request.param
    if flag:
        monkeypatch.setenv("FMP_API_KEY", "test-key-not-real")
    else:
        monkeypatch.delenv("FMP_API_KEY", raising=False)
    return flag


# ---------------------------------------------------------------------------
# Tests (parametrized across the 4-cell matrix)
# ---------------------------------------------------------------------------

_MATRIX = [
    pytest.param(False, False, id="A_no_parquet_no_key"),
    pytest.param(False, True,  id="B_no_parquet_with_key"),
    pytest.param(True,  False, id="C_parquet_no_key"),
    pytest.param(True,  True,  id="D_parquet_with_key"),
]


@pytest.mark.parametrize(("parquet_present", "fmp_key_env"), _MATRIX, indirect=True)
def test_landing_page_renders_in_all_four_states(parquet_present, fmp_key_env):
    """FR-021 / SC-005: landing page renders without exception in every
    (parquet × fmp_key) combination."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, (
        f"Landing page raised an exception "
        f"(parquet_present={parquet_present}, fmp_key_env={fmp_key_env}): "
        f"{[str(e) for e in at.exception]}"
    )


@pytest.mark.parametrize(("parquet_present", "fmp_key_env"), [pytest.param(False, False, id="no_parquet_no_key")], indirect=True)
def test_fallback_notice_visible_when_parquet_absent(parquet_present, fmp_key_env):
    """FR-008 / SC-007 / US1 acceptance #3: when no parquets exist, the
    sidebar shows the literal substrings 'Bundled CSV fallback' and
    'scripts/fetch_data.py' (operator guidance)."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, str([str(e) for e in at.exception])

    # Concatenate all sidebar markdown blocks
    sidebar_text = "\n".join(md.value for md in at.sidebar.markdown)
    assert "Bundled CSV fallback" in sidebar_text, (
        f"Fallback notice substring 'Bundled CSV fallback' not found in sidebar.\n"
        f"Actual sidebar markdown:\n{sidebar_text}"
    )
    assert "scripts/fetch_data.py" in sidebar_text, (
        f"Operator-action substring 'scripts/fetch_data.py' not found in sidebar.\n"
        f"Actual sidebar markdown:\n{sidebar_text}"
    )


@pytest.mark.parametrize(("parquet_present", "fmp_key_env"), [pytest.param(False, False, id="no_parquet_no_key")], indirect=True)
def test_paper_cards_present_and_navigable(parquet_present, fmp_key_env):
    """FR-014 / SC-002 / US1 acceptance #2: three paper cards visible with
    the paper titles and arXiv links."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, str([str(e) for e in at.exception])

    # All markdown across main + sidebar (cards rendered as markdown blocks)
    all_markdown = "\n".join(md.value for md in at.main.markdown)

    expected_arxiv = [
        "arxiv.org/abs/1808.03668",  # DeepLOB
        "arxiv.org/abs/1904.04912",  # Deep Momentum Networks
        "arxiv.org/abs/2005.13665",  # Deep Portfolio Optimization
    ]
    for link in expected_arxiv:
        assert link in all_markdown, (
            f"arXiv link '{link}' not found on the landing page.\n"
            f"Actual landing-page markdown:\n{all_markdown[:2000]}"
        )

    expected_titles = ["DeepLOB", "Time Series Momentum", "Deep Learning for Portfolio"]
    for title in expected_titles:
        assert title in all_markdown, (
            f"Paper title '{title}' not found on the landing page.\n"
            f"Actual landing-page markdown:\n{all_markdown[:2000]}"
        )


@pytest.mark.parametrize(("parquet_present", "fmp_key_env"), [pytest.param(False, False, id="no_parquet_no_key")], indirect=True)
def test_common_thread_section_present(parquet_present, fmp_key_env):
    """FR-015: 'Common Thread' section present with unifying-move keywords."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, str([str(e) for e in at.exception])

    all_markdown = "\n".join(md.value for md in at.main.markdown).lower()
    assert "common thread" in all_markdown, "Missing 'Common Thread' heading"
    # Unifying-move keywords (lowercase compare): bypass + objective + gradient
    for keyword in ("bypass", "objective", "gradient"):
        assert keyword in all_markdown, (
            f"Unifying-move keyword '{keyword}' missing from Common Thread section.\n"
            f"Actual landing-page markdown (lowercased):\n{all_markdown[:2000]}"
        )
