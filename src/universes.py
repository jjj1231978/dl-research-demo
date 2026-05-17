"""Universe definitions for the Deep Finance Showcase.

Each universe is a named collection of tradeable instruments with metadata
about provenance and which page consumes it. The actual price/return data
lives in parquet (preferred) or CSV (fallback) files resolved via
`Universe.parquet_path` and `Universe.csv_fallback_path` — both are
lazy properties so the `DEEP_FINANCE_DATA_DIR` env var (FR-022) takes effect
mid-process if changed.

See specs/001-phase-0-skeleton-data/data-model.md §E1 for the contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# 20-stock ticker list, sourced directly from data/portfolio_data.csv header
# (FR-006). Order matches the CSV column order.
_SP500_20_TICKERS = [
    "AAPL", "ABT", "AEP", "AXP", "BAC", "CI", "GD", "GE", "HON", "MMM",
    "MO", "MRK", "NEM", "NKE", "NSC", "PFE", "PG", "PTC", "SNA", "SO",
]


@dataclass(frozen=True)
class Universe:
    """A named collection of tradeable instruments.

    Attributes are immutable; path resolution happens lazily via the two
    `@property` accessors so `DEEP_FINANCE_DATA_DIR` overrides take effect
    on every call.
    """

    name: str
    label: str
    members: list[str] | None
    source: Literal["fmp", "cme_data_lake", "kaggle", "local_csv"]
    consumers: list[Literal["momentum", "portfolio", "order_book", "landing"]]
    # When the universe has no parquet path at all (e.g., aapl_toy uses CSV
    # only), set has_parquet=False so the resolver doesn't synthesize one.
    has_parquet: bool = True
    # When the universe has a bundled CSV fallback, name its filename.
    # None means "no CSV fallback — parquet or bust".
    csv_fallback_name: str | None = None

    @property
    def parquet_path(self) -> Path | None:
        """Resolve the parquet path under DEEP_FINANCE_DATA_DIR (FR-022).

        Returns None when the universe has no parquet representation (e.g.,
        aapl_toy uses CSV only).
        """
        if not self.has_parquet:
            return None
        # Import here to avoid circular imports during package init.
        from src.data import data_root
        return data_root() / f"{self.name}.parquet"

    @property
    def csv_fallback_path(self) -> Path | None:
        """Resolve the bundled CSV fallback path. ALWAYS repo-relative —
        does NOT honour DEEP_FINANCE_DATA_DIR (FR-022).
        """
        if self.csv_fallback_name is None:
            return None
        from src.data import BUNDLED_CSV_DIR
        return BUNDLED_CSV_DIR / self.csv_fallback_name


UNIVERSES: dict[str, Universe] = {
    "etf_basket": Universe(
        name="etf_basket",
        label="ETF Basket (VTI / AGG / DBC / VIXY)",
        # Per OD-1 clarification (Session 2026-05-16): VIXY is the VIX proxy.
        members=["VTI", "AGG", "DBC", "VIXY"],
        source="fmp",
        consumers=["portfolio"],
        csv_fallback_name=None,  # no bundled CSV for the ETF basket
    ),
    "sp500_20": Universe(
        name="sp500_20",
        label="20-stock (sector-balanced S&P 500)",
        # Per FR-006: ticker list mirrors data/portfolio_data.csv header.
        members=_SP500_20_TICKERS,
        source="fmp",
        consumers=["portfolio"],
        csv_fallback_name="portfolio_data.csv",
    ),
    "sp500_100": Universe(
        name="sp500_100",
        label="100-stock S&P 500 (historical constituents)",
        # Per OD-2 clarification (Session 2026-05-16): Claude proposes a
        # 100-ticker sector-balanced list; the developer confirms before
        # scripts/fetch_data.py writes the parquet. Until confirmation:
        # members=None and the universe has no CSV fallback.
        members=None,
        source="fmp",
        consumers=["portfolio"],
        csv_fallback_name=None,
    ),
    "cme_futures": Universe(
        name="cme_futures",
        label="CME Futures (~88 ratio-adjusted continuous contracts)",
        # Members come from the developer's data lake at fetch time.
        members=None,
        source="cme_data_lake",
        consumers=["momentum"],
        csv_fallback_name=None,
    ),
    "aapl_toy": Universe(
        name="aapl_toy",
        label="AAPL (single-asset toy mode)",
        members=["AAPL"],
        source="local_csv",
        consumers=["momentum"],
        has_parquet=False,        # AAPL is CSV-only — no parquet path
        csv_fallback_name="aapl.csv",
    ),
}
