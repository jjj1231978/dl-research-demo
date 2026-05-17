"""Data-loading layer for the Deep Finance Showcase.

Provides:
- `data_root()` — env-var-aware parquet directory resolver (FR-022)
- `BUNDLED_CSV_DIR` — fixed repo-relative CSV fallback directory
- `DataSnapshot` — value object describing the on-disk file backing a universe
- `get_data_snapshot()` / `load_universe()` — primary read API
- `render_data_status_sidebar()` — landing-page sidebar component

FMP-side code lives in `src/fmp.py` (function-based, mirrors the conventions
of ~/projects/ML_short_reversion/src/data/fmp.py).

Contract: see specs/001-phase-0-skeleton-data/contracts/data_loader_api.md
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    import streamlit as st  # noqa: F401  (only imported for type hints)
    from src.universes import Universe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class DeepFinanceError(Exception):
    """Base for all errors raised by src.data and src.fmp."""


class DataLoadError(DeepFinanceError):
    """Raised when a parquet or CSV file exists but cannot be read
    (corrupted, schema mismatch, wrong file type). Per FR-009, the loader
    surfaces this rather than silently re-falling-back to a different file.
    """


class DataNotFoundError(DeepFinanceError):
    """Raised when neither parquet nor CSV exists for the requested universe."""


class UniverseMembersMismatchError(DeepFinanceError):
    """Raised by load_universe() when the loaded DataFrame's symbol set does
    not match Universe.members. Catches ticker drift (FR-006 edge case).
    """


# ---------------------------------------------------------------------------
# Path resolution (FR-022 / FR-023)
# ---------------------------------------------------------------------------

# Repo root: this file lives at <repo>/src/data.py, so two parents up.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed repo-relative path for bundled CSV fallbacks. Per FR-022, this does
# NOT honour DEEP_FINANCE_DATA_DIR — the bundled CSVs ship in git and must
# always resolve relative to the repo root, otherwise an override pointing
# at a directory with no CSVs would break fresh-clone usability.
BUNDLED_CSV_DIR: Path = _REPO_ROOT / "data"


def data_root() -> Path:
    """Resolve the parquet directory honoured by both the live app and
    scripts/fetch_data.py.

    Reads `DEEP_FINANCE_DATA_DIR` from the environment; defaults to `./data/`
    relative to the current working directory.

    Per FR-022: bundled CSV fallbacks are NOT affected by this variable
    (use `BUNDLED_CSV_DIR` for those).
    """
    raw = os.environ.get("DEEP_FINANCE_DATA_DIR", "data")
    return Path(raw).expanduser().resolve()


# ---------------------------------------------------------------------------
# DataSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSnapshot:
    """Point-in-time materialisation of a universe to disk.

    See data-model.md §E2 for the contract.
    """

    universe: "Universe"
    path: Path
    source_kind: Literal["parquet", "csv", "missing"]
    refresh_ts: _dt.datetime | None
    row_count: int | None
    date_range: tuple[_dt.date, _dt.date] | None


def _read_file_metadata(path: Path, kind: Literal["parquet", "csv"]) -> tuple[int, tuple[_dt.date, _dt.date] | None]:
    """Read row count + date range from a parquet or CSV without loading the
    full DataFrame into memory if avoidable.
    """
    if kind == "parquet":
        # pyarrow can read just the metadata + the date column cheaply
        df = pd.read_parquet(path, columns=None)
    else:
        df = pd.read_csv(path)
    row_count = len(df)
    # Prefer a "date" column; fall back to "Date"; otherwise no date range
    date_col = None
    for candidate in ("date", "Date", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        return row_count, None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return row_count, None
    return row_count, (dates.min().date(), dates.max().date())


def get_data_snapshot(universe: "Universe") -> DataSnapshot:
    """Resolve which on-disk file backs a universe right now and report its
    metadata. Per contracts/data_loader_api.md.

    Raises
    ------
    DataLoadError
        If `universe.parquet_path` exists but is unreadable (per FR-009 —
        does NOT silently fall back to CSV).
    """
    parquet_path = universe.parquet_path
    csv_path = universe.csv_fallback_path

    # 1. Parquet preferred
    if parquet_path is not None and parquet_path.exists():
        try:
            row_count, date_range = _read_file_metadata(parquet_path, "parquet")
        except Exception as exc:  # noqa: BLE001 — surface any pyarrow/pandas error as DataLoadError
            raise DataLoadError(
                f"Parquet exists but is unreadable: {parquet_path} — {exc.__class__.__name__}: {exc}"
            ) from exc
        ts = _dt.datetime.fromtimestamp(parquet_path.stat().st_mtime, tz=_dt.timezone.utc)
        return DataSnapshot(universe, parquet_path, "parquet", ts, row_count, date_range)

    # 2. CSV fallback
    if csv_path is not None and csv_path.exists():
        try:
            row_count, date_range = _read_file_metadata(csv_path, "csv")
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"CSV exists but is unreadable: {csv_path} — {exc.__class__.__name__}: {exc}"
            ) from exc
        ts = _dt.datetime.fromtimestamp(csv_path.stat().st_mtime, tz=_dt.timezone.utc)
        return DataSnapshot(universe, csv_path, "csv", ts, row_count, date_range)

    # 3. Missing
    fallback_path = parquet_path if parquet_path is not None else (csv_path if csv_path is not None else _REPO_ROOT / "data" / f"{universe.name}.missing")
    return DataSnapshot(universe, fallback_path, "missing", None, None, None)


# ---------------------------------------------------------------------------
# load_universe
# ---------------------------------------------------------------------------


def load_universe(universe: "Universe") -> pd.DataFrame:
    """Load a universe's data into a pandas DataFrame.

    Parquet preferred over CSV (per FR-007). For universes with non-None
    `members`, checks that the loaded symbol set matches and raises
    UniverseMembersMismatchError on drift (FR-006 edge case).
    """
    snapshot = get_data_snapshot(universe)

    if snapshot.source_kind == "missing":
        raise DataNotFoundError(
            f"No data found for universe '{universe.name}'. "
            f"Run scripts/fetch_data.py to materialise it, or check that "
            f"DEEP_FINANCE_DATA_DIR points to the right directory."
        )

    if snapshot.source_kind == "parquet":
        df = pd.read_parquet(snapshot.path, engine="pyarrow")
    else:  # csv
        df = pd.read_csv(snapshot.path)
        # Coerce a date column to datetime if present
        for candidate in ("date", "Date"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
                break

    # Ticker-drift check
    if universe.members is not None and "symbol" in df.columns:
        loaded_members = set(df["symbol"].unique().tolist())
        expected = set(universe.members)
        if loaded_members != expected:
            missing = expected - loaded_members
            extra = loaded_members - expected
            raise UniverseMembersMismatchError(
                f"Universe '{universe.name}': loaded symbols {sorted(loaded_members)} "
                f"do not match expected {sorted(expected)} "
                f"(missing: {sorted(missing) or 'none'}; extra: {sorted(extra) or 'none'})"
            )

    return df


# ---------------------------------------------------------------------------
# Sidebar component (per contracts/landing_page_sidebar.md)
# ---------------------------------------------------------------------------


def render_data_status_sidebar(sidebar) -> None:
    """Render the data-status component into a Streamlit sidebar.

    Imports streamlit lazily so this module is importable in non-Streamlit
    contexts (e.g., scripts/fetch_data.py).
    """
    import streamlit as st  # noqa: F401
    from src import GITHUB_REPO_URL
    from src.universes import UNIVERSES

    sidebar.markdown("## Data status")

    for universe in UNIVERSES.values():
        snapshot = get_data_snapshot(universe)
        if snapshot.source_kind == "parquet":
            icon = "✓"
            caption = f"Refreshed {snapshot.refresh_ts:%Y-%m-%d}"
        elif snapshot.source_kind == "csv":
            icon = "⚠"
            caption = (
                "Bundled CSV fallback — run `scripts/fetch_data.py` for live data"
            )
        else:
            icon = "⊝"
            # Special-case sp500_100: signal OD-2 pending confirmation
            if universe.name == "sp500_100":
                caption = "Not yet available — 100-ticker list pending confirmation (see OD-2)"
            else:
                caption = "Not yet available"
        sidebar.markdown(f"**{icon} {universe.label}** — {caption}")

    sidebar.divider()

    sidebar.markdown("## Refresh data")
    sidebar.text_input(
        "FMP API key (optional)",
        type="password",
        key="fmp_api_key_input",
        help=(
            "Captured for a future in-app refresh feature. Phase 0 keeps "
            "the fetch workflow as a developer-local script (see "
            "`scripts/fetch_data.py`)."
        ),
    )

    sidebar.divider()

    sidebar.markdown("## Links")
    sidebar.markdown(
        f"- 🔗 [GitHub repository]({GITHUB_REPO_URL})\n"
        f"- 📄 [Project brief](Project_brief.md)\n"
    )


# FMP client lives in src/fmp.py (function-based API mirroring
# ~/projects/ML_short_reversion/src/data/fmp.py). Import as:
#     from src.fmp import (
#         fetch_historical_prices,
#         fetch_current_sp500_constituents,
#         fetch_historical_sp500_constituents,
#         fetch_shares_outstanding,
#     )
