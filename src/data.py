"""Data-loading layer for the Deep Finance Showcase.

Provides:
- `data_root()` — env-var-aware parquet directory resolver (FR-022)
- `BUNDLED_CSV_DIR` — fixed repo-relative CSV fallback directory
- `DataSnapshot` — value object describing the on-disk file backing a universe
- `get_data_snapshot()` / `load_universe()` — primary read API
- `render_data_status_sidebar()` — landing-page sidebar component
- `FMPClient` — thin REST wrapper used by scripts/fetch_data.py

Contract: see specs/001-phase-0-skeleton-data/contracts/data_loader_api.md
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import requests

if TYPE_CHECKING:
    import streamlit as st  # noqa: F401  (only imported for type hints)
    from src.universes import Universe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class DeepFinanceError(Exception):
    """Base for all errors raised by src.data."""


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


class FMPAuthError(DeepFinanceError):
    """FMP returned 401/403 — credential missing or rejected."""


class FMPRateLimitError(DeepFinanceError):
    """FMP returned 429 — rate-limit exhaustion."""


class FMPClientError(DeepFinanceError):
    """FMP returned an unhandled 4xx (other than the OD-3 shares-outstanding skip)."""


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


# ---------------------------------------------------------------------------
# FMPClient (used only by scripts/fetch_data.py)
# ---------------------------------------------------------------------------


class FMPClient:
    """Thin REST wrapper over the FMP endpoints used by the fetch script.

    Per contracts/data_loader_api.md §"FMPClient" — `shares_outstanding`
    returns None on 403/404 to support the OD-3 warn-and-skip path.
    """

    _BASE = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str, *, timeout: int = 30, max_retries: int = 3):
        if not api_key:
            raise FMPAuthError("FMP API key is missing or empty")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()

    # ---- internal helpers ----

    def _get(self, path: str, params: dict | None = None, *, allow_shares_outstanding_skip: bool = False):
        """GET <BASE>/<path>?apikey=...&<params>. Returns parsed JSON.

        Retries 5xx and timeouts with exponential backoff. Raises typed
        exceptions for 4xx (except when `allow_shares_outstanding_skip` is
        True, in which case 403/404 returns None per OD-3).
        """
        url = f"{self._BASE}/{path}"
        params = {**(params or {}), "apikey": self._api_key}
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise FMPClientError(f"Network error after {self._max_retries + 1} attempts: {exc}") from exc

            status = resp.status_code
            if status in (200, 201):
                return resp.json()
            if status in (403, 404) and allow_shares_outstanding_skip:
                return None
            if status in (401, 403):
                # Don't log the URL (would leak the key)
                raise FMPAuthError(f"FMP authentication rejected (HTTP {status}) on {path}")
            if status == 429:
                raise FMPRateLimitError(f"FMP rate-limit exhausted (HTTP 429) on {path}")
            if 400 <= status < 500:
                raise FMPClientError(f"FMP returned HTTP {status} on {path}")
            # 5xx — retry
            if attempt < self._max_retries:
                time.sleep(2 ** attempt)
                continue
            raise FMPClientError(f"FMP returned HTTP {status} on {path} after {self._max_retries + 1} attempts")

        raise FMPClientError(f"Exhausted retries on {path}: {last_exc}")

    # ---- public API ----

    def historical_prices(self, symbol: str, *, start: _dt.date, end: _dt.date) -> pd.DataFrame:
        """Fetch historical EOD prices for a symbol. Returns a long-format
        DataFrame: date, symbol, open, high, low, close, volume.
        """
        data = self._get(
            f"historical-price-full/{symbol}",
            params={"from": start.isoformat(), "to": end.isoformat()},
        )
        hist = data.get("historical", []) if isinstance(data, dict) else []
        if not hist:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(hist)
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
        return df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)

    def historical_sp500_constituents(self) -> pd.DataFrame:
        """Fetch the historical S&P 500 constituents list (used to build the
        100-stock universe without survivorship bias per FR-011).
        """
        data = self._get("historical/sp500_constituent")
        if not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def shares_outstanding(self, symbol: str, *, start: _dt.date, end: _dt.date) -> pd.DataFrame | None:
        """Fetch shares-outstanding history for a symbol.

        Returns None (and logs a warning) if the FMP tier in use does not
        expose the endpoint (HTTP 403/404), per OD-3 warn-and-skip (FR-012).
        """
        data = self._get(
            f"historical/shares_float/{symbol}",
            allow_shares_outstanding_skip=True,
        )
        if data is None:
            log.warning(
                "shares-outstanding endpoint unavailable for %s — DWP benchmark will be unavailable",
                symbol,
            )
            return None
        if not isinstance(data, list) or not data:
            return pd.DataFrame(columns=["date", "symbol", "shares_outstanding"])
        df = pd.DataFrame(data)
        df["symbol"] = symbol
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        if "floatShares" in df.columns:
            df = df.rename(columns={"floatShares": "shares_outstanding"})
        elif "outstandingShares" in df.columns:
            df = df.rename(columns={"outstandingShares": "shares_outstanding"})
        keep = [c for c in ("date", "symbol", "shares_outstanding") if c in df.columns]
        # filter to date window
        if "date" in df.columns:
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df[keep].sort_values("date").reset_index(drop=True)
