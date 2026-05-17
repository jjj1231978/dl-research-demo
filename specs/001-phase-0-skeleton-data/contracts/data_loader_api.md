# Contract: `src/data.py` — data loader API

**Importer**: `streamlit_app.py`, future `pages/*.py`, `tests/integration/`.

## Public surface

```python
from src.data import (
    data_root,              # () → pathlib.Path — resolves DEEP_FINANCE_DATA_DIR
    get_data_snapshot,      # Universe → DataSnapshot
    load_universe,          # Universe → pandas.DataFrame
    render_data_status_sidebar,  # st.sidebar → None (side-effecting)
)
```

**FMP-side code** lives in `src/fmp.py` (function-based API mirroring the
conventions of `~/projects/ML_short_reversion/src/data/fmp.py`):

```python
from src.fmp import (
    fetch_historical_prices,             # multi-symbol, per-ticker cache
    fetch_current_sp500_constituents,
    fetch_historical_sp500_constituents,
    fetch_shares_outstanding,            # returns None on 404 (OD-3 skip)
    DEFAULT_FMP_CACHE,                   # Path("~/data_lake/fmp/deep-finance/")
)
```

`src/fmp.py` is consumed only by `scripts/fetch_data.py`; the live app never
imports it.

Anything not in this list is implementation detail and MUST NOT be imported
by callers.

---

## `data_root() -> pathlib.Path`

**Purpose**: Single source of truth for the parquet directory location.
Honoured by both the live app (read side) and `scripts/fetch_data.py`
(write side).

**Behavior**:

- Reads `os.environ.get("DEEP_FINANCE_DATA_DIR", "data")`.
- Expands `~` (`Path.expanduser()`).
- Resolves to an absolute path (`Path.resolve()`).
- Does NOT create the directory — callers that need it ensure their own
  parent dirs exist.

**Default**: `./data/` relative to the current working directory. On HF
Space (env unset, cwd = `/app`), this resolves to `/app/data/` — the
checked-in parquets the Docker image baked in.

**Override behaviour on developer's machine**: setting
`DEEP_FINANCE_DATA_DIR=~/data_lake/deep-finance/` re-points the fetch
script and the loader at that directory.

**Does NOT affect**: the bundled CSV fallback paths
(`data/aapl.csv`, `data/portfolio_data.csv`). Those resolve relative to the
repository root via a separate constant `BUNDLED_CSV_DIR` so that an
override pointing at an empty `~/data_lake/deep-finance/` does not break
fresh-clone usability (FR-022).

---

## `get_data_snapshot(universe: Universe) -> DataSnapshot`

**Purpose**: Resolve which on-disk file backs a universe right now and report
its metadata. Pure function from `Universe` to the `DataSnapshot` value
object defined in `data-model.md`.

**Behavior**:

1. If `data_root() / f"{universe.name}.parquet"` exists and is readable →
   return `DataSnapshot(source_kind="parquet", path=…, refresh_ts=mtime,
   row_count=…, date_range=…)`. (FR-022 — parquet location honours the
   `DEEP_FINANCE_DATA_DIR` override.)
2. Else if `universe.csv_fallback_path` exists and is readable (the CSV
   fallback path is fixed relative to the repo root via `BUNDLED_CSV_DIR`,
   NOT `data_root()`) → return `DataSnapshot(source_kind="csv", path=…,
   refresh_ts=mtime, row_count=…, date_range=…)`.
3. Else → return `DataSnapshot(source_kind="missing", path=…,
   refresh_ts=None, row_count=None, date_range=None)`.

**Errors**:

- If `universe.parquet_path` exists but is unreadable (corrupted, schema
  mismatch, wrong file type) → raise `DataLoadError` with a message naming
  the file and the underlying pyarrow error. Do NOT fall back to CSV (FR-009).
- If `universe.csv_fallback_path` exists but is unreadable → raise
  `DataLoadError` with a message naming the file.

**Caching**: this function is wrapped in `@st.cache_data` with `ttl=300`
(5 minutes) when called from the live app, so the data-status sidebar
re-renders without re-stat'ing every page navigation. The cache is keyed on
`universe.name`. Tests bypass the cache via `st.cache_data.clear()` in a
fixture.

---

## `load_universe(universe: Universe) -> pd.DataFrame`

**Purpose**: Load the universe's data into a pandas DataFrame, with parquet
preferred over CSV.

**Behavior**:

1. Determine which file via `get_data_snapshot(universe)`.
2. If `source_kind == "parquet"`: `pd.read_parquet(path, engine="pyarrow")`.
3. If `source_kind == "csv"`: `pd.read_csv(path)`; coerce `date` column to
   datetime; pivot to long format if the CSV is wide-format (the bundled
   `portfolio_data.csv` is wide). Schema must match the parquet schema
   columns (less `shares_outstanding`, which never exists in CSVs).
4. If `source_kind == "missing"`: raise `DataNotFoundError` with a clear
   message ("No data found for universe '{name}'. Run scripts/fetch_data.py
   to materialise it.").

**Caching**: `@st.cache_data(ttl=3600)` keyed on `universe.name`.

**Validation** (FR-006, edge case "ticker drift"):

- For universes with a non-`None` `members` list, the loaded DataFrame's
  unique `symbol` set MUST equal `set(universe.members)`. Mismatch raises
  `UniverseMembersMismatchError` naming the missing or extra tickers.

---

## `render_data_status_sidebar(sidebar: st.sidebar) -> None`

**Purpose**: Render the data-status component into a Streamlit sidebar. Used
by the landing page and any future page that wants the same widget.

**Behavior**:

1. Show heading "Data status".
2. For each universe in `src.universes.UNIVERSES.values()`:
   - Build a `DataSnapshot` for the universe.
   - Render one row:
     - `source_kind == "parquet"`: `✓ {universe.label}` — `Refreshed {refresh_ts:%Y-%m-%d}`
     - `source_kind == "csv"`: `⚠ {universe.label}` — `Bundled CSV fallback — run scripts/fetch_data.py for live data` (the exact copy text required by FR-008 and SC-007).
     - `source_kind == "missing"`: `⊝ {universe.label}` — `Not yet available` (e.g., `sp500_100` before OD-2 confirmation).
3. Show heading "Refresh data".
4. Render `st.text_input("FMP API key (optional)", type="password", key="fmp_api_key_input")` — purely captured into `st.session_state`; no automatic fetch is triggered (Phase 0 keeps the fetch script developer-local; the in-app refresh feature is a Phase 1+ enhancement).
5. Show "GitHub repository" link (URL configured via a constant in
   `src/__init__.py` so it can be updated in one place when the repo is
   created upstream).

**Side effects**: writes to `st.session_state["fmp_api_key_input"]`. Pure
otherwise.

---

## FMP module (`src/fmp.py`)

Function-based API (no client class), mirroring the conventions of
`~/projects/ML_short_reversion/src/data/fmp.py`. Used only by
`scripts/fetch_data.py`.

**Module-level constants**:
- `FMP_BASE = "https://financialmodelingprep.com/stable"` — the `/stable/`
  namespace (the legacy `/api/v3/` returns 403 on Premium tiers).
- `DEFAULT_FMP_CACHE = Path("~/data_lake/fmp/deep-finance/").expanduser()` —
  per-ticker parquet cache root. Distinct from `DEEP_FINANCE_DATA_DIR`
  (which controls OUTPUT parquet location, not the cache).
- `DEFAULT_REFRESH_DAYS = 7` — trailing window re-fetched on each
  incremental update (catches FMP back-revisions like split adjustments
  and late corrections).

**Public surface**:

```python
fetch_historical_prices(
    symbols: list[str], start: date, end: date, *,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,        # FMP Starter tier cap is 250
    refresh_days: int = DEFAULT_REFRESH_DAYS,
) -> pd.DataFrame                          # long-format

fetch_current_sp500_constituents(*, cache_dir=None, api_key=None) -> pd.DataFrame
fetch_historical_sp500_constituents(*, cache_dir=None, api_key=None) -> pd.DataFrame

fetch_shares_outstanding(
    symbols, start, end, *, cache_dir=None, api_key=None, rate_limit_per_min=240,
) -> pd.DataFrame | None                   # None on 404 (OD-3 warn-and-skip)
```

**Caching pattern**:
- Per-ticker parquet at `{cache_dir}/prices/by_symbol/{TICKER}.parquet`.
- Membership at `{cache_dir}/membership/sp500_current.parquet` and
  `sp500_historical.parquet`.
- Shares at `{cache_dir}/shares/by_symbol/{TICKER}.parquet`.
- Incremental updates re-fetch only `start` ↔ `cache_max - refresh_days`,
  then merge-and-write atomically (existing rows win on key collision).
- Tickers with no cached parquet are full-history fetched from `start`.

**Error handling**:
- HTTP 5xx and timeouts: standard urllib retry semantics; one explicit retry
  on 429 after a 60 s sleep.
- HTTP 403 / 404: `_fetch_with_429_retry` returns `None` so callers can
  treat unavailable endpoints (e.g., shares-outstanding on a tier without
  it) as warn-and-skip rather than failure.
- `_resolve_api_key()` raises `EnvironmentError` when `FMP_API_KEY` is
  missing.
- `fetch_historical_prices()` refuses to assemble a wide matrix if >5 % of
  symbols failed for batches of ≥20 symbols (defends against silent
  partial caches from throttling).

**Logging**: standard `logging.getLogger(__name__)`; per-symbol INFO line;
URLs containing the API key are never logged.

---

## Exception hierarchy

```text
DataLoadError              # parquet/CSV unreadable
DataNotFoundError          # parquet AND csv both absent
UniverseMembersMismatchError  # ticker drift detected
```

All inherit from a common `DeepFinanceError` base in `src/data.py`.
`src/fmp.py` uses Python's built-in `EnvironmentError` for missing
credentials and raw `urllib.error.HTTPError` for unrecovered HTTP failures.
