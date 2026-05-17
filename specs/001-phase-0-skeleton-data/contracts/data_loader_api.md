# Contract: `src/data.py` — data loader API

**Importer**: `streamlit_app.py`, future `pages/*.py`, `tests/integration/`.

## Public surface

```python
from src.data import (
    data_root,              # () → pathlib.Path — resolves DEEP_FINANCE_DATA_DIR
    get_data_snapshot,      # Universe → DataSnapshot
    load_universe,          # Universe → pandas.DataFrame
    render_data_status_sidebar,  # st.sidebar → None (side-effecting)
    FMPClient,              # Class, used only by scripts/fetch_data.py
)
```

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

## `class FMPClient`

**Purpose**: Thin REST wrapper over the FMP endpoints used by
`scripts/fetch_data.py`. Not consumed by the live app.

**Public surface**:

```python
class FMPClient:
    def __init__(self, api_key: str, *, timeout: int = 30, max_retries: int = 3): ...
    def historical_prices(self, symbol: str, *, start: date, end: date) -> pd.DataFrame: ...
    def historical_sp500_constituents(self) -> pd.DataFrame: ...
    def shares_outstanding(self, symbol: str, *, start: date, end: date) -> pd.DataFrame | None: ...
        # Returns None when the endpoint returns 403/404 (OD-3 warn-and-skip path).
```

**Error handling**:

- HTTP 4xx (other than the OD-3 403/404 skip on shares-outstanding) → raise
  `FMPAuthError` for 401/403, `FMPRateLimitError` for 429, `FMPClientError`
  for everything else.
- HTTP 5xx → retry up to `max_retries` with exponential backoff (1s, 2s, 4s).
- Network timeout → retry per `max_retries`.

**Logging**: per-symbol INFO line when verbose; never logs the URL with the
key (`api_key` is stripped before any log message).

---

## Exception hierarchy

```text
DataLoadError              # parquet/CSV unreadable
DataNotFoundError          # parquet AND csv both absent
UniverseMembersMismatchError  # ticker drift detected
FMPAuthError               # 401/403 (other than the OD-3 skip)
FMPRateLimitError          # 429
FMPClientError             # other 4xx
```

All inherit from a common `DeepFinanceError` base in `src/data.py` so callers
can catch broadly when needed.
